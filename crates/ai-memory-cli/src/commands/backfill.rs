//! `ai-memory backfill` — one-time import of a project's pre-hook local history.
//!
//! Capture is forward-only: hooks record events only from the moment they are
//! installed. If you have worked in a project for weeks and only now install
//! ai-memory, the tool that exists for cross-session continuity starts
//! amnesiac about the very session you are resuming. This command fills that
//! first-boot gap: when the project's store is **brand new (empty)**, it
//! imports the existing local harness transcript history once.
//!
//! For each local native session it **replays the transcript through `/hook`**,
//! the same ingress live capture uses (as the companion importer does for
//! external conversations):
//!
//! - `export_transcript` (ai-memory-workstream) reads the harness's native
//!   transcript read-only into a normalized, bounded visible-event ledger,
//!   doing all the per-harness parsing in one place;
//! - each event is posted to `/hook/batch` as a session-start, `user-prompt`,
//!   or backfill-extension observation, attributed to the original harness and
//!   its native session id — so it becomes a real session + observations that
//!   consolidate into pages and are searchable via `memory_query`, exactly like
//!   live capture. The server **sanitizes and bounds every event**, so
//!   retroactive text crosses the same trust boundary as live capture.
//!
//! (An earlier revision imported through the managed-workstream begin/finish
//! endpoints, but that populates the `ai-memory run` continuity ledger, not the
//! searchable memory pipeline — verified by a live smoke where the observation
//! count stayed zero. `/hook` replay is what makes the history recall-able.)
//!
//! Safety: the automatic path only ever bootstraps an **empty** project (never
//! overwrites an established one — hook capture from install-time forward and
//! backfill of before-install history do not overlap), and it runs at most once
//! per checkout (a local sentinel).

use std::path::Path;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use ai_memory_core::{NewWorkstreamEvent, WorkstreamEventKind};
use ai_memory_hooks::capture_policy::metadata_only_body;
use ai_memory_workstream::{
    LaunchRoots, ManagedHarness, build_launch_plan_with_env, export_transcript,
    list_native_sessions, wait_for_transcript_flush,
};

use super::doctor::{SCANNED_HARNESSES, relocated_session_dir};
use super::run;
use super::transcript_capture::{EventCapture, TranscriptCapture};
use crate::config::Config;
use crate::http_client::{ServerEndpoint, get_json, post_json};

/// Circuit breaker on total imported events across the whole bootstrap. Once
/// reached, no further sessions are started; the remaining sessions are
/// reported as skipped rather than partially imported.
const MAX_EVENTS_TOTAL: usize = 50_000;

/// Newest local sessions to enumerate per harness before applying `--max-sessions`.
const PER_HARNESS_SCAN_LIMIT: usize = 200;

/// One `POST /hook/batch` request carries at most this many events — matches the
/// server's `MAX_HOOK_BATCH_ITEMS` so a batch is never rejected for size.
const HOOK_BATCH_ITEMS: usize = 256;

/// The extension label backfilled non-user events (assistant/tool) are recorded
/// under, mirroring the companion importer's replay-through-`/hook` mechanism.
const BACKFILL_EXTENSION: &str = "ai-memory-backfill";

/// A local native session eligible for import.
#[derive(Debug, Clone)]
pub(crate) struct SessionRef {
    pub(crate) harness: ManagedHarness,
    pub(crate) native_session_id: String,
    pub(crate) updated_at: SystemTime,
}

/// The outcome of a backfill run, and the JSON output shape.
#[derive(Debug, Default, Serialize)]
struct BackfillReport {
    workspace: String,
    project: String,
    /// Sessions selected for import (after `--max-sessions`).
    selected: usize,
    /// Sessions actually imported.
    imported_sessions: usize,
    /// Events imported across all sessions.
    imported_events: usize,
    /// Sessions skipped because the total-events circuit breaker tripped.
    skipped_for_cap: usize,
    /// Per-session failures (import errors); the run continues past them.
    failed_sessions: usize,
    /// True when the store was already populated (or `--force` off) and nothing
    /// was imported.
    skipped_non_empty: bool,
    /// True when this install's capture mode admits nothing from this
    /// checkout (allowlist mode without a marker), so nothing was imported.
    skipped_not_admitted: bool,
    /// True for a `--dry-run` (planning only, no import).
    dry_run: bool,
}

/// The captured side: `GET /admin/sessions/by-agent` — summing its counts tells
/// us whether the project store is empty.
#[derive(Debug, Deserialize)]
struct ByAgentResponse {
    by_agent: Vec<AgentCount>,
}

#[derive(Debug, Deserialize)]
struct AgentCount {
    #[allow(dead_code)]
    agent: String,
    sessions: u64,
}

/// Newest-first, then cap to `max_sessions`. Pure, so the selection/cap policy
/// is unit-tested without a server.
pub(crate) fn select_sessions(mut all: Vec<SessionRef>, max_sessions: usize) -> Vec<SessionRef> {
    all.sort_by(|a, b| {
        b.updated_at
            .cmp(&a.updated_at)
            .then_with(|| a.native_session_id.cmp(&b.native_session_id))
    });
    all.truncate(max_sessions);
    all
}

/// Filename under `<data_dir>/backfill-state/` marking that the automatic
/// backfill has already been attempted for this checkout on this machine, so
/// the SessionStart trigger does not re-spawn on every boot. Keyed by a hash of
/// the working directory — which both the hook (before it spawns) and the
/// worker (which inherits the hook's cwd) compute identically without resolving
/// scope — so arbitrary paths stay path-safe.
pub(crate) fn sentinel_path(data_dir: &Path, cwd: &Path) -> PathBuf {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(cwd.as_os_str().as_encoded_bytes());
    let digest = hasher.finalize();
    data_dir.join("backfill-state").join(format!("{digest:x}"))
}

fn write_sentinel(data_dir: &Path, cwd: &Path) {
    let path = sentinel_path(data_dir, cwd);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Best-effort: a missing sentinel only means a redundant future probe, not
    // incorrect data, so a write failure is not worth failing the import over.
    let _ = std::fs::write(&path, b"");
}

/// The server this backfill delivers to.
///
/// A SessionStart-spawned run names the hook's own target: a server profile
/// (#992), whose stored token is the only credential it will present, or the
/// install-time hook URL, authenticated exactly like the hook's own events to
/// it: the persisted hook token, then OIDC. The config/env bearer is never
/// used there, because it may belong to a different server than the one the
/// hook is installed against. A manual run keeps resolving from config.
async fn backfill_endpoint(
    config: &Config,
    args: &crate::cli::BackfillArgs,
) -> Result<ServerEndpoint> {
    if let Some(raw) = args.server_profile.as_deref() {
        let name = crate::server_profiles::ProfileName::parse(raw)
            .with_context(|| format!("`{raw}` is not a valid server profile name"))?;
        let profile = crate::server_profiles::lookup(&config.data_dir, &name)
            .map_err(|r| anyhow::anyhow!("server profile `{name}` was refused ({})", r.as_str()))?;
        return Ok(ServerEndpoint::for_hook_target(
            profile.url,
            Some(profile.token),
        ));
    }
    if let Some(url) = args.server_url.as_deref() {
        let static_token = crate::config::read_hook_auth_token(&config.data_dir);
        let token = super::hook_spool::resolve_bearer(
            &reqwest::Client::new(),
            &config.data_dir,
            static_token.as_deref(),
        )
        .await;
        return Ok(ServerEndpoint::for_hook_target(url.to_owned(), token));
    }
    Ok(ServerEndpoint::from_config_resolving_auth(config).await)
}

/// Run the backfill.
///
/// # Errors
/// Returns an error when the scope cannot be resolved, the working directory
/// cannot be read, the server is unreachable for the emptiness check, or any
/// selected session fails to import. The report is emitted before import errors
/// are returned, including in JSON mode.
pub async fn run(config: &Config, args: crate::cli::BackfillArgs) -> Result<()> {
    let cwd = std::env::current_dir().context("resolving the current working directory")?;
    run_in(config, args, cwd).await
}

/// [`run`] for the checkout at `cwd`.
async fn run_in(config: &Config, args: crate::cli::BackfillArgs, cwd: PathBuf) -> Result<()> {
    // The automatic (SessionStart-spawned) path honors the opt-out and records
    // that it has been attempted for this checkout, so it runs at most once per
    // machine regardless of outcome. Manual runs ignore both.
    if args.auto && !config.backfill_on_start {
        if !args.dry_run {
            write_sentinel(&config.data_dir, &cwd);
        }
        return Ok(());
    }

    let (workspace, project) =
        super::resolve_scope(config, args.workspace.as_deref(), args.project.as_deref())?;
    let home = run::native_home(config).context("locating the local harness session stores")?;
    let mut report = BackfillReport {
        workspace: workspace.clone(),
        project: project.clone(),
        dry_run: args.dry_run,
        ..BackfillReport::default()
    };

    // The live hook's admission gate and `[capture] ignore_paths`, applied to
    // the import as the hook applies them to live events. A checkout the hook
    // would emit nothing for (allowlist capture mode without a marker) imports
    // nothing, before the server is contacted at all.
    let capture = TranscriptCapture::for_checkout(&config.data_dir, &cwd);
    if !capture.admits_capture() {
        report.skipped_not_admitted = true;
        return finish(&args, &report);
    }
    let endpoint = backfill_endpoint(config, &args).await?;

    // Emptiness gate. Only bootstrap a brand-new store; an established project
    // is never retro-imported (that would duplicate hook-captured history).
    if !args.force {
        let empty = project_is_empty(&endpoint, &workspace, &project).await?;
        if !empty {
            report.skipped_non_empty = true;
            // Mark attempted so the auto-trigger stops probing this checkout.
            if !args.dry_run {
                write_sentinel(&config.data_dir, &cwd);
            }
            return finish(&args, &report);
        }
    }

    // Enumerate local sessions for this cwd across every supported harness.
    let (candidates, _limit_hit) =
        collect_local_sessions(&home, &cwd, args.session.as_deref()).await;
    let selected = select_sessions(candidates, args.max_sessions.max(1));
    report.selected = selected.len();

    if args.dry_run {
        return finish(&args, &report);
    }

    // Import each selected session, oldest-first so the imported history reads
    // chronologically, until the total-events circuit breaker trips.
    for session in selected.into_iter().rev() {
        if report.imported_events >= MAX_EVENTS_TOTAL {
            report.skipped_for_cap += 1;
            continue;
        }
        match import_one(
            &endpoint, &workspace, &project, &home, &cwd, &capture, &session,
        )
        .await
        {
            Ok(events) => {
                report.imported_sessions += 1;
                report.imported_events += events;
            }
            Err(error) => {
                report.failed_sessions += 1;
                // Quiet suppresses the success summary, not failures: the
                // detached worker's stderr is the operator's diagnostic log.
                eprintln!(
                    "ai-memory: backfill of {} session {} failed: {error:#}",
                    session.harness.as_str(),
                    display_id(&session.native_session_id)
                );
            }
        }
    }

    write_sentinel(&config.data_dir, &cwd);
    finish(&args, &report)?;
    if report.failed_sessions > 0 {
        bail!(
            "backfill failed to import {} of {} selected session(s)",
            report.failed_sessions,
            report.selected
        );
    }
    Ok(())
}

/// Sum the server's per-agent session counts for this scope; zero means empty.
async fn project_is_empty(
    endpoint: &ServerEndpoint,
    workspace: &str,
    project: &str,
) -> Result<bool> {
    match get_json::<ByAgentResponse>(
        endpoint,
        "/admin/sessions/by-agent",
        &[("workspace", workspace), ("project", project)],
    )
    .await
    {
        Ok(response) => Ok(response.by_agent.iter().map(|c| c.sessions).sum::<u64>() == 0),
        // A project that has never been written to does not exist server-side
        // yet, so the no-create lookup answers 404 — which is the strongest
        // possible "empty", and the common case for a first-time backfill.
        Err(error) if super::is_scope_not_found(&error) => Ok(true),
        // A multi-user server keeps `/admin/*` for the operator, so a
        // developer's own key is refused here. Ask the grant-checked web API
        // instead: it answers for the sessions this caller can see.
        Err(error) if is_forbidden(&error) => {
            project_is_empty_for_member(endpoint, workspace, project).await
        }
        Err(error) => Err(error).with_context(|| {
            format!("checking whether {workspace}/{project} already has captured sessions")
        }),
    }
}

fn is_forbidden(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<crate::http_client::ServerResponseError>()
        .is_some_and(|response| response.status() == reqwest::StatusCode::FORBIDDEN)
}

/// The sessions the web API lists for a project, open ones included.
#[derive(Debug, Deserialize)]
struct SessionListResponse {
    sessions: Vec<serde_json::Value>,
}

/// [`project_is_empty`] through `GET /api/v1/workspaces/{ws}/projects/{p}/sessions`,
/// which any member may read. Its 404 means the project does not exist (or is
/// not this caller's to see); either way the hook ingress, not this check,
/// decides whether an import may write there. A 404 whose body is not a JSON
/// error — a server without the web API — stays an error: guessing "empty"
/// there could import history twice.
async fn project_is_empty_for_member(
    endpoint: &ServerEndpoint,
    workspace: &str,
    project: &str,
) -> Result<bool> {
    let path = member_sessions_path(workspace, project)?;
    match get_json::<SessionListResponse>(
        endpoint,
        &path,
        &[("limit", "1"), ("include_open", "true")],
    )
    .await
    {
        Ok(response) => Ok(response.sessions.is_empty()),
        Err(error) if is_json_not_found(&error) => Ok(true),
        Err(error) => Err(error).with_context(|| {
            format!("checking whether {workspace}/{project} already has captured sessions")
        }),
    }
}

/// The path with each name percent-encoded as one segment, so a name can
/// never reach another route.
fn member_sessions_path(workspace: &str, project: &str) -> Result<String> {
    let mut url = reqwest::Url::parse("http://placeholder/")?;
    url.path_segments_mut()
        .map_err(|()| anyhow::anyhow!("placeholder URL cannot hold a path"))?
        .clear()
        .extend([
            "api",
            "v1",
            "workspaces",
            workspace,
            "projects",
            project,
            "sessions",
        ]);
    Ok(url.path().to_owned())
}

fn is_json_not_found(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<crate::http_client::ServerResponseError>()
        .is_some_and(|response| {
            response.status() == reqwest::StatusCode::NOT_FOUND
                && serde_json::from_str::<serde_json::Value>(response.body())
                    .is_ok_and(|body| body.get("error").is_some())
        })
}

/// Enumerate local native sessions for `cwd` across every scanned harness.
/// Read-only; a harness whose store is absent/unreadable contributes nothing.
///
/// Returns, alongside the sessions, the harnesses whose scan came back at
/// exactly [`PER_HARNESS_SCAN_LIMIT`] — a caller that cares about missing
/// older sessions (as opposed to `backfill`'s own newest-first + cap
/// selection, which does not) can surface that.
pub(crate) async fn collect_local_sessions(
    home: &Path,
    cwd: &Path,
    only_session: Option<&str>,
) -> (Vec<SessionRef>, Vec<ManagedHarness>) {
    collect_local_sessions_with(home, cwd, only_session, relocated_session_dir).await
}

/// [`collect_local_sessions`] with the relocation lookup passed in, for the
/// same reason as `doctor::scan_local_with`: tests pass `|_| None` so a
/// developer's `CLAUDE_CONFIG_DIR` cannot hide a fixture planted under a
/// temporary `$HOME`.
pub(crate) async fn collect_local_sessions_with(
    home: &Path,
    cwd: &Path,
    only_session: Option<&str>,
    session_dir_for: impl Fn(ManagedHarness) -> Option<PathBuf>,
) -> (Vec<SessionRef>, Vec<ManagedHarness>) {
    let mut out = Vec::new();
    let mut limit_hit = Vec::new();
    for &harness in SCANNED_HARNESSES {
        let session_dir = session_dir_for(harness);
        let Ok(sessions) = list_native_sessions(
            harness,
            home,
            cwd,
            session_dir.as_deref(),
            PER_HARNESS_SCAN_LIMIT,
        )
        .await
        else {
            continue;
        };
        if sessions.len() >= PER_HARNESS_SCAN_LIMIT {
            limit_hit.push(harness);
        }
        for session in sessions {
            if only_session.is_some_and(|want| want != session.native_session_id) {
                continue;
            }
            out.push(SessionRef {
                harness,
                native_session_id: session.native_session_id,
                updated_at: session.updated_at,
            });
        }
    }
    (out, limit_hit)
}

/// One item in a `POST /hook/batch` request: the full hook URL (whose query the
/// server parses for event/agent/scope/session) plus the JSON body.
#[derive(Debug, Serialize)]
struct HookItem {
    url: String,
    body: serde_json::Value,
}

/// The server's `/hook/batch` acknowledgement (subset we act on).
#[derive(Debug, Deserialize)]
struct HookBatchAck {
    /// Contiguous leading prefix committed, oldest-first.
    accepted: usize,
}

/// Import one native session by replaying its transcript through `/hook`, the
/// same ingress live capture uses — so the events become real sanitized
/// observations that consolidate into pages and are searchable, attributed to
/// the original harness and its native session id. Returns the number of
/// content events (messages/tools) imported. The server sanitizes and bounds
/// every event, so retroactive text crosses the same trust boundary as live
/// capture, and `capture` drops or strips tool events the checkout's
/// `[capture] ignore_paths` excludes before anything is posted.
async fn import_one(
    endpoint: &ServerEndpoint,
    workspace: &str,
    project: &str,
    home: &Path,
    cwd: &Path,
    capture: &TranscriptCapture,
    session: &SessionRef,
) -> Result<usize> {
    let roots = LaunchRoots { home, cwd };
    let session_dir =
        build_launch_plan_with_env(session.harness, None, Vec::new(), None, &[], Some(roots))
            .ok()
            .and_then(|plan| plan.session_dir);
    // These are historical sessions, so the flush wait is a quick no-op; ignore
    // its result and read whatever is on disk.
    let _ = wait_for_transcript_flush(
        session.harness,
        home,
        cwd,
        session_dir.as_deref(),
        &session.native_session_id,
    )
    .await;
    let transcript = export_transcript(
        session.harness,
        home,
        cwd,
        session_dir.as_deref(),
        &session.native_session_id,
        None,
    )
    .await
    .with_context(|| format!("reading the {} transcript", session.harness.as_str()))?;

    let sid = &session.native_session_id;
    let agent = session.harness.agent_kind().as_str();
    let resolved = resolve_occurred_at(&transcript.events);
    let mut items = Vec::with_capacity(transcript.events.len() + 2);
    items.push(hook_item(
        endpoint,
        workspace,
        project,
        agent,
        "session-start",
        sid,
        &format!("{sid}:session-start"),
        None,
        serde_json::json!({ "session_id": sid, "occurred_at": resolved.earliest }),
    )?);
    let mut content = 0usize;
    let captures = capture.decide(&transcript.events);
    for ((event, occurred_at), capture) in transcript
        .events
        .iter()
        .zip(&resolved.per_event)
        .zip(&captures)
    {
        if let Some(mapped) = map_event(sid, event, occurred_at.as_deref(), capture) {
            items.push(hook_item(
                endpoint,
                workspace,
                project,
                agent,
                &mapped.event,
                sid,
                &mapped.ingest_key,
                mapped.source_event.as_deref(),
                mapped.body,
            )?);
            content += 1;
        }
    }
    items.push(hook_item(
        endpoint,
        workspace,
        project,
        agent,
        "session-end",
        sid,
        &format!("{sid}:session-end"),
        None,
        serde_json::json!({ "session_id": sid, "occurred_at": resolved.latest }),
    )?);

    post_hook_items(endpoint, &items).await?;
    Ok(content)
}

/// Per-event `occurred_at` resolution for one transcript, plus the session's
/// overall boundary times.
struct ResolvedOccurredAt {
    /// Effective `occurred_at` for each event, in transcript order (RFC 3339).
    per_event: Vec<Option<String>>,
    /// The earliest valid event time — the session-start's `occurred_at`.
    /// A transcript is not guaranteed to be time-sorted (a reordered or
    /// clock-skewed import), so the first *entry* is not reliably the
    /// earliest *time*.
    earliest: Option<String>,
    /// The latest valid event time — the session-end's `occurred_at`, same
    /// reasoning as `earliest`.
    latest: Option<String>,
}

/// Resolve each event's effective `occurred_at`, letting one missing or
/// unparsable own timestamp inherit the nearest preceding *valid* one; an
/// event before the first valid timestamp inherits that first one instead of
/// staying unresolved (there is nothing earlier to inherit from). A value
/// that fails to parse as RFC 3339 is treated exactly like a missing one — it
/// never reaches the hook body, so a malformed transcript timestamp cannot
/// masquerade as a validated one downstream.
fn resolve_occurred_at(events: &[NewWorkstreamEvent]) -> ResolvedOccurredAt {
    let valid: Vec<Option<jiff::Timestamp>> = events
        .iter()
        .map(|event| {
            event
                .occurred_at
                .as_deref()
                .and_then(|s| s.parse::<jiff::Timestamp>().ok())
        })
        .collect();

    let mut per_event: Vec<Option<jiff::Timestamp>> = Vec::with_capacity(events.len());
    let mut last_valid: Option<jiff::Timestamp> = None;
    for ts in &valid {
        if ts.is_some() {
            last_valid = *ts;
        }
        per_event.push(last_valid);
    }
    // Backward-fill the leading gap: events before the first valid timestamp
    // had nothing preceding them to inherit above.
    if let Some(first_valid) = valid.iter().copied().flatten().next() {
        for slot in per_event.iter_mut() {
            match slot {
                Some(_) => break,
                None => *slot = Some(first_valid),
            }
        }
    }

    let (earliest, latest) = valid.into_iter().flatten().fold(
        (None, None),
        |(min, max): (Option<jiff::Timestamp>, Option<jiff::Timestamp>), ts| {
            (
                Some(min.map_or(ts, |m| m.min(ts))),
                Some(max.map_or(ts, |m| m.max(ts))),
            )
        },
    );

    ResolvedOccurredAt {
        per_event: per_event
            .into_iter()
            .map(|ts| ts.map(|t| t.to_string()))
            .collect(),
        earliest: earliest.map(|t| t.to_string()),
        latest: latest.map(|t| t.to_string()),
    }
}

/// A transcript event mapped to its `/hook` shape.
struct MappedEvent {
    event: String,
    source_event: Option<String>,
    ingest_key: String,
    body: serde_json::Value,
}

/// Map one transcript event to a hook event. User messages become the canonical
/// `user-prompt` observation; every other content-bearing event is recorded as
/// a backfill extension observation. Non-content boundary events (compaction,
/// checkpoint, annotation) are dropped — they are not session content.
///
/// `occurred_at` (RFC 3339), already resolved by the caller, rides in the hook
/// body so the imported observation is dated at the transcript's own event
/// time rather than at import time.
///
/// `capture` is the checkout's capture decision for this event: a dropped
/// event maps to nothing, and a metadata-only one carries the hook's
/// metadata-only body with its `_ai_memory_capture` protocol, which the server
/// validates and rebuilds from its allowlist exactly as for a live hook. A kept
/// event carries no protocol: its body is a normalized transcript record, not
/// the native tool payload the server's re-inspection reads, so a claim would
/// be judged against the wrong schema and stripped; the server never receives
/// the ignore patterns, so the client's decision is the only one possible.
/// An unrecognized tool's event carries only its family, as the live server
/// stores for an unknown-family tool, never its arguments or output.
fn map_event(
    session_id: &str,
    event: &NewWorkstreamEvent,
    occurred_at: Option<&str>,
    capture: &EventCapture,
) -> Option<MappedEvent> {
    let content = event.content.trim();
    if content.is_empty() {
        return None;
    }
    let decision = match capture {
        EventCapture::Drop => return None,
        EventCapture::MetadataOnly(decision) => Some(decision),
        EventCapture::Keep | EventCapture::Unrecognized => None,
    };
    let role = event.role.as_deref().unwrap_or("");
    let ingest_key = format!("{session_id}:{}", event.event_id);
    match event.kind {
        WorkstreamEventKind::Message if role == "user" || role == "human" => Some(MappedEvent {
            event: "user-prompt".to_string(),
            source_event: None,
            ingest_key,
            body: serde_json::json!({
                "session_id": session_id,
                "prompt": event.content,
                "occurred_at": occurred_at,
            }),
        }),
        WorkstreamEventKind::Message
        | WorkstreamEventKind::ToolCall
        | WorkstreamEventKind::ToolResult => {
            let source_event = match event.kind {
                WorkstreamEventKind::ToolCall => "tool_call".to_string(),
                WorkstreamEventKind::ToolResult => "tool_result".to_string(),
                _ if role.is_empty() => "message".to_string(),
                _ => format!("{role}-message"),
            };
            let body = match decision {
                Some(decision) => {
                    let mut body = metadata_only_body(Some(session_id), None, decision);
                    if let Some(object) = body.as_object_mut() {
                        object.insert("occurred_at".into(), occurred_at.into());
                    }
                    body
                }
                None if matches!(capture, EventCapture::Unrecognized) => {
                    let outcome = if event.kind == WorkstreamEventKind::ToolResult {
                        "\noutcome: unknown"
                    } else {
                        ""
                    };
                    serde_json::json!({
                        "session_id": session_id,
                        "title": "tool unknown",
                        "message": format!("tool_family: unknown{outcome}"),
                        "occurred_at": occurred_at,
                    })
                }
                None => serde_json::json!({
                    "session_id": session_id,
                    "title": first_line(content),
                    "message": event.content,
                    "occurred_at": occurred_at,
                }),
            };
            Some(MappedEvent {
                event: format!("backfill.{source_event}"),
                source_event: Some(source_event),
                ingest_key,
                body,
            })
        }
        WorkstreamEventKind::Compaction
        | WorkstreamEventKind::Checkpoint
        | WorkstreamEventKind::Annotation => None,
    }
}

/// A short one-line title from the event content.
fn first_line(content: &str) -> String {
    let line = content.lines().next().unwrap_or("").trim();
    let capped: String = line.chars().take(120).collect();
    if capped.is_empty() {
        "(backfilled event)".to_string()
    } else {
        capped
    }
}

/// Build a `/hook` URL whose query the server parses for event/agent/scope. The
/// URL is data the batch item carries, not the request target (that is
/// `/hook/batch`), so it is built against this server's origin exactly as a live
/// hook would have spooled it.
#[allow(clippy::too_many_arguments)]
fn hook_item(
    endpoint: &ServerEndpoint,
    workspace: &str,
    project: &str,
    agent: &str,
    event: &str,
    session_id: &str,
    ingest_key: &str,
    source_event: Option<&str>,
    body: serde_json::Value,
) -> Result<HookItem> {
    let mut url = reqwest::Url::parse(&endpoint.build_url("/hook"))
        .context("building the hook URL for backfill")?;
    {
        let mut query = url.query_pairs_mut();
        query
            .append_pair("event", event)
            .append_pair("agent", agent)
            .append_pair("workspace", workspace)
            .append_pair("project", project)
            .append_pair("project_src", "marker")
            .append_pair("session_id", session_id)
            .append_pair("ingest_key", ingest_key);
        if let Some(source_event) = source_event {
            query
                .append_pair("extension", BACKFILL_EXTENSION)
                .append_pair("source_event", source_event);
        }
    }
    Ok(HookItem {
        url: url.into(),
        body,
    })
}

/// POST the events in server-sized batches, retrying the unaccepted suffix of a
/// partially accepted batch (per-source rate limiting can skip a tail) before
/// failing. Events are oldest-first, so a leading-prefix ack is safe to resume.
async fn post_hook_items(endpoint: &ServerEndpoint, items: &[HookItem]) -> Result<()> {
    for chunk in items.chunks(HOOK_BATCH_ITEMS) {
        let mut offset = 0usize;
        let mut stalled = 0u32;
        while offset < chunk.len() {
            let batch: Vec<&HookItem> = chunk[offset..].iter().collect();
            let ack: HookBatchAck = post_json(endpoint, "/hook/batch", &batch)
                .await
                .context("replaying transcript events through /hook/batch")?;
            if ack.accepted == 0 {
                stalled += 1;
                if stalled >= 5 {
                    bail!(
                        "server accepted none of a hook batch after {stalled} attempts \
                         (rate limited or saturated); rerun to resume"
                    );
                }
                tokio::time::sleep(Duration::from_millis(200 * u64::from(stalled))).await;
                continue;
            }
            stalled = 0;
            offset += ack.accepted.min(chunk.len() - offset);
        }
    }
    Ok(())
}

fn finish(args: &crate::cli::BackfillArgs, report: &BackfillReport) -> Result<()> {
    if args.json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }
    if args.quiet {
        return Ok(());
    }
    if report.skipped_not_admitted {
        println!(
            "ai-memory: capture mode is allowlist and this checkout has no .ai-memory.toml marker opting in (or its marker is unreadable) — nothing to backfill for {}/{}.",
            report.workspace, report.project
        );
        return Ok(());
    }
    if report.skipped_non_empty {
        println!(
            "ai-memory: {}/{} already has captured sessions — nothing to backfill (use --force to import anyway).",
            report.workspace, report.project
        );
        return Ok(());
    }
    if report.dry_run {
        println!(
            "ai-memory: would import {} local session(s) into {}/{} (dry run).",
            report.selected, report.workspace, report.project
        );
        return Ok(());
    }
    if report.imported_sessions == 0 && report.failed_sessions == 0 {
        println!(
            "ai-memory: no local session history found to backfill for {}/{}.",
            report.workspace, report.project
        );
        return Ok(());
    }
    let mut line = format!(
        "📼 ai-memory imported {} prior local session(s) (~{} events) for {}/{}.",
        report.imported_sessions, report.imported_events, report.workspace, report.project
    );
    if report.failed_sessions > 0 {
        line.push_str(&format!(" {} session(s) failed.", report.failed_sessions));
    }
    if report.skipped_for_cap > 0 {
        line.push_str(&format!(
            " {} older session(s) skipped (import cap).",
            report.skipped_for_cap
        ));
    }
    line.push_str(" Opt out with AI_MEMORY_BACKFILL_ON_START=false.");
    println!("{line}");
    Ok(())
}

fn display_id(id: &str) -> &str {
    id.get(..8).unwrap_or(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn session(id: &str, secs_ago: u64) -> SessionRef {
        SessionRef {
            harness: ManagedHarness::Claude,
            native_session_id: id.to_string(),
            updated_at: SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000 - secs_ago),
        }
    }

    #[test]
    fn select_sessions_keeps_the_newest_up_to_the_cap() {
        let all = vec![session("old", 300), session("new", 10), session("mid", 100)];
        let picked = select_sessions(all, 2);
        assert_eq!(
            picked
                .iter()
                .map(|s| s.native_session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["new", "mid"],
            "newest first, capped to 2"
        );
    }

    #[test]
    fn member_sessions_path_encodes_each_name_as_one_segment() {
        assert_eq!(
            member_sessions_path("team", "app").unwrap(),
            "/api/v1/workspaces/team/projects/app/sessions"
        );
        assert_eq!(
            member_sessions_path("a/b", "c d?").unwrap(),
            "/api/v1/workspaces/a%2Fb/projects/c%20d%3F/sessions"
        );
    }

    #[test]
    fn select_sessions_cap_zero_is_raised_to_one_by_caller_contract() {
        // The caller passes max_sessions.max(1); select_sessions itself honors
        // whatever it is given, so a literal 0 yields nothing.
        assert!(select_sessions(vec![session("a", 1)], 0).is_empty());
        assert_eq!(select_sessions(vec![session("a", 1)], 1).len(), 1);
    }

    #[test]
    fn sentinel_path_is_cwd_specific_and_path_safe() {
        let dir = Path::new("/data");
        let a = sentinel_path(dir, Path::new("/home/me/projects/app"));
        let b = sentinel_path(dir, Path::new("/home/me/projects/other"));
        let c = sentinel_path(dir, Path::new("/home/me/projects/app"));
        assert_eq!(a, c, "same cwd -> same path");
        assert_ne!(a, b, "different cwd -> different path");
        assert!(
            a.starts_with("/data/backfill-state/"),
            "under the data dir: {a:?}"
        );
        // A path with separators hashes to a single flat, path-safe leaf.
        assert_eq!(
            a.strip_prefix("/data/backfill-state/")
                .unwrap()
                .components()
                .count(),
            1,
            "the hashed leaf must be a single path component: {a:?}"
        );
    }

    fn event(kind: WorkstreamEventKind, role: Option<&str>, content: &str) -> NewWorkstreamEvent {
        NewWorkstreamEvent {
            event_id: "evt-1".to_string(),
            agent: ai_memory_core::AgentKind::ClaudeCode,
            native_session_id: "sid".to_string(),
            source_record_id: None,
            kind,
            role: role.map(str::to_string),
            content: content.to_string(),
            occurred_at: None,
            metadata: serde_json::Value::Null,
        }
    }

    #[test]
    fn user_message_maps_to_the_canonical_user_prompt() {
        let m = map_event(
            "sid",
            &event(WorkstreamEventKind::Message, Some("user"), "do the thing"),
            Some("2026-09-10T12:00:00Z"),
            &EventCapture::Keep,
        )
        .expect("user message maps");
        assert_eq!(m.event, "user-prompt");
        assert!(
            m.source_event.is_none(),
            "user-prompt is a lifecycle event, not an extension"
        );
        assert_eq!(m.body["prompt"], "do the thing");
        assert_eq!(m.body["occurred_at"], "2026-09-10T12:00:00Z");
        assert_eq!(
            m.ingest_key, "sid:evt-1",
            "ingest key is stable per source event"
        );
    }

    #[test]
    fn assistant_and_tool_events_map_to_backfill_extension_observations() {
        let a = map_event(
            "sid",
            &event(
                WorkstreamEventKind::Message,
                Some("assistant"),
                "here is the plan\nline2",
            ),
            None,
            &EventCapture::Keep,
        )
        .expect("assistant maps");
        assert_eq!(a.event, "backfill.assistant-message");
        assert_eq!(a.source_event.as_deref(), Some("assistant-message"));
        assert_eq!(
            a.body["title"], "here is the plan",
            "title is the first line"
        );
        assert_eq!(a.body["message"], "here is the plan\nline2");
        assert!(
            a.body["occurred_at"].is_null(),
            "no occurred_at was supplied"
        );

        let t = map_event(
            "sid",
            &event(WorkstreamEventKind::ToolCall, None, "grep foo"),
            None,
            &EventCapture::Keep,
        )
        .expect("tool call maps");
        assert_eq!(t.event, "backfill.tool_call");
        assert_eq!(t.source_event.as_deref(), Some("tool_call"));
    }

    /// Round-trips a literal RFC 3339 string through `jiff::Timestamp` so
    /// expectations match `resolve_occurred_at`'s own parse-then-format
    /// output rather than assuming it echoes the input string verbatim.
    fn ts(literal: &str) -> String {
        literal.parse::<jiff::Timestamp>().unwrap().to_string()
    }

    #[test]
    fn resolve_occurred_at_fills_gaps_from_the_preceding_event() {
        let mut e1 = event(WorkstreamEventKind::Message, Some("user"), "one");
        e1.occurred_at = Some("2026-09-10T12:00:00Z".to_string());
        let e2 = event(WorkstreamEventKind::Message, Some("assistant"), "two"); // no timestamp
        let mut e3 = event(WorkstreamEventKind::ToolCall, None, "three");
        e3.occurred_at = Some("2026-09-10T12:05:00Z".to_string());
        let e4 = event(WorkstreamEventKind::ToolResult, None, "four"); // no timestamp

        let resolved = resolve_occurred_at(&[e1, e2, e3, e4]);
        assert_eq!(
            resolved.per_event,
            vec![
                Some(ts("2026-09-10T12:00:00Z")),
                Some(ts("2026-09-10T12:00:00Z")),
                Some(ts("2026-09-10T12:05:00Z")),
                Some(ts("2026-09-10T12:05:00Z")),
            ],
            "an event without its own timestamp inherits the nearest preceding one"
        );
        assert_eq!(resolved.earliest, Some(ts("2026-09-10T12:00:00Z")));
        assert_eq!(resolved.latest, Some(ts("2026-09-10T12:05:00Z")));
    }

    #[test]
    fn resolve_occurred_at_backfills_the_leading_gap_from_the_first_valid_timestamp() {
        let e1 = event(WorkstreamEventKind::Message, Some("user"), "one"); // no timestamp
        let e2 = event(WorkstreamEventKind::Message, Some("assistant"), "two"); // no timestamp
        let mut e3 = event(WorkstreamEventKind::ToolCall, None, "three");
        e3.occurred_at = Some("2026-09-10T12:05:00Z".to_string());

        let resolved = resolve_occurred_at(&[e1, e2, e3]);
        assert_eq!(
            resolved.per_event,
            vec![
                Some(ts("2026-09-10T12:05:00Z")),
                Some(ts("2026-09-10T12:05:00Z")),
                Some(ts("2026-09-10T12:05:00Z")),
            ],
            "events before the first valid timestamp inherit it backward, \
             not just the ones after"
        );
    }

    #[test]
    fn resolve_occurred_at_treats_an_unparsable_timestamp_as_missing() {
        let mut e1 = event(WorkstreamEventKind::Message, Some("user"), "one");
        e1.occurred_at = Some("2026-09-10T12:00:00Z".to_string());
        let mut e2 = event(WorkstreamEventKind::Message, Some("assistant"), "two");
        e2.occurred_at = Some("not-a-timestamp".to_string());

        let resolved = resolve_occurred_at(&[e1, e2]);
        assert_eq!(
            resolved.per_event,
            vec![
                Some(ts("2026-09-10T12:00:00Z")),
                Some(ts("2026-09-10T12:00:00Z"))
            ],
            "an unparsable timestamp must not reach the hook body; the \
             preceding valid one is inherited instead"
        );
        assert_eq!(resolved.latest, Some(ts("2026-09-10T12:00:00Z")));
    }

    #[test]
    fn resolve_occurred_at_uses_min_and_max_not_first_and_last_when_out_of_order() {
        // A transcript is not guaranteed to be time-sorted (clock skew,
        // reordering); the session boundary must reflect the actual extremes,
        // not just the first/last entries.
        let mut e1 = event(WorkstreamEventKind::Message, Some("user"), "one");
        e1.occurred_at = Some("2026-09-10T12:05:00Z".to_string());
        let mut e2 = event(WorkstreamEventKind::Message, Some("assistant"), "two");
        e2.occurred_at = Some("2026-09-10T12:00:00Z".to_string());

        let resolved = resolve_occurred_at(&[e1, e2]);
        assert_eq!(
            resolved.earliest,
            Some(ts("2026-09-10T12:00:00Z")),
            "earliest must be the minimum valid time, not the first entry"
        );
        assert_eq!(
            resolved.latest,
            Some(ts("2026-09-10T12:05:00Z")),
            "latest must be the maximum valid time, not the last entry"
        );
    }

    #[test]
    fn resolve_occurred_at_stays_none_when_nothing_has_a_timestamp() {
        let events = vec![
            event(WorkstreamEventKind::Message, Some("user"), "one"),
            event(WorkstreamEventKind::Message, Some("assistant"), "two"),
        ];
        let resolved = resolve_occurred_at(&events);
        assert_eq!(resolved.per_event, vec![None, None]);
        assert_eq!(resolved.earliest, None);
        assert_eq!(resolved.latest, None);
    }

    #[test]
    fn empty_and_boundary_events_are_dropped() {
        assert!(
            map_event(
                "sid",
                &event(WorkstreamEventKind::Message, Some("user"), "   "),
                None,
                &EventCapture::Keep,
            )
            .is_none(),
            "whitespace-only content is not an observation"
        );
        for kind in [
            WorkstreamEventKind::Compaction,
            WorkstreamEventKind::Checkpoint,
            WorkstreamEventKind::Annotation,
        ] {
            assert!(
                map_event(
                    "sid",
                    &event(kind, None, "boundary"),
                    None,
                    &EventCapture::Keep
                )
                .is_none(),
                "{kind:?} is not session content"
            );
        }
    }

    /// End-to-end proof that `collect_local_sessions` wires to the real
    /// workstream path encoder: a Claude transcript planted under the actual
    /// `~/.claude/projects/<enc-cwd>/` layout for this cwd is discovered, and a
    /// transcript for a different cwd is ignored.
    ///
    /// Unix-gated: the fixture uses a POSIX-encoded projects path; cross-platform
    /// native-store discovery is owned and tested by `ai-memory-workstream`. The
    /// pure selection/cap tests above run on every platform.
    #[cfg(unix)]
    #[tokio::test]
    async fn collect_local_sessions_finds_a_planted_claude_session_for_this_cwd() {
        let home = tempfile::tempdir().unwrap();
        let cwd = tempfile::tempdir().unwrap();
        let other_cwd = tempfile::tempdir().unwrap();

        let session_dir = home
            .path()
            .join(".claude")
            .join("projects")
            .join(cwd.path().to_string_lossy().replace('/', "-"));
        std::fs::create_dir_all(&session_dir).unwrap();
        let header = serde_json::json!({
            "sessionId": "11111111-2222-3333-4444-555555555555",
            "cwd": cwd.path().to_string_lossy(),
        });
        std::fs::write(session_dir.join("sess.jsonl"), format!("{header}\n")).unwrap();
        let foreign = serde_json::json!({
            "sessionId": "99999999-8888-7777-6666-555555555555",
            "cwd": other_cwd.path().to_string_lossy(),
        });
        std::fs::write(session_dir.join("foreign.jsonl"), format!("{foreign}\n")).unwrap();

        let (found, _limit_hit) =
            collect_local_sessions_with(home.path(), cwd.path(), None, |_| None).await;
        let claude: Vec<_> = found
            .iter()
            .filter(|s| s.harness == ManagedHarness::Claude)
            .collect();
        assert_eq!(claude.len(), 1, "only the matching-cwd session: {found:?}");
        assert_eq!(
            claude[0].native_session_id,
            "11111111-2222-3333-4444-555555555555"
        );

        // `--session` narrows to one id.
        let (only, _limit_hit) =
            collect_local_sessions_with(home.path(), cwd.path(), Some("nope"), |_| None).await;
        assert!(only.is_empty(), "no session matches the filter: {only:?}");
    }

    /// GHSA-gcpc-88hm-3xqr: a planted Claude transcript reads one file under
    /// the checkout's `[capture] ignore_paths` and one outside it. The
    /// excluded read's call and output never reach `/hook/batch`; the public
    /// read's do. A malformed read is posted as the hook's metadata-only body.
    #[cfg(unix)]
    #[tokio::test]
    async fn backfill_never_posts_tool_content_the_capture_policy_excludes() {
        use axum::{Router, routing::post};
        use std::sync::{Arc, Mutex};

        let posted = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let captured = posted.clone();
        let app = Router::new().route(
            "/hook/batch",
            post(
                move |axum::Json(items): axum::Json<Vec<serde_json::Value>>| {
                    let captured = captured.clone();
                    async move {
                        let accepted = items.len();
                        captured.lock().unwrap().extend(items);
                        axum::Json(serde_json::json!({ "accepted": accepted }))
                    }
                },
            ),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let home = tempfile::tempdir().unwrap();
        let repo = tempfile::tempdir().unwrap();
        std::fs::write(
            repo.path().join(".ai-memory.toml"),
            "[capture]\nignore_paths = [\"docs/adr/**\"]\n",
        )
        .unwrap();
        let session_dir = home
            .path()
            .join(".claude")
            .join("projects")
            .join(repo.path().to_string_lossy().replace('/', "-"));
        std::fs::create_dir_all(&session_dir).unwrap();
        let sid = "11111111-2222-3333-4444-555555555555";
        let cwd = repo.path().to_string_lossy();
        let record = |uuid: &str, kind: &str, content: serde_json::Value| {
            serde_json::json!({
                "type": kind, "uuid": uuid, "sessionId": sid, "cwd": cwd,
                "timestamp": "2026-09-01T10:00:00Z",
                "message": {"role": kind, "content": content},
            })
            .to_string()
        };
        let read = |id: &str, input: serde_json::Value| serde_json::json!({"type": "tool_use", "id": id, "name": "Read", "input": input});
        let output = |id: &str, text: &str| serde_json::json!({"type": "tool_result", "tool_use_id": id, "content": text});
        let lines = [
            record("u1", "user", serde_json::json!("read the adr and lib")),
            record(
                "a1",
                "assistant",
                serde_json::json!([
                    read(
                        "toolu_adr",
                        serde_json::json!({"file_path": repo.path().join("docs/adr/1.md")})
                    ),
                    read(
                        "toolu_pub",
                        serde_json::json!({"file_path": repo.path().join("src/lib.rs")})
                    ),
                    read(
                        "toolu_bad",
                        serde_json::json!({"file_path": 7, "note": "ARGSENTINEL"})
                    ),
                ]),
            ),
            record(
                "u2",
                "user",
                serde_json::json!([
                    output("toolu_adr", "ADRSENTINEL private decision"),
                    output("toolu_pub", "PUBLICCONTROL public code"),
                    output("toolu_bad", "BADSENTINEL unprovable output"),
                ]),
            ),
        ];
        std::fs::write(
            session_dir.join(format!("{sid}.jsonl")),
            lines.join("\n") + "\n",
        )
        .unwrap();

        let endpoint = ServerEndpoint::for_hook_target(format!("http://{address}"), None);
        let session = SessionRef {
            harness: ManagedHarness::Claude,
            native_session_id: sid.into(),
            updated_at: SystemTime::now(),
        };
        let capture = TranscriptCapture::for_checkout(home.path(), repo.path());
        import_one(
            &endpoint,
            "ws",
            "proj",
            home.path(),
            repo.path(),
            &capture,
            &session,
        )
        .await
        .unwrap();
        server.abort();

        let posted = posted.lock().unwrap();
        let text = serde_json::to_string(&*posted).unwrap();
        for sentinel in ["ADRSENTINEL", "docs/adr", "ARGSENTINEL", "BADSENTINEL"] {
            assert!(!text.contains(sentinel), "{sentinel} was posted: {text}");
        }
        assert!(text.contains("PUBLICCONTROL"), "public output: {text}");
        assert!(text.contains("src/lib.rs"), "public call: {text}");
        let stripped: Vec<_> = posted
            .iter()
            .filter(|item| item["body"].get("_ai_memory_capture").is_some())
            .collect();
        assert_eq!(
            stripped.len(),
            2,
            "the malformed call and its output: {text}"
        );
        for item in stripped {
            assert_eq!(
                item["body"]["_ai_memory_capture"]["disposition"],
                "metadata-only"
            );
            assert_eq!(item["body"]["tool_family"], "file");
            assert_eq!(item["body"]["tool_call_id"], "toolu_bad");
        }
    }

    /// Allowlist capture mode (#446) makes an unmarked checkout emit nothing
    /// from the live hook, so a manual backfill of it must not contact the
    /// server at all. Controls: the same checkout with a marker, and the same
    /// unmarked checkout under the default denylist mode, both proceed to the
    /// server.
    #[tokio::test]
    async fn allowlist_mode_backfills_nothing_from_an_unmarked_checkout() {
        use axum::{Router, extract::Request};
        use std::sync::{Arc, Mutex};

        let requests = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = requests.clone();
        let app = Router::new().fallback(move |request: Request| {
            let seen = seen.clone();
            async move {
                seen.lock().unwrap().push(request.uri().path().to_owned());
                axum::Json(serde_json::json!({ "by_agent": [], "accepted": 0 }))
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let attempt = |mode: Option<&'static str>, marked: bool| {
            let requests = requests.clone();
            async move {
                let home = tempfile::tempdir().unwrap();
                let data_dir = tempfile::tempdir().unwrap();
                let repo = tempfile::tempdir().unwrap();
                if let Some(mode) = mode {
                    std::fs::write(
                        data_dir.path().join(super::super::hook::CAPTURE_MODE_FILE),
                        mode,
                    )
                    .unwrap();
                }
                if marked {
                    std::fs::write(repo.path().join(".ai-memory.toml"), "[capture]\n").unwrap();
                }
                let mut config =
                    crate::config::Config::load(None, Some(home.path().to_path_buf())).unwrap();
                config.data_dir = data_dir.path().to_path_buf();
                config.home_dir = Some(home.path().to_string_lossy().into_owned());
                config.server_url = format!("http://{address}");
                config.auth.bearer_token = None;
                let mut args = spawned_args(None, None);
                args.auto = false;
                args.workspace = Some("ws".into());
                args.project = Some("proj".into());
                requests.lock().unwrap().clear();
                run_in(&config, args, repo.path().to_path_buf())
                    .await
                    .unwrap();
                requests.lock().unwrap().clone()
            }
        };

        assert!(
            attempt(Some("allowlist\n"), false).await.is_empty(),
            "allowlist + unmarked checkout must not reach the server"
        );
        assert!(
            !attempt(Some("allowlist\n"), true).await.is_empty(),
            "control: a marked checkout proceeds under allowlist"
        );
        assert!(
            !attempt(None, false).await.is_empty(),
            "control: denylist (default) is unchanged"
        );
        server.abort();
    }

    /// Plant `lines` as one native session and backfill it from `repo`,
    /// returning everything posted to `/hook/batch` as one JSON string.
    #[cfg(unix)]
    async fn backfill_posted(
        harness: ManagedHarness,
        home: &Path,
        data_dir: &Path,
        repo: &Path,
        sid: &str,
        lines: &[serde_json::Value],
    ) -> String {
        use axum::{Router, routing::post};
        use std::sync::{Arc, Mutex};

        let session_file = match harness {
            ManagedHarness::Claude => home
                .join(".claude/projects")
                .join(repo.to_string_lossy().replace('/', "-"))
                .join(format!("{sid}.jsonl")),
            ManagedHarness::Codex => home
                .join(".codex/sessions/2026/09/01")
                .join(format!("rollout-2026-09-01T10-00-00-{sid}.jsonl")),
            other => panic!("no fixture layout for {}", other.as_str()),
        };
        std::fs::create_dir_all(session_file.parent().unwrap()).unwrap();
        let body: String = lines.iter().map(|line| format!("{line}\n")).collect();
        std::fs::write(&session_file, body).unwrap();

        let posted = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let captured = posted.clone();
        let app = Router::new().route(
            "/hook/batch",
            post(
                move |axum::Json(items): axum::Json<Vec<serde_json::Value>>| {
                    let captured = captured.clone();
                    async move {
                        let accepted = items.len();
                        captured.lock().unwrap().extend(items);
                        axum::Json(serde_json::json!({ "accepted": accepted }))
                    }
                },
            ),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let endpoint = ServerEndpoint::for_hook_target(format!("http://{address}"), None);
        let session = SessionRef {
            harness,
            native_session_id: sid.into(),
            updated_at: SystemTime::now(),
        };
        let capture = TranscriptCapture::for_checkout(data_dir, repo);
        import_one(&endpoint, "ws", "proj", home, repo, &capture, &session)
            .await
            .unwrap();
        server.abort();
        serde_json::to_string(&*posted.lock().unwrap()).unwrap()
    }

    #[cfg(unix)]
    fn marked_checkout(marker: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".ai-memory.toml"), marker).unwrap();
        dir
    }

    /// Codex records its shell tool as `exec_command` with `{cmd, workdir}`;
    /// the live hook receives it as `Bash`. A command whose `workdir`-relative
    /// argument names an ignored path never reaches `/hook/batch`, call or
    /// output; the same command run from a public directory does.
    #[cfg(unix)]
    #[tokio::test]
    async fn backfill_inspects_codex_exec_command_as_the_hook_shell_call() {
        let home = tempfile::tempdir().unwrap();
        let repo = marked_checkout("[capture]\nignore_paths = [\"docs/adr/**\"]\n");
        for dir in ["docs/adr", "src/adr"] {
            std::fs::create_dir_all(repo.path().join(dir)).unwrap();
        }
        let sid = "019a0000-0000-7000-8000-000000000001";
        let item = |payload: serde_json::Value| {
            serde_json::json!({
                "type": "response_item", "timestamp": "2026-09-01T10:00:00Z", "payload": payload,
            })
        };
        let exec = |id: &str, workdir: &str| {
            item(serde_json::json!({
                "type": "function_call", "name": "exec_command", "call_id": id,
                "arguments": serde_json::json!({
                    "cmd": "cat adr/x.md", "workdir": repo.path().join(workdir),
                }).to_string(),
            }))
        };
        let output = |id: &str, text: &str| {
            item(serde_json::json!({"type": "function_call_output", "call_id": id, "output": text}))
        };
        let lines = [
            serde_json::json!({"type": "session_meta", "payload": {"id": sid, "cwd": repo.path()}}),
            item(serde_json::json!({
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "show both adr notes"}],
            })),
            exec("call_adr", "docs"),
            output("call_adr", "ADRSENTINEL private decision"),
            exec("call_pub", "src"),
            output("call_pub", "PUBLICCONTROL public note"),
        ];
        let posted = backfill_posted(
            ManagedHarness::Codex,
            home.path(),
            home.path(),
            repo.path(),
            sid,
            &lines,
        )
        .await;
        assert!(!posted.contains("ADRSENTINEL"), "excluded output: {posted}");
        assert!(!posted.contains("/docs\\\""), "excluded call: {posted}");
        assert!(posted.contains("PUBLICCONTROL"), "public output: {posted}");
        assert!(posted.contains("/src"), "public call: {posted}");
    }

    /// A Claude Code transcript record: `content` is the message content.
    #[cfg(unix)]
    fn claude_record(
        sid: &str,
        uuid: &str,
        cwd: &Path,
        role: &str,
        content: serde_json::Value,
    ) -> serde_json::Value {
        serde_json::json!({
            "type": role, "uuid": uuid, "sessionId": sid, "cwd": cwd,
            "timestamp": "2026-09-01T10:00:00Z",
            "message": {"role": role, "content": content},
        })
    }

    #[cfg(unix)]
    fn claude_tool(name: &str, id: &str, input: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"type": "tool_use", "id": id, "name": name, "input": input})
    }

    #[cfg(unix)]
    fn claude_output(id: &str, text: &str) -> serde_json::Value {
        serde_json::json!({"type": "tool_result", "tool_use_id": id, "content": text})
    }

    /// Claude Code records each line's `cwd`, which follows the session as it
    /// moves. The live hook resolves a relative argument from that cwd, so an
    /// import must too: `cat adr/x.md` run from `docs/` reads an ignored file
    /// and is never posted, while the same command run from the checkout root
    /// names a public path and is.
    #[cfg(unix)]
    #[tokio::test]
    async fn backfill_resolves_each_call_from_its_record_cwd() {
        let home = tempfile::tempdir().unwrap();
        let repo = marked_checkout("[capture]\nignore_paths = [\"docs/adr/**\"]\n");
        let docs = repo.path().join("docs");
        std::fs::create_dir_all(docs.join("adr")).unwrap();
        let sid = "11111111-2222-3333-4444-555555555556";
        let cat = |id| claude_tool("Bash", id, serde_json::json!({"command": "cat adr/x.md"}));
        let lines = [
            claude_record(sid, "u1", repo.path(), "user", "read the adr".into()),
            claude_record(
                sid,
                "a1",
                &docs,
                "assistant",
                serde_json::json!([cat("toolu_docs")]),
            ),
            claude_record(
                sid,
                "u2",
                &docs,
                "user",
                serde_json::json!([claude_output("toolu_docs", "ADRSENTINEL private decision")]),
            ),
            claude_record(
                sid,
                "a2",
                repo.path(),
                "assistant",
                serde_json::json!([cat("toolu_root")]),
            ),
            claude_record(
                sid,
                "u3",
                repo.path(),
                "user",
                serde_json::json!([claude_output("toolu_root", "PUBLICCONTROL root note")]),
            ),
        ];
        let posted = backfill_posted(
            ManagedHarness::Claude,
            home.path(),
            home.path(),
            repo.path(),
            sid,
            &lines,
        )
        .await;
        assert!(!posted.contains("ADRSENTINEL"), "excluded output: {posted}");
        assert!(posted.contains("PUBLICCONTROL"), "control output: {posted}");
        assert_eq!(
            posted.matches("cat adr/x.md").count(),
            2,
            "only the root call, as title and message: {posted}"
        );
    }

    /// A call the live hook would reroute to another checkout must pass that
    /// checkout's admission gate and reach the same server, or it is never
    /// posted: under allowlist mode an unmarked repository admits nothing, a
    /// marker selecting an unregistered profile emits nothing, and a marker
    /// selecting a registered profile sends to another server. Control: a read
    /// in a marked checkout of the same project with no server selection is
    /// posted.
    #[cfg(unix)]
    #[tokio::test]
    async fn backfill_drops_reroutes_the_live_hook_would_not_deliver_here() {
        let home = tempfile::tempdir().unwrap();
        let data_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            data_dir.path().join(super::super::hook::CAPTURE_MODE_FILE),
            "allowlist\n",
        )
        .unwrap();
        let name = crate::server_profiles::ProfileName::parse("team-e").unwrap();
        crate::server_profiles::add(data_dir.path(), &name, "https://e.example", &[], Some("E"))
            .unwrap();
        let source = marked_checkout("workspace = \"w\"\nproject = \"p\"\n");
        let unmarked = tempfile::tempdir().unwrap();
        let init = std::process::Command::new("git")
            .args(["init", "-q"])
            .arg(unmarked.path())
            .status()
            .unwrap();
        assert!(init.success());
        let unknown_profile = marked_checkout("server = \"team-b\"\n");
        let other_server = marked_checkout("server = \"team-e\"\n");
        let same_server = marked_checkout("workspace = \"w\"\nproject = \"p\"\n");
        let sid = "11111111-2222-3333-4444-555555555557";
        let reads = [
            (&unmarked, "UNMARKEDSENTINEL"),
            (&unknown_profile, "UNKNOWNPROFILESENTINEL"),
            (&other_server, "OTHERSERVERSENTINEL"),
            (&same_server, "SAMESERVERCONTROL"),
        ];
        let mut lines = vec![claude_record(
            sid,
            "u0",
            source.path(),
            "user",
            "read the notes".into(),
        )];
        for (index, (repo, text)) in reads.iter().enumerate() {
            let file = repo.path().join(format!("{text}.txt"));
            std::fs::write(&file, text).unwrap();
            let id = format!("toolu_{index}");
            lines.push(claude_record(
                sid,
                &format!("a{index}"),
                source.path(),
                "assistant",
                serde_json::json!([claude_tool(
                    "Read",
                    &id,
                    serde_json::json!({"file_path": file})
                )]),
            ));
            lines.push(claude_record(
                sid,
                &format!("u{}", index + 1),
                source.path(),
                "user",
                serde_json::json!([claude_output(&id, &format!("{text} content"))]),
            ));
        }
        let posted = backfill_posted(
            ManagedHarness::Claude,
            home.path(),
            data_dir.path(),
            source.path(),
            sid,
            &lines,
        )
        .await;
        for sentinel in [
            "UNMARKEDSENTINEL",
            "UNKNOWNPROFILESENTINEL",
            "OTHERSERVERSENTINEL",
        ] {
            assert!(
                !posted.contains(sentinel),
                "{sentinel} was posted: {posted}"
            );
        }
        assert!(
            posted.contains("SAMESERVERCONTROL content"),
            "control: {posted}"
        );
        assert!(posted.contains("read the notes"), "prompt kept: {posted}");
    }

    /// The live server stores only the family of an unknown-family tool, with
    /// no policy at all. A backfilled Codex MCP call and its output therefore
    /// post only `tool_family: unknown`, never their arguments or output, while
    /// a recognized shell call's output is posted as before.
    #[cfg(unix)]
    #[tokio::test]
    async fn backfill_posts_no_content_for_unrecognized_tools() {
        let home = tempfile::tempdir().unwrap();
        let repo = tempfile::tempdir().unwrap();
        let sid = "019a0000-0000-7000-8000-000000000002";
        let item = |payload: serde_json::Value| {
            serde_json::json!({
                "type": "response_item", "timestamp": "2026-09-01T10:00:00Z", "payload": payload,
            })
        };
        let call = |id: &str, name: &str, arguments: serde_json::Value| {
            item(serde_json::json!({
                "type": "function_call", "name": name, "call_id": id,
                "arguments": arguments.to_string(),
            }))
        };
        let output = |id: &str, text: &str| {
            item(serde_json::json!({"type": "function_call_output", "call_id": id, "output": text}))
        };
        let lines = [
            serde_json::json!({"type": "session_meta", "payload": {"id": sid, "cwd": repo.path()}}),
            call(
                "call_mcp",
                "mcp__vault__fetch",
                serde_json::json!({"query": "ARGSENTINEL"}),
            ),
            output("call_mcp", "MCPSENTINEL private record"),
            call(
                "call_pub",
                "exec_command",
                serde_json::json!({"cmd": "cargo test"}),
            ),
            output("call_pub", "PUBLICCONTROL test result"),
        ];
        let posted = backfill_posted(
            ManagedHarness::Codex,
            home.path(),
            home.path(),
            repo.path(),
            sid,
            &lines,
        )
        .await;
        for sentinel in ["ARGSENTINEL", "MCPSENTINEL", "mcp__vault"] {
            assert!(
                !posted.contains(sentinel),
                "{sentinel} was posted: {posted}"
            );
        }
        assert_eq!(
            posted.matches("\"title\":\"tool unknown\"").count(),
            2,
            "the call and its output are still recorded: {posted}"
        );
        assert!(posted.contains("PUBLICCONTROL"), "control: {posted}");
    }

    /// A reroute is delivered only when the destination's marker selects the
    /// same registered server profile as the source and the same project: a
    /// different registered profile is dropped, and so is a checkout of
    /// another project on the same profile, whose events the live hook files
    /// under that project. The same holds for records written in such a
    /// checkout, whatever their tool calls name.
    #[cfg(unix)]
    #[tokio::test]
    async fn backfill_keeps_reroutes_to_the_same_server_profile_only() {
        let home = tempfile::tempdir().unwrap();
        let data_dir = tempfile::tempdir().unwrap();
        let base = tempfile::tempdir().unwrap();
        let checkout = |name: &str, profile: &str, project: &str| {
            let dir = base.path().join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join(".ai-memory.toml"),
                format!("server = \"{profile}\"\nworkspace = \"w\"\nproject = \"{project}\"\n"),
            )
            .unwrap();
            dir
        };
        let source = checkout("source", "team-a", "p");
        let same = checkout("same", "team-a", "p");
        let other = checkout("other", "team-e", "p");
        let sibling = checkout("sibling", "team-a", "q");
        let root = |dir: &Path| dir.to_string_lossy().into_owned();
        for (name, roots) in [
            ("team-a", vec![root(&source), root(&same), root(&sibling)]),
            ("team-e", vec![root(&other)]),
        ] {
            let name = crate::server_profiles::ProfileName::parse(name).unwrap();
            crate::server_profiles::add(
                data_dir.path(),
                &name,
                "https://a.example",
                &roots,
                Some("T"),
            )
            .unwrap();
        }
        let sid = "11111111-2222-3333-4444-555555555558";
        let mut lines = vec![claude_record(sid, "u0", &source, "user", "read".into())];
        for (index, (repo, text)) in [
            (&same, "SAMEPROFILECONTROL"),
            (&other, "OTHERPROFILESENTINEL"),
            (&sibling, "OTHERPROJECTSENTINEL"),
        ]
        .into_iter()
        .enumerate()
        {
            let file = repo.join(format!("{text}.txt"));
            let id = format!("toolu_{index}");
            lines.push(claude_record(
                sid,
                &format!("a{index}"),
                &source,
                "assistant",
                serde_json::json!([claude_tool(
                    "Read",
                    &id,
                    serde_json::json!({"file_path": file})
                )]),
            ));
            lines.push(claude_record(
                sid,
                &format!("u{}", index + 1),
                &source,
                "user",
                serde_json::json!([claude_output(&id, &format!("{text} content"))]),
            ));
        }
        lines.push(claude_record(
            sid,
            "moved",
            &sibling,
            "user",
            "SIBLINGPROMPTSENTINEL".into(),
        ));
        let posted = backfill_posted(
            ManagedHarness::Claude,
            home.path(),
            data_dir.path(),
            &source,
            sid,
            &lines,
        )
        .await;
        assert!(
            posted.contains("SAMEPROFILECONTROL content"),
            "same profile: {posted}"
        );
        for sentinel in [
            "OTHERPROFILESENTINEL",
            "OTHERPROJECTSENTINEL",
            "SIBLINGPROMPTSENTINEL",
        ] {
            assert!(!posted.contains(sentinel), "{sentinel}: {posted}");
        }
    }

    /// Claude Code writes a monitored command's streamed output and a
    /// background agent's report as user turns, and its compact summary too;
    /// no prompt hook fires for any of them, so none is posted as a prompt.
    /// Control: the operator's own prompt is.
    #[cfg(unix)]
    #[tokio::test]
    async fn backfill_posts_no_task_notification_or_compact_summary_as_a_prompt() {
        let home = tempfile::tempdir().unwrap();
        let repo = tempfile::tempdir().unwrap();
        let sid = "11111111-2222-3333-4444-555555555559";
        let mut monitor = claude_record(
            sid,
            "n1",
            repo.path(),
            "user",
            "<task-notification>\n<event>MONITORSENTINEL stdout line</event>\n</task-notification>"
                .into(),
        );
        monitor["origin"] = serde_json::json!({"kind": "task-notification"});
        let report = claude_record(
            sid,
            "n2",
            repo.path(),
            "user",
            "<task-notification>\n<result>AGENTSENTINEL report</result>\n</task-notification>"
                .into(),
        );
        let mut summary = claude_record(
            sid,
            "c1",
            repo.path(),
            "user",
            "This session is being continued. SUMMARYSENTINEL".into(),
        );
        summary["isCompactSummary"] = true.into();
        let lines = [
            claude_record(sid, "u1", repo.path(), "user", "PROMPTCONTROL".into()),
            monitor,
            report,
            summary,
        ];
        let posted = backfill_posted(
            ManagedHarness::Claude,
            home.path(),
            home.path(),
            repo.path(),
            sid,
            &lines,
        )
        .await;
        for sentinel in ["MONITORSENTINEL", "AGENTSENTINEL", "SUMMARYSENTINEL"] {
            assert!(!posted.contains(sentinel), "{sentinel}: {posted}");
        }
        assert!(posted.contains("PROMPTCONTROL"), "control: {posted}");
    }

    fn spawned_args(
        server_url: Option<&str>,
        server_profile: Option<&str>,
    ) -> crate::cli::BackfillArgs {
        crate::cli::BackfillArgs {
            workspace: None,
            project: None,
            session: None,
            force: false,
            dry_run: false,
            max_sessions: 25,
            json: false,
            quiet: true,
            auto: true,
            server_url: server_url.map(str::to_owned),
            server_profile: server_profile.map(str::to_owned),
        }
    }

    /// #992: a SessionStart-spawned backfill presents only the credential the
    /// hook itself uses for that server — never the config/env bearer, which
    /// may belong to another server entirely.
    #[tokio::test]
    async fn a_spawned_backfill_authenticates_like_the_hook_that_spawned_it() {
        let home = tempfile::tempdir().unwrap();
        let data_dir = tempfile::tempdir().unwrap();
        let mut config =
            crate::config::Config::load(None, Some(home.path().to_path_buf())).unwrap();
        config.data_dir = data_dir.path().to_path_buf();
        config.auth.bearer_token = Some("CONFIG-SERVER-TOKEN".into());

        let url = "https://hook.example/wiki";
        let endpoint = backfill_endpoint(&config, &spawned_args(Some(url), None))
            .await
            .unwrap();
        assert_eq!(
            endpoint.auth_token, None,
            "no hook token: nothing, not config's"
        );
        assert_eq!(endpoint.url, "https://hook.example");
        assert_eq!(endpoint.base_path, "/wiki");

        crate::config::store_hook_auth_token(data_dir.path(), "HOOK-TOKEN").unwrap();
        let endpoint = backfill_endpoint(&config, &spawned_args(Some(url), None))
            .await
            .unwrap();
        assert_eq!(endpoint.auth_token.as_deref(), Some("HOOK-TOKEN"));

        let name = crate::server_profiles::ProfileName::parse("team-b").unwrap();
        crate::server_profiles::add(data_dir.path(), &name, "https://b.example", &[], Some("B"))
            .unwrap();
        let endpoint = backfill_endpoint(&config, &spawned_args(None, Some("team-b")))
            .await
            .unwrap();
        assert_eq!(endpoint.url, "https://b.example");
        assert_eq!(endpoint.auth_token.as_deref(), Some("B"));
        assert!(
            backfill_endpoint(&config, &spawned_args(None, Some("nobody")))
                .await
                .is_err(),
            "an unregistered profile is refused, not replaced by config"
        );
    }

    /// The automatic path must honor the `backfill_on_start` opt-out: it records
    /// the attempt (so it never re-spawns) and returns without contacting the
    /// server at all.
    #[tokio::test]
    async fn auto_run_opted_out_writes_sentinel_and_makes_no_request() {
        let home = tempfile::tempdir().unwrap();
        let data_dir = tempfile::tempdir().unwrap();
        let mut config =
            crate::config::Config::load(None, Some(home.path().to_path_buf())).unwrap();
        config.data_dir = data_dir.path().to_path_buf();
        config.backfill_on_start = false;
        // An unroutable server would make any request hang/fail; the opt-out
        // must return before we ever build the endpoint.
        config.server_url = "http://127.0.0.1:9".to_string();

        run(&config, spawned_args(None, None))
            .await
            .expect("opted-out auto run must succeed without contacting the server");

        let cwd = std::env::current_dir().unwrap();
        assert!(
            sentinel_path(&config.data_dir, &cwd).exists(),
            "the opt-out must still record the attempt so it does not re-spawn"
        );
    }
}
