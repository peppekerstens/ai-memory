//! Opt-in post-fusion ranking signals for `hybrid_search`.
//!
//! Two bounded, independently switchable signals, both off by default so a
//! store that never configures them ranks exactly as before:
//!
//! * **Session-recall routing** — a zero-LLM lexical check that recognises
//!   queries phrased as "find a past session / what did we do back then"
//!   ("上次 / 之前那次…的会话 / last time / yesterday / última sessão …"). Session pages
//!   otherwise carry a bounded authority penalty (kind `session` −0.15,
//!   tier `episodic` −0.08, ×0.77 combined) because they are rarely the
//!   right answer to a fact query — but a query that is *about* a past
//!   session needs one. When the routing fires, the penalty is handed
//!   back plus a small bonus, still inside the authority bounds every
//!   other page lives in.
//! * **Abstract vectors** — a fifth RRF stream over
//!   `page_abstract_embeddings`, the embedding of a page's frontmatter
//!   `abstract:` line (the L0 layer). A one-line summary embeds far more
//!   sharply than a multi-thousand-character body, giving the vector side
//!   a precise second vote per page. Contributes nothing while the table
//!   is empty.
//!
//! Both signals are reported in `SearchExplain`, so an `explain=true`
//! query can account for the returned rank.
//!
//! The same struct carries two **payload width** tunables. They do not
//! change the rank of any hit. They change how much text each hit shows:
//!
//! * **Snippet tokens** — the last argument of the FTS5 `snippet()` calls
//!   that build the excerpt of a lexical hit. FTS5 accepts 1 to 64.
//! * **Descriptor characters** — the budget of [`page_descriptor`], the
//!   excerpt of a vector, entity, graph-neighbour or recency hit, which
//!   has no matched term to centre on.
//!
//! Both defaults keep the payload byte-identical to a store that never
//! configures them.
//!
//! [`page_descriptor`]: crate::reader

use serde::{Deserialize, Serialize};

/// Tunables for the opt-in ranking signals. `Default` disables both.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RetrievalTuning {
    /// Lexical session-recall routing (see [`is_session_recall_query`]).
    pub session_recall_routing: bool,
    /// Extra authority added to session pages when the routing fires, on
    /// top of cancelling their kind/tier penalty.
    pub session_recall_bonus: f64,
    /// Add the L0 abstract-embedding stream to the RRF fusion. Reads
    /// `page_abstract_embeddings`; contributes nothing while it is empty.
    pub abstract_vectors: bool,
    /// Weight of the belief-strength confidence factor folded into
    /// [`crate::belief`] page authority (P2,
    /// `docs/design-hindsight-borrowings.md` §3). `0.0` (the default) leaves
    /// ranking byte-identical to a store that never heard of belief strength:
    /// no belief query runs and the authority factor is untouched. When
    /// positive, a page's derived `confidence` adds up to `weight * confidence`
    /// to its authority factor, still clamped inside the existing `[0.55,
    /// 1.50]` bounds — one more bounded factor, never a new multiplier tower.
    ///
    /// Ships OFF: this changes retrieval ranking, so per the design it must not
    /// default on without a positive R2 delta (retrieval-triple / QA), not yet
    /// performed.
    pub belief_authority_weight: f64,
    /// Token budget of the FTS5 `snippet()` excerpt on a lexical hit.
    /// FTS5 accepts 1 to 64 and fails the query outside that range, so
    /// [`RetrievalTuning::sanitised`] clamps the value.
    pub snippet_tokens: usize,
    /// Character budget of the synthesised descriptor on a vector, entity,
    /// graph-neighbour or recency hit. Clamped to
    /// [`DESCRIPTOR_CHARS_MIN`]..=[`DESCRIPTOR_CHARS_MAX`].
    pub descriptor_chars: usize,
}

/// Lowest FTS5 snippet token budget.
pub const SNIPPET_TOKENS_MIN: usize = 1;
/// Highest FTS5 snippet token budget. The FTS5 `snippet()` function fails
/// the whole query above this number.
pub const SNIPPET_TOKENS_MAX: usize = 64;
/// Default FTS5 snippet token budget. Keeps the shipped payload size.
pub const SNIPPET_TOKENS_DEFAULT: usize = 24;
/// Lowest page descriptor budget, in characters.
pub const DESCRIPTOR_CHARS_MIN: usize = 80;
/// Highest page descriptor budget, in characters. A descriptor is one hit
/// of a multi-hit answer, so this bounds the payload of the whole answer.
pub const DESCRIPTOR_CHARS_MAX: usize = 2_000;
/// Default page descriptor budget. Keeps the shipped payload size.
pub const DESCRIPTOR_CHARS_DEFAULT: usize = 240;

impl Default for RetrievalTuning {
    fn default() -> Self {
        Self {
            session_recall_routing: false,
            // Measured on a 138-query golden set: 0.25 recovers most
            // session-recall hit@1 (0.14 → 0.71 alongside the abstract
            // stream) while unrouted fact queries keep their exact baseline
            // ranking. Lower it (e.g. 0.15) if "之前/上次"-prefixed fact
            // queries are common in your traffic and rank drift on them
            // matters more than session recall.
            session_recall_bonus: 0.25,
            abstract_vectors: false,
            // OFF by default: folding belief confidence into ranking is
            // R2-gated (see the field doc).
            belief_authority_weight: 0.0,
            snippet_tokens: SNIPPET_TOKENS_DEFAULT,
            descriptor_chars: DESCRIPTOR_CHARS_DEFAULT,
        }
    }
}

impl RetrievalTuning {
    /// A copy with every payload width inside its supported range.
    ///
    /// The store calls this on the value it receives, so a bad number from
    /// a config file or an env var narrows the excerpt instead of failing
    /// every lexical query at SQL level.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            snippet_tokens: self
                .snippet_tokens
                .clamp(SNIPPET_TOKENS_MIN, SNIPPET_TOKENS_MAX),
            descriptor_chars: self
                .descriptor_chars
                .clamp(DESCRIPTOR_CHARS_MIN, DESCRIPTOR_CHARS_MAX),
            ..self
        }
    }
}

// Markers are matched on the lower-cased query. CJK phrases match as plain
// substrings; Latin phrases match on word boundaries so "now" never fires
// inside "known" or "snow". "会话" (session) only counts in a recall frame
// ("的会话 / 次会话 / 会话里"), so a query *about* sessions as a topic —
// "user_sessions 异常会话" — is not routed to past-session recall.
const SESSION_RECALL_ZH: &[&str] = &[
    "之前",
    "上次",
    "上回",
    "上一次",
    "那次",
    "那回",
    "前几天",
    "前天",
    "昨天",
    "那天",
    "上周",
    "上星期",
    "上个月",
    "的会话",
    "次会话",
    "个会话",
    "会话里",
    "会话中",
    "历史会话",
    "当时我们",
    "我们当时",
];
const SESSION_RECALL_EN: &[&str] = &[
    "last time",
    "last session",
    "previous session",
    "earlier session",
    "that session",
    "the session where",
    "yesterday",
    "the other day",
    "last week",
    "when we",
    "we did",
    "did we",
    "what did we",
    "how did we",
    "back then",
    "earlier we",
    "previously",
];
// Require an explicit historical phrase: "sessão", "antes", or "ontem"
// alone also occur in technical queries unrelated to recalling our work.
const SESSION_RECALL_PT_BR: &[&str] = &[
    "última sessão",
    "ultima sessao",
    "última sessao",
    "ultima sessão",
    "sessão anterior",
    "sessao anterior",
    "onde paramos ontem",
    "decisão anterior",
    "decisao anterior",
];

/// Route a query to session-recall retrieval by lexical markers. Deliberately
/// zero-LLM: it gates a bounded ranking nudge, not a retrieval mode, and a
/// false positive costs a few rank positions rather than a wrong answer.
#[must_use]
pub fn is_session_recall_query(query: &str) -> bool {
    let lowered = query.to_lowercase();
    let padded = latin_word_padded(&lowered);
    SESSION_RECALL_ZH.iter().any(|m| lowered.contains(m))
        || SESSION_RECALL_EN
            .iter()
            .chain(SESSION_RECALL_PT_BR.iter())
            .any(|m| padded.contains(&format!(" {m} ")))
}

/// Replace every non-alphanumeric Latin character with a space and pad the
/// ends, so multi-word markers can be matched as ` word word `.
fn latin_word_padded(lowered: &str) -> String {
    let mut out = String::with_capacity(lowered.len() + 2);
    out.push(' ');
    for ch in lowered.chars() {
        if ch.is_ascii_alphanumeric() || (!ch.is_ascii() && ch.is_alphabetic()) {
            out.push(ch);
        } else {
            out.push(' ');
        }
    }
    out.push(' ');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tuning_is_inert() {
        let t = RetrievalTuning::default();
        assert!(!t.session_recall_routing);
        assert!(!t.abstract_vectors);
        assert_eq!(t.belief_authority_weight, 0.0);
        assert_eq!(t.snippet_tokens, SNIPPET_TOKENS_DEFAULT);
        assert_eq!(t.descriptor_chars, DESCRIPTOR_CHARS_DEFAULT);
    }

    #[test]
    fn payload_widths_are_clamped_to_the_supported_range() {
        let wide = RetrievalTuning {
            snippet_tokens: 4_000,
            descriptor_chars: 40_000,
            ..RetrievalTuning::default()
        }
        .sanitised();
        assert_eq!(wide.snippet_tokens, SNIPPET_TOKENS_MAX);
        assert_eq!(wide.descriptor_chars, DESCRIPTOR_CHARS_MAX);

        let narrow = RetrievalTuning {
            snippet_tokens: 0,
            descriptor_chars: 1,
            ..RetrievalTuning::default()
        }
        .sanitised();
        assert_eq!(narrow.snippet_tokens, SNIPPET_TOKENS_MIN);
        assert_eq!(narrow.descriptor_chars, DESCRIPTOR_CHARS_MIN);
    }

    #[test]
    fn sanitised_keeps_a_supported_value() {
        let t = RetrievalTuning {
            snippet_tokens: 48,
            descriptor_chars: 400,
            ..RetrievalTuning::default()
        }
        .sanitised();
        assert_eq!(t.snippet_tokens, 48);
        assert_eq!(t.descriptor_chars, 400);
    }

    #[test]
    fn detects_session_recall_queries() {
        for q in [
            "上次排查 Caddy 配置的会话完成了吗",
            "之前那次我们把端口改成了什么",
            "之前 agent-memory 方案推到 GitHub 的首次提交哈希是什么",
            "what did we decide last time about the reranker",
            "yesterday's session on the spool bug",
        ] {
            assert!(is_session_recall_query(q), "{q}");
        }
    }

    #[test]
    fn plain_fact_queries_are_not_session_recall() {
        for q in [
            "PowerShell 没有 head 命令",
            "lark-cli docs update 报错 degrade_code 1011",
            "known issue with snow rendering",
            "git commit unable to auto-detect email",
            // "会话" as a topic, not a recall frame.
            "user_sessions 异常会话 登录失败",
        ] {
            assert!(!is_session_recall_query(q), "{q}");
        }
    }

    #[test]
    fn latin_markers_respect_word_boundaries() {
        assert!(!is_session_recall_query("unknown snowfall"));
        assert!(is_session_recall_query("what did we ship yesterday"));
    }

    #[test]
    fn detects_portuguese_session_recall_queries() {
        for q in [
            "o que fizemos na última sessão",
            "o que fizemos na ultima sessao",
            "o que fizemos na última sessao",
            "o que fizemos na ultima sessão",
            "onde paramos ontem",
            "lembre a decisão anterior",
            "lembre a decisao anterior",
            "o que discutimos na sessão anterior",
            "o que discutimos na sessao anterior",
            "ONDE PARAMOS ONTEM?",
            "lembre a (DECISÃO ANTERIOR), sobre SQLite",
        ] {
            assert!(is_session_recall_query(q), "{q}");
        }
    }

    #[test]
    fn portuguese_markers_require_history_and_word_boundaries() {
        for q in [
            "erro na sessão do usuário",
            "erro na sessao do usuario",
            "a sessão passada para o middleware é nula",
            "sessao passada como parametro",
            "sessão expira",
            "sessao expira",
            "antes de salvar",
            "sessão",
            "sessao",
            "antes",
            "ontem",
            "decisão",
            "anterior",
            "o que fizemos agora",
            "onde paramos o serviço",
            "penúltima sessão",
            "penultima sessao",
            "decisão anteriormente tomada",
            "decisao anteriormente tomada",
            "xonde paramos ontem",
            "onde paramos ontemx",
        ] {
            assert!(!is_session_recall_query(q), "{q}");
        }
    }
}
