//! `GET /search?q=…` — FTS5 search results.

use std::sync::Arc;

use askama::Template;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::Html;
use serde::Deserialize;

use crate::markdown;
use crate::state::WebState;
use crate::templates::{SearchHit, SearchView, page_href};

/// Query-string parameters for the search endpoint.
#[derive(Debug, Deserialize)]
pub(crate) struct SearchParams {
    /// The free-text search query.
    #[serde(default)]
    pub q: String,
}

/// Most results the search page renders. One more is requested so a cut-off
/// result set can be told apart from one that happens to have exactly this many.
const MAX_RESULTS: usize = 50;

/// Handler for `GET /search?q=…`.
///
/// Searches only the repositories the viewer may read (#708) — see
/// `ReaderPool::search_pages_with_meta`. No viewer (no database users, or
/// root) searches everything, as before.
pub(crate) async fn handler(
    State(state): State<Arc<WebState>>,
    viewer: Option<axum::Extension<ai_memory_core::AuthorizedViewer>>,
    Query(params): Query<SearchParams>,
) -> Result<Html<String>, StatusCode> {
    let query = params.q.trim().to_owned();

    let (hits, truncated) = if query.is_empty() {
        (Vec::new(), false)
    } else {
        let raw = state
            .reader
            .search_pages_with_meta(
                query.clone(),
                MAX_RESULTS + 1,
                None,
                viewer.map(|axum::Extension(viewer)| viewer.user()),
            )
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let truncated = raw.len() > MAX_RESULTS;
        let mut results = Vec::with_capacity(raw.len().min(MAX_RESULTS));
        for h in raw.into_iter().take(MAX_RESULTS) {
            let path = h.path.as_str().to_owned();
            let href = page_href(&h.workspace_name, &h.project_name, &path);
            results.push(SearchHit {
                workspace: h.workspace_name,
                project: h.project_name,
                path,
                href,
                title: h.title,
                snippet: markdown::escape_snippet(&h.snippet),
            });
        }
        (results, truncated)
    };

    let hit_count = hits.len();
    let html = SearchView {
        query,
        hits,
        hit_count,
        truncated,
        max_results: MAX_RESULTS,
    }
    .render()
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Html(html))
}
