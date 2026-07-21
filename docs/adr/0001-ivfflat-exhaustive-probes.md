# ADR 0001: Force exhaustive ivfflat probing in `search_chunks`

## Status

Accepted

## Context

`app/db/schema.sql` indexes `chunks.embedding` with an `ivfflat` index
(`lists = 100`). `ivfflat` is an approximate-nearest-neighbor index: at query
time it only scans `probes` of the `lists` partitions (default `probes = 1`),
so results are not guaranteed to be the true top-k nearest rows unless
`probes` is raised.

With the small row counts in this project's tests (and in early real usage),
`probes = 1` regularly missed or misordered genuinely closer chunks, making
`search_chunks`'s cosine-distance ordering non-deterministic and test
assertions flaky. `app/services/search.py` now issues
`SET LOCAL ivfflat.probes = 100` (matching the index's `lists = 100`) before
every search query, forcing an exhaustive scan of all partitions and
guaranteeing exact, deterministic top-k results.

## Decision

Keep `probes` hardcoded to `100` in `search_chunks`, matching `schema.sql`'s
`lists = 100`, for now. This trades away `ivfflat`'s approximate-search
speed advantage — every query scans all partitions — in exchange for
correctness and deterministic ordering, which matters more while the corpus
is small and there are no stated latency requirements.

This creates an implicit coupling: `probes` in `app/services/search.py` must
track `lists` in `app/db/schema.sql` to remain exhaustive. Today neither
value is derived from the other or from `app/config.py` — if `lists` is
retuned later without a matching `probes` update, the non-deterministic
behavior this ADR addresses will return.

## Consequences

- Correct, deterministic search results today, at every corpus size the
  project currently handles.
- No performance cliff has been observed or measured yet; if/when this
  project's document corpus grows large enough that ivfflat's approximate
  search speed matters, this decision should be revisited — options include
  making `probes` configurable (e.g. via `app/config.py` `Settings`) with a
  smaller production default, or reconsidering the index type entirely
  (e.g. HNSW, which does not require this probes/lists tradeoff in the same
  way).
- Until then, any change to `schema.sql`'s `lists` value must be
  accompanied by a matching review of `search.py`'s hardcoded `probes`
  value, since nothing currently enforces this link automatically.
