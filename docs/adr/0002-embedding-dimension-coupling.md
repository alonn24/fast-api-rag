# ADR 0002: Embedding dimension is duplicated across config, models, and schema

## Status

Accepted

## Context

The embedding vector dimension (`1024`, matching Voyage's `voyage-3` model) is
declared independently in three places with no shared source of truth:

- `app/config.py`: `Settings.embedding_dim` — read by nothing else in the
  codebase today.
- `app/db/models.py`: `EMBEDDING_DIM = 1024`, used in `Chunk.embedding`'s
  `Vector(EMBEDDING_DIM)` column type.
- `app/db/schema.sql`: `embedding VECTOR(1024) NOT NULL`, the actual Postgres
  column type applied at startup.

If `app.config.Settings.embedding_model` is ever changed to a Voyage model
with a different output dimension (or Voyage's `voyage-3` changes its output
size), none of these three values updates automatically, and nothing warns
that they must all change together. Since the ORM's `Vector(EMBEDDING_DIM)`
and the actual Postgres column type are independently declared, the failure
mode is a runtime `pgvector` dimension-mismatch error the first time an
embedding of the new size is inserted — not a startup-time or test-time
failure.

This is the same class of problem as [ADR 0001](0001-ivfflat-exhaustive-probes.md)'s
`ivfflat.probes`/`lists` coupling: two-or-more values in different files that
must be changed together, with nothing enforcing the link.

## Decision

Do not unify the three declarations into a single source of truth right now
— `Settings.embedding_dim` cannot easily flow into `models.py`'s
module-level `Vector(EMBEDDING_DIM)` column type without restructuring how
`Base`/`Document`/`Chunk` are defined (SQLAlchemy column types are
constructed at class-definition/import time, before `Settings()` can be
safely instantiated in every context this module is imported from, e.g.
Alembic-less schema tooling or scripts that don't set the required API key
env vars).

Instead:

1. This ADR documents the coupling explicitly.
2. `tests/test_embedding_dim_consistency.py` adds a regression test that
   fails loudly if the three values ever drift apart, so a future change to
   one of them is caught in the test suite rather than at insert time in
   production.

## Consequences

- Changing the embedding model's output dimension requires updating all
  three of `Settings.embedding_dim`, `app/db/models.py`'s `EMBEDDING_DIM`,
  and `app/db/schema.sql`'s `VECTOR(...)` column definition, in the same
  change — `tests/test_embedding_dim_consistency.py` will fail if any one is
  missed.
- If this project later needs to support multiple embedding dimensions
  simultaneously (e.g. a model migration with mixed old/new vectors), this
  ADR's assumption (one fixed dimension across the whole schema) no longer
  holds and the schema/models will need real migration support (which would
  also be the point to introduce Alembic, given the current no-Alembic
  decision assumes a single evolving schema applied idempotently).
