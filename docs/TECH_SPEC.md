# TECH_SPEC

AutoCiv Phase 0.5 layers a bounded, replayable social simulation on top of the
minimal `autoresearch` substrate without replacing the upstream experiment loop.

Current bootstrap scope:

- preserve `prepare.py`, `train.py`, and `program.md` at the repo root
- mirror the upstream substrate under `substrate/`
- add `society/`, `worlds/`, `evals/`, `roles/`, `config/`, `docs/`, `scripts/`
- store metadata in SQLite and text artifacts/logs on disk
- run prompt-only lifespans through a deterministic mock provider
- persist generation summaries, memorials, and drift metrics

The bootstrap is intentionally narrow. It is designed to make replay, audit, and
failure analysis cheap before any adapter or checkpoint evolution exists.

Future extension work, including public volunteer participation, should preserve
that same narrowness: bounded work units, replayability, and no uncontrolled
distributed training. See `docs/COMMUNITY_COMPUTE.md`.
