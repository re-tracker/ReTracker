# Local Tools (Gitignored)

This folder is for **personal / experimental utilities** that are useful during research and debugging
but are **not** part of the supported library API.

## Rules

- Everything under `experiments/tools/` is ignored by git by default.
- Only this README and `.gitignore` are committed.
- If a tool becomes generally useful, promote it to:
  - `scripts/tools/` (project-wide helper), or
  - a proper `retracker ...` CLI subcommand (with tests and documented dependencies).

