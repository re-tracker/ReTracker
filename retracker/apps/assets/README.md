# App Assets (Local Cache)

ReTracker caches small runtime assets (e.g. the example video used by
`retracker apps tracking --demo`).

By default, caches live under the user's cache directory:
- `$XDG_CACHE_HOME/retracker/assets` (if `XDG_CACHE_HOME` is set)
- otherwise `~/.cache/retracker/assets`

If you prefer to keep assets inside a source checkout, you can override:

```bash
export RETRACKER_ASSETS_DIR=/path/to/repo/retracker/apps/assets
```

This folder is committed so the path exists in the repo, but actual binaries
(mp4, images, etc.) are gitignored.
