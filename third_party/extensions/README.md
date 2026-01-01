# Extensions / Plugins

This directory contains optional extensions and third-party integrations for the ReTracker project.

## Structure

```
extensions/
├── README.md           # This file
└── .gitkeep
```

## Available Extensions

Currently no extensions are publicly available. Check back for future releases.

---

## Adding New Extensions

To add a new extension:

1. Create a subdirectory: `extensions/<extension_name>/`
2. Add a `README.md` explaining the extension
3. Add a `setup.sh` script for installation
4. Update this file with the extension information
5. Add to `.gitignore` if it includes large files or cloned repos

---

## Best Practices

1. **Keep extensions independent**: Each extension should be self-contained
2. **Document dependencies**: Clearly state requirements in extension README
3. **Provide setup scripts**: Automate installation when possible
4. **Use .gitignore**: Don't commit large third-party code
5. **Version tracking**: Document which versions are tested/compatible

---

## Maintenance

Extensions are optional and maintained separately from the core codebase.
See individual extension READMEs for update instructions.
