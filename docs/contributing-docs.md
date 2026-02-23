# Contributing Docs

## Principles

- Write for non-experts first.
- Keep wording short and direct.
- Prefer small examples that can be copied quickly.

## Style Rules

Each API entry should include:

1. What it does
2. Inputs
3. Returns
4. Simple example
5. Common mistake

Use aliases like `dingo::mat` and `dingo::vec` in examples unless a template type is required.

## Local Preview

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve
```

## Validation

Run strict build before opening PR:

```bash
mkdocs build --strict
```

## File Layout

- Site config: `mkdocs.yml`
- Markdown pages: `docs/`
- API reference pages: `docs/api-reference/`
- Pages workflow: `.github/workflows/docs-pages.yml`
