# Install

Editable install via [uv](https://github.com/astral-sh/uv):

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[test]"
```

Plain pip works too (`pip install -e ".[test]"`) but Homebrew Python may
require `--break-system-packages` or a venv.

To build the documentation locally, install the `docs` extra and run the
Sphinx build through tox:

```bash
uv pip install --python .venv/bin/python -e ".[docs]"
tox -e docs   # output in docs/build/html
```
