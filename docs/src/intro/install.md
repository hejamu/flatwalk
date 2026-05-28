# Install

flatwalk is on PyPI:

```bash
pip install flatwalk
```

## With extras

The base install carries only what the driver itself needs (NumPy). Pull in
optional extras when you want the test suite or the docs build:

```bash
pip install "flatwalk[test]"   # runtime + pytest
pip install "flatwalk[docs]"   # runtime + Sphinx, theme, gallery
```

## From the repository

To work on the source, clone the repository and install in editable mode. An
editable (`-e`) install points at the working tree, so edits to the source take
effect without reinstalling:

```bash
git clone https://github.com/hejamu/flatwalk.git
cd flatwalk
pip install -e ".[test]"
```

On a Homebrew Python you may need a virtual environment (or, less cleanly,
`pip install --break-system-packages`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Building the documentation

Install the `docs` extra and build the Sphinx site through tox:

```bash
pip install -e ".[docs]"
tox -e docs   # output in docs/build/html
```
