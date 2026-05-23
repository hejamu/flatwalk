# Install

flatwalk is not yet published on PyPI, so install it from the repository.

## From the repository

Clone the repository and install it in editable mode. The `test` extra pulls in
the dependencies needed to run the test suite:

```bash
git clone https://github.com/hejamu/flatwalk.git
cd flatwalk
pip install -e ".[test]"
```

An editable (`-e`) install points at the working tree, so edits to the source
take effect without reinstalling. To install a pinned version without keeping a
clone around, point pip straight at the repository instead:

```bash
pip install "flatwalk[test] @ git+https://github.com/hejamu/flatwalk.git"
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
