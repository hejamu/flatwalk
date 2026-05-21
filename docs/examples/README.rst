Validation gallery
==================

Runnable examples that exercise the same machinery as the spec §4.4
validation, executed live during every documentation build via
``sphinx-gallery``. Each example is a self-contained Python script
that can also be run directly from the command line.

The full ``ln_f_final = 1e-8`` L=8 validation that satisfies the spec
pass criteria is too slow to run on every docs build (~15 min);
``examples/ising_validation.py`` at the repo root is the canonical
runner for that, executed by CI on every push. The examples in this
gallery are short smoke versions of the same pipeline that complete
in seconds.
