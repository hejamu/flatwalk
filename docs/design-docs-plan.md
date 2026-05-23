# Design doc: restructuring the flatwalk documentation

Status: **implemented** (this plan has been carried out). Audience: the session
that wrote the docs, and anyone extending them.
Scope: reorganise `docs/` into five top-level sections — **Introduction**,
**Tutorials**, **Examples**, **Theory**, **API** — plus the existing
project-background pages. This document is a build plan, not a published
page; it lives in `docs/` (not `docs/src/`), so the `--fail-on-warning`
Sphinx build never tries to render it.

---

## 1. Goal

Today the docs are a flat set of pages (`get-started`, `validation`,
`storyline`, `design-unified-batched-step`, `api`) plus one
sphinx-gallery (`auto_examples/`, four `plot_*` scripts). The owner wants
three content sections with *deliberate, controlled overlap*, on top of an
introduction and the API reference:

1. **Theory** — the methods explained in depth: Monte Carlo, detailed
   balance, the density of states, Wang-Landau, the 1/t refinement,
   multiple/batched walkers, replica exchange.
2. **Examples** — recipes for a user writing *their own contract*. Mostly
   short, copy-pasteable scripts: each flat-histogram method applied to the
   Ising model (and one or two non-Ising twists).
3. **Tutorials** — a guided narrative through flat histogramming on **one
   system carried throughout** (2D Ising): start with plain Metropolis MC,
   feel its shortcomings, switch to Wang-Landau, enhance with 1/t, realise
   more walkers help, then replica exchange.

The three overlap on purpose. §4 defines who owns what so the overlap
informs rather than duplicates.

---

## 2. What exists today (inventory)

Source lives in `docs/src/` (the Sphinx source dir; `conf.py` is here).
Gallery *scripts* live in `docs/examples/` and are generated into
`docs/src/auto_examples/` at build time.

| File | Type | Disposition |
| --- | --- | --- |
| `docs/src/index.rst` | landing + toctrees | **Rewrite** — new captions/tree (§6) |
| `docs/src/get-started.rst` | install + contract + quickstart | **Split** into Introduction pages (§5.1) |
| `docs/src/api.rst` | autodoc reference | **Keep** as the API section |
| `docs/src/validation.md` | exact-reference validation story | **Keep**; move under Theory as the closing chapter (§5.5) |
| `docs/src/storyline.md` | architecture/design rationale | **Keep** under Background (§5.6) |
| `docs/src/design-unified-batched-step.md` | accepted design note | **Keep** under Background (§5.6) |
| `docs/examples/plot_1_first_run.py` | gallery: toy 1D walk | **Move** into Tutorials as step 0/1 (§5.4) |
| `docs/examples/plot_2_beale_reference.py` | gallery: exact n(E) | **Keep** in Examples (reference recipe) |
| `docs/examples/plot_3_single_walker_ising.py` | gallery: WL on Ising | **Keep** in Examples; refactor narrative bits → tutorial |
| `docs/examples/plot_4_replica_exchange_ising.py` | gallery: REWL | **Keep** in Examples |
| `docs/examples/README.rst` | gallery header | **Rewrite** as the Examples header |

Reusable physics already importable on `sys.path` during the build
(`conf.py` adds repo `examples/`): `examples/ising.py`,
`examples/ising_batched.py`, `examples/beale.py`. Tutorials and examples
should import these rather than re-deriving the Ising move — keep the
physics in one place.

### Build pipeline (constraints the author must respect)

- **Build command:** `tox -e docs` → `sphinx-build --fail-on-warning -E
  docs/src docs/build/html`. **Every warning is fatal.** Practical
  consequences: every page must sit in a toctree (or be outside `docs/src/`),
  and every cross-reference (`:doc:`, `:class:`, `:func:`, `:ref:`) must
  resolve.
- **Gallery executes on every build.** `sphinx_gallery_conf` has
  `abort_on_example_error: True` — any uncaught exception fails the build.
  Scripts must be **fast smoke versions** (seconds, `L=4`, loose
  `ln_f_final`), never the `L=8`/`1e-8` runs (those stay in repo-root
  `examples/*_validation.py`, CI slow lane). `min_reported_time: 1`.
- **Lint:** `tox -e lint` runs `ruff` and `sphinx-lint` over `docs/src` and
  `docs/examples` (extend `lint_folders` in `tox.ini` when you add
  `docs/tutorials`). Gallery scripts are real Python and are ruff-checked;
  `examples/*` already has bugbear (`B`) ignored — add the new tutorials dir
  to the same per-file ignore.
- **Generated dirs are gitignored:** `docs/build/`,
  `docs/src/auto_examples/`, `docs/src/sg_execution_times.rst`. Add
  `docs/src/auto_tutorials/` to `.gitignore` when the second gallery lands.
- **Theme/extensions:** Furo; `autodoc` + `napoleon` +
  `sphinx_autodoc_typehints` + `viewcode` + `intersphinx` (python, numpy) +
  `mathjax` + `sphinx_gallery` + `myst_parser`. MyST extensions enabled:
  `deflist, fieldlist, smartquotes, tasklist`, heading anchors depth 3.

---

## 3. Target documentation tree

```
docs/src/
  index.rst                      # landing; toctrees with new captions (§6)

  intro/                         # Introduction (RST or MyST)
    overview.md                  #   what flat histogram is; when to use flatwalk
    install.md                   #   (lifted from get-started "Install")
    the-contract.md              #   the 4-callback contract (from get-started)
    quickstart.md                #   the Ising quickstart (from get-started)

  tutorials/  -> docs/tutorials/ # gallery #2, narrative journey (§5.4)
    (generated: auto_tutorials/index)

  examples/   -> docs/examples/  # gallery #1, recipes (§5.3)
    (generated: auto_examples/index)

  theory/                        # Theory (MyST .md, math-heavy) (§5.2)
    index.md
    01-sampling-problem.md
    02-monte-carlo.md
    03-detailed-balance.md
    04-density-of-states.md
    05-wang-landau.md
    06-one-over-t.md
    07-multiple-walkers.md
    08-replica-exchange.md
    09-higher-d.md               # planned/forward-looking (optional)
    10-validation.md             # "how we know it's right" (from validation.md)

  api.rst                        # API reference (unchanged)

  background/                    # retained project material (§5.6)
    storyline.md
    design-unified-batched-step.md

docs/tutorials/                  # NEW gallery source (scripts + header)
  README.rst
  plot_1_plain_mc.py
  plot_2_wang_landau.py
  plot_3_one_over_t.py
  plot_4_more_walkers.py
  plot_5_replica_exchange.py
  plot_6_thermodynamics.py       # capstone (optional)

docs/examples/                   # EXISTING gallery source, reframed as recipes
  README.rst                     # rewritten
  plot_1_minimal_contract.py     # compact "write your own contract" (toy)
  plot_2_beale_reference.py
  plot_3_single_walker_ising.py
  plot_4_batched_ising.py        # NEW recipe
  plot_5_replica_exchange_ising.py
  plot_6_checkpoint_resume.py    # NEW recipe
  plot_7_trace_diagnostics.py    # NEW recipe (may be `code-block`, not plot_)
```

> Folder vs page: Intro/Theory/Background pages can be a single file each if
> short; the layout above assumes they grow. Keep filenames stable —
> `--fail-on-warning` punishes broken `:doc:` targets, so decide names once.

---

## 4. Dividing the overlap (the key design decision)

The same method (say, Wang-Landau) appears in Theory, in a Tutorial step,
and in an Example recipe. That is intended. The rule that keeps it from
becoming three copies of the same prose:

| Section | Owns | Voice | A reader arrives wanting… |
| --- | --- | --- | --- |
| **Theory** | the *why* and the *math* — derivations, criteria, convergence, references | expository, equations | "*Why* does this converge? What is the acceptance rule, exactly?" |
| **Tutorials** | the *journey* on one system — motivation, what breaks, what fixes it, what to look at in the plot | narrative, second-person, one continuous story | "Walk me from zero to competent; show me *why* I'd reach for each method." |
| **Examples** | the *minimal reusable script* per method — copy, change block 1, run | terse, recipe, light prose | "Give me the shortest correct script for method X that I can adapt." |

Cross-linking rules (enforced by the author, checked by `--fail-on-warning`):

- Every **Theory** page ends with a "See it run" line linking the matching
  Tutorial step and Example recipe.
- Every **Tutorial** step links the **Theory** page for the derivation ("the
  maths behind the acceptance rule: …") instead of re-deriving it, and links
  the matching **Example** recipe ("the standalone script: …").
- Every **Example** recipe opens with a one-line link to the Tutorial/Theory
  that explains it, then stays terse.
- Use `:doc:` for page links and `:ref:`/gallery backreferences for scripts.
  Sphinx-gallery emits `sphx_glr_*` targets and `mini-gallery` directives —
  use `.. minigallery:: flatwalk.WLDriver` to surface examples that touch a
  given API object on Theory/API pages.

The litmus test: **no paragraph of prose should be copy-pasted across two
sections.** If two sections want to say the same thing, one says it and the
other links.

---

## 5. Section-by-section content spec

### 5.1 Introduction

Goal: a newcomer understands what flatwalk is, installs it, grasps the
contract, and runs the quickstart — in that order — before meeting any
method depth. Source: split the current `get-started.rst`.

- **overview.md** — *new prose.* What is the density of states `g(Q)` and
  why estimate it; what "flat histogram" buys you (one run → all
  temperatures); flatwalk's distinguishing claim (order-parameter and
  energy-backend agnostic; the cut at callable boundaries — condense from
  `storyline.md` §1–2, don't duplicate it, link it). A "when to use / when
  not to" paragraph. A "where to go next" map to the three sections.
- **install.md** — lift the *Install* block from `get-started.rst` verbatim.
- **the-contract.md** — lift the contract table + the "`state` is opaque"
  paragraph from `get-started.rst`. This is the single canonical statement of
  the four-callback contract; everything else links here.
- **quickstart.md** — lift the two-block Ising quickstart from
  `get-started.rst`. Keep the "block 1 = your physics, block 2 = verbatim
  wiring" framing; it is the project's signature explanation.

After this split, delete `get-started.rst` (and fix the one inbound
reference in `index.rst`).

### 5.2 Theory

MyST `.md`, math-heavy. **Enable MyST math first** (see §6): add
`dollarmath` and `amsmath` to `myst_enable_extensions`, then `$…$` and
`$$…$$` render via the already-loaded `mathjax`. Without this, inline `$`
math will not parse. (Alternative: write Theory in `.rst` and use
`:math:` / `.. math::` — but the rest of the prose docs are MyST, so prefer
enabling dollarmath.)

Chapter outline (each chapter = the *why*; link out for the *runnable*):

1. **The sampling problem** (`01-sampling-problem.md`) — partition function
   $Z(\beta)=\sum_s e^{-\beta E_s}$; rewrite as a sum over the density of
   states $Z=\sum_E g(E)e^{-\beta E}$; the point of flat histogram: estimate
   $g$ once, get $\langle E\rangle$, $C_V$, $F$, $S$ at *every* $T$.
2. **Monte Carlo & Markov chains** (`02-monte-carlo.md`) — importance
   sampling, Metropolis, why naive sampling misses the tails (sets up
   Tutorial 1's failure).
3. **Detailed balance & ergodicity** (`03-detailed-balance.md`) — derive the
   acceptance probability; where `log_proposal_ratio` comes from (asymmetric
   moves); map each term to flatwalk's
   $\Delta = -\beta\,\Delta E + g[\text{old}] - g[\text{new}] + \text{lpr}$.
   This is the page Tutorials/Examples link for "why this acceptance rule".
4. **Density of states & thermodynamics from `g`** (`04-density-of-states.md`)
   — given $g(E)$, compute observables at any $T$; numerical care
   (log-sum-exp, the additive constant in `g`). Pairs with the capstone
   tutorial.
5. **Wang-Landau** (`05-wang-landau.md`) — the running bias on `g`, the
   histogram `H`, the flatness criterion (`flatness_threshold`), the
   `ln_f` halving schedule, what convergence means. Cite Wang & Landau PRL
   86, 2050 (2001) (already referenced in `core.py`).
6. **The 1/t refinement** (`06-one-over-t.md`) — why `ln_f` halving
   saturates the error; the Belardinelli-Pereyra switch to `ln_f = 1/t` when
   halving would drop below `1/t`; cite PRE 75, 046701 (2007). flatwalk does
   this automatically — explain when it kicks in.
7. **Multiple / batched walkers** (`07-multiple-walkers.md`) — many walkers
   sharing one `g` (scatter-add into the same bins); variance reduction;
   why batching matters for vectorised/GPU energy backends (condense from
   `storyline.md` §4, link it). Note the design rule "≥2 walkers move as a
   batch, never a Python loop."
8. **Replica-exchange WL** (`08-replica-exchange.md`) — overlapping windows,
   per-window `g`, the swap acceptance
   $\Delta = g_i(E_j) - g_i(E_i) + g_j(E_i) - g_j(E_j)$, even/odd pairing for
   detailed balance, and `join_g` (least-squares log-shift over overlaps).
   Why REWL beats single-walker (no $E\leftrightarrow -E$ asymmetry, fast
   tails, parallelism).
9. **Higher-dimensional order parameters** (`09-higher-d.md`, optional) —
   forward-looking; `BinND`, joint `g(E, M)`. Mark clearly as *planned*
   (matches `storyline.md` §6 and README "Planned"). Skip if you'd rather not
   document unbuilt features.
10. **How we know it's right** (`10-validation.md`) — the
    exact-reference lever; the content of today's `validation.md`. Closes the
    Theory section: here is the math (ch. 1–8), and here is the proof the
    implementation reproduces a known-exact `g(E)`. (Decision: validation
    lives under Theory, not Background — see §5.5.)

Math conventions: `$\beta = 1/(k_B T)$`, `$g$` = log density of states (be
explicit — flatwalk's `result.g` is the *log* density; the README/examples
say so, the theory must too). Define notation once on the index page.

### 5.3 Examples (recipes — gallery #1)

Keep the existing sphinx-gallery in `docs/examples/`, reframed: each script
is a **minimal, adaptable recipe** for one method, opening with a one-line
link to the Tutorial/Theory that explains it, then mostly code. Audience:
someone writing their own contract who wants the shortest correct template.

Recipes (the `block 1 = your physics / block 2 = verbatim wiring` framing
from the quickstart should recur, so a reader sees what to change):

1. `plot_1_minimal_contract.py` — the smallest contract on a toy (the 1D
   walk, distilled from today's `plot_1_first_run`; the *narrative* version
   moves to Tutorial 1, this is the bare recipe).
2. `plot_2_beale_reference.py` — keep; the exact `n(E)` reference, used by the
   validation and the Ising recipes.
3. `plot_3_single_walker_ising.py` — keep; trim the long "reaching the strict
   criteria" narrative (that belongs in Tutorial 3/4) to a link.
4. `plot_4_batched_ising.py` — **new**; `WLDriver.run_batched` with
   `examples/ising_batched.py` callbacks. The batched-callbacks recipe.
5. `plot_5_replica_exchange_ising.py` — keep (today's `plot_4`), renumbered.
6. `plot_6_checkpoint_resume.py` — **new**; show `checkpoint_path` /
   `checkpoint_every_t`, interrupt, resume, assert bit-identical (mirror
   `tests/test_checkpoint.py`, smoke-sized).
7. `plot_7_trace_diagnostics.py` — **new**; `trace_path` + `read_trace`,
   plot `ln_f`/flatness over time. (If it has no figure, name it without the
   `plot_` prefix so gallery includes it as code, not a thumbnail — or give
   it a small diagnostic plot.)

Keep every script under a few seconds. Renumbering touches inbound `:doc:` /
gallery refs — grep for `sphx_glr_` and `plot_` targets and the
`validation.md` worked-examples list before/after.

### 5.4 Tutorials (the journey — gallery #2, NEW)

A second sphinx-gallery, `docs/tutorials/` → `auto_tutorials/`. One system
(2D Ising, `L=4` smoke) carried through every step. Heavy `# %%` prose
blocks — these are *narrative scripts*, the prose-to-code ratio is high. The
arc the owner described:

1. `plot_1_plain_mc.py` — **plain Metropolis MC.** Hand-write a fixed-`T`
   Metropolis loop on the Ising model (no flatwalk WL yet — just to show the
   baseline). Measure $\langle E\rangle$ at one temperature. **Show the
   shortcoming:** you can't reach the tails; you'd need a separate run per
   temperature; reweighting across a wide $T$ range is unreliable. End: "what
   if one run gave us *every* temperature?" → motivates `g(E)`.
2. `plot_2_wang_landau.py` — **enter Wang-Landau.** Same Ising, now via
   `WLDriver.run`. Recover `g(E)`; derive $\langle E\rangle(T)$ and $C_V(T)$
   for a *range* of $T$ from the single run; compare to the per-`T` MC points
   from step 1. The payoff moment.
3. `plot_3_one_over_t.py` — **it stalls.** Show the error floor of pure
   `ln_f` halving (e.g. compare `g` vs Beale at moderate vs deep
   `ln_f_final`); explain the saturation; note flatwalk's automatic 1/t
   switch and show convergence improving. Link Theory ch. 6.
4. `plot_4_more_walkers.py` — **one walker isn't enough.** Surface the
   single-walker $E\leftrightarrow -E$ asymmetry / slow tails (today's
   `plot_3` "reaching strict criteria" note is the seed of this step). Switch
   to `run_batched` with several walkers sharing `g`; show variance dropping.
   Link Theory ch. 7.
5. `plot_5_replica_exchange.py` — **windows & exchange.** `make_windows`,
   `RewlDriver`, `join_g`; robustness and parallelism. Reuse the plot style
   from today's `plot_4_replica_exchange_ising`. Link Theory ch. 8.
6. `plot_6_thermodynamics.py` (capstone, optional) — from a converged `g(E)`,
   compute $\langle E\rangle$, $C_V$ (find the peak ≈ $T_c$), $F$, $S$ across
   $T$; sanity-check against Beale exact. Ties back to Theory ch. 4 and the
   validation story.

Each step must run clean and fast; step 1's hand-written MC and step 3's
"stall" demonstration must produce a *bad-but-finite* result (a plot showing
the failure), **never raise** — `abort_on_example_error` is on. Use assertions
only on the *good* steps, with loose smoke bounds (as the existing scripts do).

Gallery wiring: `docs/tutorials/README.rst` is the section header; scripts
keep the `plot_` prefix so `filename_pattern: r"plot_"` executes them;
`within_subsection_order: FileNameSortKey` already orders by the numeric
prefix.

### 5.5 Validation

**Decision: `validation.md` becomes the closing Theory chapter,
`theory/10-validation.md`** ("How we know it's right"). The exact-reference
lever is conceptual content and pairs with ch. 4 (density of states) and the
thermodynamics capstone tutorial — it reads as the payoff to the method
derivations, not as engineering background.

Content stays as-is, with two edits: update its "Worked examples" list
(currently four items) to point at the new Examples *and* Tutorials galleries,
and fix the relative links (`../../examples/...`) for the new directory depth
(`docs/src/theory/` is one level deeper than `docs/src/`).

### 5.6 Background (retained project material)

`storyline.md` and `design-unified-batched-step.md` are design rationale, not
user docs. Keep them under a "Background / design" caption (today's "Design
and roadmap"). Theory pages should *link* to `storyline.md` for architecture
rather than restate it. No content change needed beyond fixing any links if
moved into a `background/` subdir.

### 5.7 API

`api.rst` stays. One enhancement: add `.. minigallery::` directives (or
"Examples using …" backreferences via sphinx-gallery's
`doc_module=("flatwalk",)`, already set) so each documented driver links the
recipes/tutorials that exercise it. This is the cheapest, highest-value
cross-link from reference → runnable.

---

## 6. Sphinx mechanics (concrete config changes)

All in `docs/src/conf.py` unless noted.

1. **Second gallery.** Change `sphinx_gallery_conf` `examples_dirs` and
   `gallery_dirs` to parallel lists:
   ```python
   "examples_dirs": [str(ROOT / "docs" / "examples"),
                     str(ROOT / "docs" / "tutorials")],
   "gallery_dirs": ["auto_examples", "auto_tutorials"],
   ```
   Everything else (`filename_pattern`, `within_subsection_order`,
   `abort_on_example_error`, `doc_module`, `reference_url`) applies to both.
   Each source dir needs its own `README.rst`.

2. **MyST math.** Extend:
   ```python
   myst_enable_extensions = ["deflist", "fieldlist", "smartquotes",
                             "tasklist", "dollarmath", "amsmath"]
   ```
   Required for `$…$`/`$$…$$` in the Theory `.md` pages (mathjax already
   loaded).

3. **`index.rst` toctrees** — replace the four current captions with:
   ```rst
   .. toctree::
      :caption: Introduction
      intro/overview
      intro/install
      intro/the-contract
      intro/quickstart

   .. toctree::
      :caption: Tutorials
      auto_tutorials/index

   .. toctree::
      :caption: Examples
      auto_examples/index

   .. toctree::
      :caption: Theory
      theory/index            # or list each chapter; ends with theory/10-validation

   .. toctree::
      :caption: Reference
      api

   .. toctree::
      :caption: Background
      background/storyline
      background/design-unified-batched-step
   ```
   `theory/index.md` can carry its own nested `:::{toctree}` of chapters
   (MyST colon-fence) so the chapter list lives next to the prose.

4. **`.gitignore`** — add `docs/src/auto_tutorials/`.

5. **`tox.ini`** — add `{toxinidir}/docs/tutorials` to `lint_folders`.

6. **`pyproject.toml`** — add `"B"` ignore for `docs/tutorials/*` under
   `[tool.ruff.lint.per-file-ignores]` (mirrors `examples/*`), since the
   tutorial scripts use loop variables / `warnings.warn` like the examples.

---

## 7. Conventions / style guide (match the existing voice)

The current docs have a distinctive, tight register; keep it.

- **Person & tone:** second person ("you supply", "you'd replace block 1"),
  declarative, no filler. Each sentence earns its place.
- **Unicode math inline in prose** (`β`, `Δ`, `ε`, `≥`, `ln f`); reserve
  `$…$`/`.. math::` for displayed derivations in Theory.
- **`g` is the *log* density of states.** Say so wherever it first appears on
  a page; never let a reader think it's the raw count.
- **The contract table is canonical** in `intro/the-contract.md`; other pages
  link it, never re-paste it.
- **"block 1 = your physics / block 2 = verbatim wiring"** is the signature
  framing for runnable code — reuse it in recipes and tutorials.
- **Code references:** in RST/MyST prose use `:class:`/`:func:`/`:meth:`
  roles (e.g. ``:meth:`~flatwalk.WLDriver.run` ``) so they link and survive
  `--fail-on-warning`. In gallery scripts, refer to APIs in `# %%` comments;
  `doc_module` turns them into backreferences.
- **Cross-refs over duplication** (§4). Prefer `:doc:` and gallery
  `:ref:`/`minigallery` to repeating prose.
- **Smoke sizing in runnable code:** `L=4`, loose `ln_f_final` (1e-4…1e-6),
  assertions with generous bounds; cite the strict `L=8`/`1e-8` numbers in
  prose and link the repo-root validation runners.
- **File naming is load-bearing** (toctree targets, gallery prefixes). Choose
  the numbering once; renumbering later means chasing `:doc:` and `sphx_glr_`
  references under a fail-on-warning build.

---

## 8. Build & verify

Run after each section lands, not just at the end:

```bash
tox -e docs     # sphinx-build --fail-on-warning (the real gate)
tox -e lint     # ruff + sphinx-lint over docs/src and docs/{examples,tutorials}
open docs/build/html/index.html
```

Watch for, specifically:

- "document isn't included in any toctree" → page added but not wired.
- "undefined label" / "unknown document" → broken `:ref:`/`:doc:`.
- gallery tracebacks → a `plot_*` script raised (likely too-strict assertion
  or too-large `L`/`ln_f`). Time budget: the whole gallery should add seconds,
  not minutes, to the build.
- `nitpicky`-style autodoc xref misses if you tighten settings — leave
  `nitpicky` off unless you want to fix every numpy/stdlib xref.

---

## 9. Suggested order of work

1. **Config + skeleton first** (cheap, de-risks the build): second gallery in
   `conf.py`, MyST math extensions, new toctree in `index.rst`, empty
   `theory/`, `intro/`, `docs/tutorials/` with stub pages + READMEs. Build
   green with stubs before writing prose.
2. **Introduction:** split `get-started.rst` into the four `intro/` pages;
   delete `get-started.rst`; fix the inbound link. Build green.
3. **Examples:** reframe headers, add the three new recipes, trim narrative
   out of the existing Ising scripts (move it to tutorial seeds). Build green.
4. **Tutorials:** write the 5–6 narrative scripts, reusing
   `examples/ising*.py` and `beale.py`. This is the biggest writing effort.
   Build green at each step.
5. **Theory:** write the chapters; wire the "see it run" cross-links into the
   now-existing tutorials/examples.
6. **Validation + Background:** re-home, fix links, add `minigallery` to API.
7. **Final pass:** `tox -e docs && tox -e lint` clean; click every nav entry;
   confirm no orphan pages and no duplicated prose (§4 litmus test).

---

## 10. Acceptance checklist

- [ ] `tox -e docs` passes with `--fail-on-warning`; `tox -e lint` clean.
- [ ] Sidebar shows: Introduction, Tutorials, Examples, Theory, Reference,
      Background.
- [ ] Both galleries render with thumbnails and execute on the build.
- [ ] Theory renders math (dollarmath/amsmath confirmed working).
- [ ] The four-callback contract is stated once (`intro/the-contract.md`) and
      linked everywhere else.
- [ ] Every Theory chapter links its Tutorial step and Example recipe; every
      Tutorial step links its Theory chapter; every Example links its
      explainer. No paragraph is copy-pasted across sections (§4).
- [ ] Tutorials tell one continuous Ising story: MC → WL → 1/t → walkers →
      REWL (→ thermodynamics).
- [ ] No runnable script raises; all use smoke sizes with loose bounds.
- [ ] `g` is identified as the *log* density of states on first mention per
      page.
- [ ] API page surfaces example/tutorial backreferences (`minigallery`).

---

## 11. Decisions and open questions

Decided (owner-approved):

- **Tutorials are a runnable second sphinx-gallery** (not prose pages), so
  they execute and stay correct under CI, accepting the smoke-size
  constraint.
- **Validation lives under Theory** as the closing chapter
  `theory/10-validation.md` ("How we know it's right"), not under Background.

Still open for the author:

1. **Document the planned/unbuilt features?** Theory ch. 9 (higher-D OP) and
   parts of `storyline.md` describe roadmap, not shipped code. Keep them
   clearly marked "planned", or omit from Theory and leave them only in
   Background? Recommendation: a short, clearly-labelled forward-looking
   section; don't imply `BinND` exists.
2. **Plain-MC baseline in Tutorial 1** is hand-written (flatwalk has no plain
   Metropolis driver). Confirm that's acceptable as a motivating, non-flatwalk
   first step — it is the cleanest way to *show* the shortcoming that WL fixes.
```
