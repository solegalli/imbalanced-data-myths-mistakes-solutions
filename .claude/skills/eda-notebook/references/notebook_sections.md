# EDA notebook conventions

This skill drafts EDA notebooks in the author's own style, learned by
reading their actual reference notebooks rather than from a static template
baked into this file. That keeps the skill small, portable across projects,
and automatically up to date if the author's conventions evolve — a copy
pasted into this file would silently go stale.

## Step 1: read the reference notebooks fresh, every time

Default reference notebooks (override these if the user points you at a
different repo or notebook of theirs to imitate instead):

```
https://raw.githubusercontent.com/solegalli/resampling-experiments/main/notebooks/exploratory-data-analysis/00-analysis-of-datasets-reel-imblearn.ipynb
https://raw.githubusercontent.com/solegalli/resampling-experiments/main/notebooks/exploratory-data-analysis/01-analysis-of-datasets-uci-openml.ipynb
https://raw.githubusercontent.com/solegalli/resampling-experiments/main/notebooks/exploratory-data-analysis/02-datasets-with-strong-class-separation.ipynb
```

Fetch and parse them directly with `curl` + Python's `json` module — **not**
via a web-fetch tool that summarizes through another model. A notebook's
value here is in its exact structure (header wording, cell boundaries, code
idioms), and a summary would blur exactly the details you need to copy:

```bash
curl -s "<raw-url>" | python3 -c "
import json, sys
nb = json.load(sys.stdin)
for i, c in enumerate(nb['cells']):
    print(f'--- cell {i} [{c[\"cell_type\"]}] ---')
    print(''.join(c['source'])[:400])
    print()
"
```

If the URLs are unreachable (offline, repo moved, user has no reference
notebook), fall back to the generic structure in "Fallback structure" below
and say so — don't silently invent a style and call it theirs.

## Step 2: extract the conventions, don't just extract the checks

Reading the fetched notebooks, look specifically for:

- **Section headers**: the exact `##` (and `###` for per-dataset deep dives)
  wording and ordering used, e.g. "Load data", "Missing data", "Constant
  features", "Categorical variables".
- **Cell granularity**: does one code cell do one thing (one print, one
  plot, one computed value)? Existing example: a cell that computes
  `find_categorical_variables` output is separate from the cell that prints
  `dtypes`, even though both are "variable types" — don't merge them.
- **Markdown tone**: first-person, short, specific to the actual finding —
  not generic boilerplate. E.g. "There are 3 datasets with categorical data
  in KEEL. I will add an encoding step before training models" rather than
  "This section analyzes the categorical variables."
- **Recurring code idioms worth reusing directly**: e.g. a histogram grid
  sized by `n_cols=5` and `n_rows = ceil(n_features / n_cols)`, or a loop
  that prints each categorical variable's name followed by its unique
  values. Reuse these idioms verbatim in the new notebook rather than
  inventing a different-looking equivalent — that consistency is the whole
  point of using a template.

## Fallback structure (only if the reference notebooks can't be fetched)

A reasonable default order, using only pandas/numpy/matplotlib so it runs
anywhere without extra installs:

1. **Load data** — one cell to load, a separate cell to show `.head()`.
2. **Dataset size** — `X.shape[0]` observations, `X.shape[1]` features.
3. **Variable types** — `X.dtypes`; then a separate cell identifying
   categorical columns: `X.select_dtypes(include=["object", "category"]).columns`.
4. **Missing data** — `X.isnull().mean().sort_values(ascending=False)`; if
   any is non-zero, a bar plot (`kind="bar"`, `figsize=(20, 4)`).
5. **Constant features** — columns where `X.nunique(dropna=False) <= 1`.
6. **Categorical variables** — for each categorical column, print the name
   and `X[col].unique()` (or `.value_counts().head(20)` if there are many
   distinct values) so the reader sees what the categories actually mean,
   not just that they exist.
7. **Distributions of continuous variables** — a histogram grid over the
   numeric columns (`X.select_dtypes(include="number")`), e.g. 5 columns
   wide, row count computed from feature count.
8. **Target class balance** (only if there's a target) — `y.value_counts()`
   and the majority/minority ratio.
9. **Summary** (markdown) — the concrete findings, written after seeing the
   real outputs, not before.

## Cell granularity and tone, regardless of source

Whether copying the fetched notebooks' exact structure or falling back to
the generic one above:

- One idea, one output, per code cell — a reader should be able to tell
  what an output belongs to without reading the code above it.
- A markdown cell explains *why* the next cell exists or states the finding
  *after* the code ran — it doesn't restate what the code obviously does.
- Write the closing summary last, based on what the executed outputs
  actually showed.
