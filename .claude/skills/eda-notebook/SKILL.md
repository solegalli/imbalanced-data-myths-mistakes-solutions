---
name: eda-notebook
description: Draft an initial exploratory data analysis (EDA) Jupyter notebook for a dataset — dataset size, variable types, missing data, constant features, categorical variable values, distributions of continuous variables, and target class balance — with real executed outputs, written in the author's own established EDA style (learned by reading their reference notebooks live from GitHub). Use this whenever the user wants a first-look/exploration notebook for a new dataset, wants to understand what's in a dataset before modeling or cleaning it, or asks for an "EDA notebook," "exploration notebook," or "initial analysis" — even if they don't name this skill directly. Works in any project with pandas/numpy/matplotlib; not tied to any specific repo's dependencies.
---

# EDA notebook

Drafts a fresh, executed Jupyter notebook giving a first look at a dataset —
size, variable types, missing data, constant features, what the categorical
variables' values actually mean, distributions of continuous variables, and
target class balance.

The point of this skill isn't a generic EDA checklist. It's to draft the 
notebook the way the author actually
writes theirs: specific section headers, specific cell granularity, a
specific first-person tone in the markdown commentary, specific code idioms
repeated across notebooks. That style lives in the author's own reference
notebooks, not in this file, so **read `references/notebook_sections.md`
first and follow its instructions to fetch and read the reference notebooks
live from GitHub before drafting a single cell.** A locally-cached summary
of "here's the style" would go stale the moment the author's conventions
evolve; fetching fresh each time doesn't.

## Why real execution, not hand-typed output

A notebook with fabricated or hand-typed "outputs" looks legitimate but
doesn't reflect what the code actually does — a `df.head()` you typed from
imagination is worse than no output at all. So this skill never hand-writes
`.ipynb` JSON directly. It assembles the notebook from a plain cell list and
actually runs it, via the bundled `scripts/build_notebook.py`, which uses
`nbformat` to build a structurally valid notebook and `nbclient` to execute
it against a real Jupyter kernel. What ships is what actually ran.

## Workflow

1. **Learn the style.** Follow `references/notebook_sections.md` step 1: fetch
   the reference notebook(s) — default to the author's own set listed there,
   or whatever notebook/repo the user points you at instead — and read them
   with `curl` + `python3 -m json` (not a summarizing web-fetch tool; you
   need exact structure, not a paraphrase). Note the section headers, cell
   granularity, markdown tone, and any recurring code idioms (e.g. a
   particular histogram-grid layout, or how categorical values get printed).
   If nothing is reachable, use the fallback structure in that same file and
   say so.

2. **Understand the new dataset.** Find out how it loads (a file path, an
   existing loader function, a DataFrame already in the user's session) and
   whether it has a target/label column — the class-balance section only
   applies if it does. Ask if this isn't clear from the request; don't guess
   at a data source.

3. **Check what's actually installed.** This skill deliberately avoids
   assuming any dependency beyond pandas/numpy/matplotlib (present in nearly
   every data science environment). If the reference notebooks use a library
   the current project doesn't have (e.g. `feature_engine`), either install
   it if the user wants that, or use the plain-pandas equivalent from the
   fallback structure (`nunique() <= 1` for constant features,
   `select_dtypes(include=["object","category"])` for categorical columns)
   — don't add a new dependency to someone's project just to match a
   template's exact code.

4. **Draft the cell spec.** Write a JSON file (a scratch file is fine — it
   doesn't need to live in the project) shaped like:

   ```json
   {
     "cells": [
       {"type": "markdown", "source": "# EDA: <dataset name>\n\n..."},
       {"type": "code", "source": "import pandas as pd\n..."}
     ]
   }
   ```

   Order sections the way the reference notebooks do (or the fallback order
   if none were reachable): load data → dataset size → variable types →
   missing data → constant features → categorical variables → distributions
   of continuous variables → target class balance (if applicable) → summary.
   Skip a section if it plainly doesn't apply — say so in one line rather
   than adding an empty subsection. Keep cells small, per the granularity
   guidance: one action, one output, per cell.

5. **Build and execute it:**

   ```bash
   python .claude/skills/eda-notebook/scripts/build_notebook.py <spec.json> <output.ipynb> --kernel <kernel-name>
   ```

   Check `jupyter kernelspec list` for what's installed and pick whichever
   kernel has the current project's dependencies — don't assume a
   particular kernel name exists. If none is set up yet, the default
   `python3` kernel works as long as pandas/numpy/matplotlib are importable
   there.

6. **Ask where to save it**, rather than assuming a fixed folder or naming
   convention — that's specific to each project, not something this skill
   should hardcode. A reasonable default to suggest is
   `eda_<dataset-slug>.ipynb` in the project's notebooks directory (or the
   current directory if there isn't one), but confirm with the user.

7. **Verify before calling it done.** Skim the produced notebook — or use
   `jupyter nbconvert --to script --stdout <file>` to read it as plain code
   — to confirm every cell executed without error and the outputs say
   something real (e.g. the categorical section actually printed unique
   values, not an empty list because the loader returned the wrong object).
   `build_notebook.py` saves a partial notebook on failure so you can see
   exactly where it broke — fix the spec and rerun rather than shipping a
   partially-executed notebook.

8. **Write the closing Summary cell from what actually showed up** in the
   executed outputs (missing-data columns found, what the categorical
   values mean, constant features, class balance) — write it last, not
   before execution.

## Notes

- This drafts an EDA for **one dataset**. Comparing several datasets side by
  side in one notebook is a different, more bespoke task — build that
  directly rather than forcing this skill's single-dataset flow to loop over
  a dataset list.
- Don't add a train/test split, encoding, or imputation step here — this is
  exploration before preprocessing decisions are made. Those decisions
  belong in whatever code the project uses once the EDA has informed them.
