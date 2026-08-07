#!/usr/bin/env python3
"""
Assemble a Jupyter notebook from a JSON cell spec and (by default) execute
it against a real kernel.

Why this exists: hand-writing .ipynb JSON (cell ids, execution_count,
outputs schema) is easy to get subtly wrong, and a notebook with
fabricated/hand-typed "outputs" is misleading. Composing cells with
nbformat and running them with nbclient guarantees a structurally valid
notebook whose outputs are exactly what the code produced.

Usage:
    python build_notebook.py <spec.json> <output.ipynb> [--kernel NAME] [--no-execute]

spec.json format:
{
  "cells": [
    {"type": "markdown", "source": "# Title\n\nSome intro text"},
    {"type": "code", "source": "import pandas as pd\ndf = pd.read_csv('x.csv')"}
  ]
}
"""
import argparse
import json
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build(spec_path, output_path, kernel_name, execute, timeout):
    with open(spec_path) as f:
        spec = json.load(f)

    cells = []
    for c in spec["cells"]:
        if c["type"] == "markdown":
            cells.append(new_markdown_cell(c["source"]))
        elif c["type"] == "code":
            cells.append(new_code_cell(c["source"]))
        else:
            raise ValueError(f"Unknown cell type: {c['type']!r} (use 'markdown' or 'code')")

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": kernel_name,
        "language": "python",
        "name": kernel_name,
    }

    if execute:
        # Imported lazily: only needed on the execute path, so --no-execute
        # works even in an environment without a Jupyter kernel manager.
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError

        client = NotebookClient(nb, kernel_name=kernel_name, timeout=timeout)
        try:
            client.execute()
        except CellExecutionError:
            # Save the partially executed notebook so the failing cell and
            # its traceback are visible for debugging, then re-raise so the
            # caller knows execution didn't fully succeed.
            nbformat.write(nb, output_path)
            print(f"Execution failed partway through; partial notebook saved to {output_path}", file=sys.stderr)
            raise

    nbformat.write(nb, output_path)
    print(f"Notebook written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="Path to the JSON cell spec")
    parser.add_argument("output", help="Path to write the .ipynb to")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name to execute with (check `jupyter kernelspec list` for what's installed)")
    parser.add_argument("--no-execute", action="store_true", help="Only assemble the notebook, don't run it")
    parser.add_argument("--timeout", type=int, default=600, help="Per-cell execution timeout in seconds")
    args = parser.parse_args()
    build(args.spec, args.output, args.kernel, execute=not args.no_execute, timeout=args.timeout)
