# Contributing

## Scope

AIEQ-Core is the orchestration and memory layer around automated research
agents. The highest-value contributions are the ones that make the controller
more trustworthy, the ledger more explicit, or the adapters more reproducible.

## Development setup

Use Python `3.10+`.

From the repo root:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the full test suite:

```bash
python -m unittest discover -s tests -v
```

Run the end-to-end demo:

```bash
python examples/demo/run_demo.py
```

## Repo conventions

- Keep edits ASCII unless the file already uses Unicode for a real reason.
- Prefer small, reviewable changes over broad refactors.
- Update tests when behavior changes.
- Update docs when CLI behavior, controller logic, or artifact semantics change.
- Do not commit generated demo output from `examples/demo/output/`.
- Do not vendor the nested upstream repos under `external/` into AIEQ-Core commits.

## Good contribution targets

- new ledger nodes or metrics that improve controller decisions
- stronger adapter fidelity for `autoresearch` or Denario artifacts
- reproducibility improvements for the demo flow
- controller policies that reduce blind repetition or reward better evidence
- artifact graph expansion for plots, datasets, and reviews

## Pull requests

Before opening a pull request:

1. Run `python -m unittest discover -s tests -v`.
2. If you touched the demo flow, run `python examples/demo/run_demo.py`.
3. Summarize the behavioral change, not just the file diff.
4. Call out any residual risks, assumptions, or missing follow-up work.

Use the pull request template and link any relevant issue.

## Issues

Use the GitHub issue templates for:

- bug reports
- feature requests
- research proposals

When reporting a bug, include the command you ran, the expected result, and the
actual result. When proposing a feature, explain which part of the research
loop gets better and how success should be measured.
