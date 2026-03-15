# AIEQ-Core Demo

This demo runs the real `aieq_core` CLI against checked-in fixture data.

It performs the following flow:

1. creates a fresh ledger
2. imports a Denario project fixture
3. imports two `autoresearch` branch histories
4. records the controller's next action
5. imports a follow-up `autoresearch` run to close that decision
6. asks the controller for the next action again

Run it from the repo root:

```bash
python examples/demo/run_demo.py
```

By default the script writes JSON outputs into `examples/demo/output/`.
That folder is git-ignored so you can re-run the demo without polluting the
repo.

The most important output files are:

- `summary.json`
- `initial_decision.json`
- `run_import.json`
- `final_snapshot.json`
- `final_decision.json`
