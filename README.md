# AIEQ-Core

AIEQ-Core is the orchestration and memory layer for an automated research
system designed to generate, test, write up, and stress-test unconventional
scientific hypotheses.

The name comes from **Axiomatic Inversion & Epistemic Quarantine**:

- **Axiomatic inversion**: deliberately perturb or invert common assumptions to
  surface hypotheses that ordinary research loops would not propose.
- **Epistemic quarantine**: temporarily reduce the influence of dominant papers,
  citations, or consensus narratives so the system is forced to explore
  neglected directions.

## Core idea

The most important addition in this repository is the **epistemic ledger**.

Instead of treating research as a one-shot prompt chain, AIEQ-Core treats it as
a persistent belief graph:

- claims are the units of novelty
- assumptions are the hidden load-bearing beams
- evidence moves belief up or down
- attacks act as explicit falsifiers
- the controller ranks what to do next by expected information gain

That is the piece that makes the system compounding rather than stateless.

## Project intent

The broader concept is still a hybrid research engine:

1. **Brain**: generate hypotheses using AIEQ-style inversion and quarantine
   rules.
2. **Hands**: run rapid experiments to evaluate whether those hypotheses hold
   up empirically.
3. **Mouth**: synthesize validated results into a structured scientific writeup.
4. **Critic**: attack the hypothesis, methods, and paper before a human reviews
   it.

In other words, this repository is not meant to be just another auto-research
loop. It is intended to be the coordination core for a research pipeline that
explicitly searches for ideas outside the current local optimum.

## Proposed system shape

- `hypothesis engine`
  - Generates candidate claims by modifying assumptions, constraints, or priors.
- `experiment runner`
  - Sends candidates to downstream evaluation and training workflows.
- `paper synthesis`
  - Converts validated runs into structured research artifacts.
- `adversarial review`
  - Red-teams claims, methods, and evidence before publication or handoff.

## What exists now

The repo now contains the first concrete subsystem:

- `src/aieq_core/models.py`
  - typed schema for claims, assumptions, evidence, attacks, artifacts, and ranked actions
- `src/aieq_core/ledger.py`
  - persistent JSON ledger with derived belief/status updates
  - now includes first-class artifact nodes for methods and papers
- `src/aieq_core/policy.py`
  - action ranking based on expected information gain
- `src/aieq_core/adapters/autoresearch.py`
  - adapter that parses `autoresearch` run logs and aggregate `results.tsv` histories into ledger evidence plus controller-facing series metrics
- `src/aieq_core/adapters/denario.py`
  - adapter that imports Denario project directories into claims, evidence, attacks, method artifacts, and paper artifacts
- `src/aieq_core/controller.py`
  - controller that chooses between generation, experimentation, critique, reproduction, and synthesis
- `src/aieq_core/cli.py`
  - minimal CLI for creating and inspecting the ledger
- `docs/architecture.md`
  - architectural rationale and integration boundaries
- `external/autoresearch`
  - cloned upstream experiment engine
- `external/denario`
  - cloned upstream research/paper pipeline

## Quick start

Create an empty ledger:

```bash
PYTHONPATH=src python -m aieq_core.cli init data/ledger.json
```

Add a claim and some supporting structure:

```bash
PYTHONPATH=src python -m aieq_core.cli add-claim data/ledger.json \
  --title "Sparse attention improves short-budget pretraining" \
  --statement "A sparse attention variant should reduce validation bpb." \
  --novelty 0.8 \
  --falsifiability 0.9

PYTHONPATH=src python -m aieq_core.cli add-assumption data/ledger.json \
  --claim-id <claim-id> \
  --text "The sparse kernel does not destabilize training." \
  --risk 0.7

PYTHONPATH=src python -m aieq_core.cli add-evidence data/ledger.json \
  --claim-id <claim-id> \
  --summary "First run improved validation bpb by 0.003." \
  --direction support \
  --strength 0.8 \
  --confidence 0.9 \
  --source-type autoresearch \
  --source-ref commit:abc1234
```

Inspect the ledger and rank next actions:

```bash
PYTHONPATH=src python -m aieq_core.cli show data/ledger.json
PYTHONPATH=src python -m aieq_core.cli rank-actions data/ledger.json --limit 5
```

Import a real `autoresearch` run:

```bash
PYTHONPATH=src python -m aieq_core.cli import-autoresearch-run data/ledger.json \
  --claim-id <claim-id> \
  --run-log /path/to/run.log \
  --commit abc1234 \
  --description "Sparse attention pilot" \
  --status keep \
  --baseline-bpb 1.000500
```

If that run came from a recorded controller decision, thread the id through the
import and let the adapter auto-record the execution outcome:

```bash
PYTHONPATH=src python -m aieq_core.cli import-autoresearch-run data/ledger.json \
  --claim-id <claim-id> \
  --decision-id <decision-id> \
  --run-log /path/to/run.log \
  --status keep \
  --cost-usd 0.42
```

Import an aggregate `autoresearch` history so the controller can reason over
the whole experiment series instead of one run at a time:

```bash
PYTHONPATH=src python -m aieq_core.cli import-autoresearch-results data/ledger.json \
  --claim-id <claim-id> \
  --results-tsv /path/to/results.tsv \
  --branch main
```

That series import stores a branch-specific rollup and updates the claim-wide
aggregate view. You can import more than one branch for the same claim:

```bash
PYTHONPATH=src python -m aieq_core.cli import-autoresearch-results data/ledger.json \
  --claim-id <claim-id> \
  --results-tsv /path/to/radical-results.tsv \
  --branch radical
```

The adapter keeps per-branch summaries and also chooses one preferred branch to
drive controller hints and scoring. The stored rollups include:

- run count
- keep rate
- crash rate
- frontier improvements
- stagnation since the last real improvement
- best `val_bpb` delta against the baseline
- active branch count
- plateau branch count

Import a Denario project directory:

```bash
PYTHONPATH=src python -m aieq_core.cli import-denario-project data/ledger.json \
  --project-dir /path/to/denario/project \
  --results-direction support \
  --novelty 0.75 \
  --falsifiability 0.60
```

That import now materializes:

- one claim
- one results evidence record when `results.md` exists
- one method artifact when `methods.md` exists
- one paper artifact per file in `paper/`
- literature and referee attacks when those files exist

For a recorded `generate_idea`, `generate_method`, or `synthesize_paper`
decision, pass the decision id here and the adapter will attach the resulting
claim/evidence/attacks to the execution record automatically:

```bash
PYTHONPATH=src python -m aieq_core.cli import-denario-project data/ledger.json \
  --project-dir /path/to/denario/project \
  --decision-id <decision-id> \
  --results-direction support \
  --runtime-seconds 480 \
  --cost-usd 1.75
```

Ask the controller for the next move:

```bash
PYTHONPATH=src python -m aieq_core.cli decide-next data/ledger.json --backlog-limit 5
```

Persist the controller decision and then record the outcome:

```bash
PYTHONPATH=src python -m aieq_core.cli decide-next data/ledger.json --record

PYTHONPATH=src python -m aieq_core.cli record-execution data/ledger.json \
  --decision-id <decision-id> \
  --status succeeded \
  --notes "Imported the new run and updated the claim."
```

The controller now reads this execution history and discounts repeated failed or
already-completed one-shot stages instead of recommending them forever.
It also reads runtime, cost, and inferred artifact quality, so expensive dead
ends are penalized more aggressively than cheap exploratory misses.
If an `autoresearch` results series is present, it also reads that aggregate
history so long stagnation or crash-heavy trajectories push work toward
assumption-challenge and critique instead of blind re-runs. When more than one
branch history is imported, the controller uses the preferred branch for direct
experiment hints while still considering cross-branch breadth and plateau
signals.
It also reads first-class Denario method and paper artifacts, so methodology and
paper presence no longer depend on ad hoc claim metadata alone.

## External systems

Two upstream repos are cloned locally for integration work:

- [`external/autoresearch`](/Users/speed/AIEQ-Core/external/autoresearch)
  - single-file experiment engine centered on `train.py`
- [`external/denario`](/Users/speed/AIEQ-Core/external/denario)
  - multi-agent idea, method, results, and paper pipeline

The intended shape is to keep AIEQ-Core above them as the controller and memory
layer, rather than merging their internals into one monolith.

## Status

This repo is still early, but the ledger core, controller, and first two
integration adapters are in place.

## Near-term build goals

1. Compare branch-level exploration efficiency directly instead of only picking a preferred branch.
2. Expand the artifact graph beyond methods and papers to include plots, datasets, and reviews.
3. Add richer execution metadata so the controller can learn from crash
   signatures and artifact quality in addition to runtime and cost.

## Immediate next steps

- Compare exploration efficiency across autoresearch branches explicitly
- Add branch-aware reports or dashboards on top of the ledger summaries
- Expand first-class artifacts beyond Denario methods and papers
- Create the initial commit
