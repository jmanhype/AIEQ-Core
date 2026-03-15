# AIEQ-Core Architecture

## Thesis

`AIEQ-Core` should not treat research as a linear prompt chain. It should treat
research as a persistent belief graph:

- claims are the units of novelty
- assumptions are the hidden load-bearing beams
- evidence moves belief up or down
- attacks act as explicit falsifiers
- the controller chooses what to do next based on expected information gain

That ledger is the central addition that makes the hybrid system compounding
instead of stateless.

## Why this matters

Without a ledger:

- `autoresearch` can run many experiments but has weak long-term memory
- `Denario` can generate ideas, methods, and papers but much of its output is
  prose-first
- critique risks becoming generic instead of claim-specific

With a ledger:

- experiment runs become evidence objects tied to claims
- Denario outputs can be written back as claims, methods, papers, and attacks
- novelty can be scored and revisited across runs
- the next action is selected from the current epistemic bottleneck

## Proposed control loop

1. Generate or import a claim.
2. Attach assumptions that must hold for the claim to survive.
3. Run an experiment or analysis and record structured evidence.
4. Generate attacks against the claim, method, assumptions, or evidence.
5. Rank the next action by expected information gain.
6. Repeat until confidence is high enough or the claim is falsified.

## Integration boundaries

### `autoresearch`

`autoresearch` should remain the bounded experiment engine.

Expected write-back into the ledger:

- claim id
- experiment identifier or commit
- metric deltas
- runtime constraints
- artifacts such as plots or logs
- evidence direction: support, contradict, or inconclusive
- aggregate history summaries from `results.tsv`, including keep rate, crash rate, stagnation, and best frontier improvement
- separate rollups per branch, plus one preferred branch for controller execution hints

### `Denario`

`Denario` should remain the idea/method/results/paper pipeline.

Expected write-back into the ledger:

- generated claim candidate
- method artifact
- results summary
- paper artifacts
- literature-based attacks or caveats
- publication-ready synthesis after the ledger has stabilized

## Current implementation

The initial codebase now contains:

- `src/aieq_core/ledger.py`
  - persistent JSON ledger
- `src/aieq_core/models.py`
  - typed schema for claims, assumptions, evidence, attacks, artifacts, and actions
- `src/aieq_core/policy.py`
  - expected-information-gain ranking policy
- `src/aieq_core/cli.py`
  - minimal CLI for interacting with the ledger
- `src/aieq_core/adapters/autoresearch.py`
  - adapter that converts `autoresearch` run logs into ledger evidence
  - and now imports aggregate `results.tsv` histories per branch so the controller can react to series-level momentum or stagnation
- `src/aieq_core/adapters/denario.py`
  - adapter that imports Denario project artifacts into claims, evidence, attacks, and first-class method/paper artifacts
- `src/aieq_core/controller.py`
  - controller that routes work across Denario, autoresearch, and manual critique
  - and now persists decision/execution history so it can avoid blind repetition
- `src/aieq_core/runtime.py`
  - unified runtime contract for repo paths, model defaults, environment variables, and machine diagnostics
- `src/aieq_core/orchestrator.py`
  - execution plane that can launch supported Denario and `autoresearch` actions from this repo
  - and then close the loop by importing the resulting artifacts back into the ledger

The import adapters now accept controller decision ids, which means the ledger
can connect:

- the decision the controller made
- the external execution that actually ran
- the evidence or attacks imported from that execution

Execution history now also stores:

- runtime in seconds
- optional cost estimates
- inferred or user-supplied artifact quality

This lets the controller treat a cheap failed probe differently from an
expensive dead end that already consumed significant budget.

The controller now also reads aggregate `autoresearch` series metrics from
claim metadata:

- run count
- keep rate
- crash rate
- number of frontier improvements
- stagnation since the last improvement
- best improvement against the baseline `val_bpb`
- branch count
- active versus plateaued branch counts
- preferred branch identity and `results.tsv` path

That means a claim with a long, low-yield experiment history is no longer
treated the same as a fresh claim with the same current evidence count, and a
claim with one dead branch plus one live branch is no longer treated like a
fully exhausted search.

Methods and papers are now first-class artifact nodes as well. That matters for
two reasons:

- Denario imports can update the same method or paper artifact in place on re-import
- controller logic can detect methodology or paper presence without relying only on claim metadata

## Execution plane

This repo is no longer only a recommendation engine.

The execution layer now makes a specific architectural choice:

- `AIEQ-Core` is the control plane
- Denario and `autoresearch` remain external execution engines
- the integration happens through guarded subprocess boundaries plus structured imports

That means `AIEQ-Core` can expose one command surface while still preserving:

- separate Python/runtime requirements
- upstream update boundaries
- fault isolation
- clearer licensing boundaries

The execution plane currently automates:

- Denario `generate_idea`
- Denario `generate_method`
- Denario `synthesize_paper`
- `autoresearch` `run_experiment`
- `autoresearch` `reproduce_result`

It also introduces a capability-aware preflight layer:

- repo launcher detection (`.venv` or `uv`)
- provider-key checks against the configured Denario models
- GPU and `autoresearch` cache checks
- action-level readiness reporting through `doctor`

That shifts the UX from “read the controller hint and manually piece together
the environment” to “ask one repo what is runnable right now.”

## Immediate next build steps

1. Automate more controller actions, especially `triage_attack` and `collect_counterevidence`, instead of leaving them manual.
2. Expand the artifact graph to include plots, datasets, and review artifacts instead of only methods and papers.
3. Build a capability-aware operator UI around the same runtime and controller primitives.
