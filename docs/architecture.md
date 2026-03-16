# AIEQ-Core Architecture

## Thesis

`AIEQ-Core` should not treat research as a linear prompt chain. It should treat
innovation as a persistent belief graph:

- claims are the units of novelty
- assumptions are the hidden load-bearing beams
- evidence moves belief up or down
- attacks act as explicit falsifiers
- the controller chooses what to do next based on expected information gain

That ledger is the central addition that makes the system compounding instead
of stateless.

## Why this matters

Without a ledger:

- `autoresearch` can run many experiments but has weak long-term memory
- `Denario` can generate ideas, methods, and papers but much of its output is
  prose-first
- prompt or skill optimization loops forget which mutations already failed
- critique risks becoming generic instead of claim-specific

With a ledger:

- experiment runs and eval runs become evidence objects tied to claims
- Denario outputs can be written back as claims, methods, papers, and attacks
- prompt/skill optimization can be tracked with explicit targets, mutation candidates, eval suites, and promoted winners
- novelty can be scored and revisited across runs
- the next action is selected from the current epistemic bottleneck

## Dual-layer shape

The repo now has two layers:

- a **generic innovation kernel**
  - ledger
  - controller
  - policy
  - execution history
  - shared runtime config
- a **mode layer**
  - `ml_research`
  - `skill_optimizer`

The kernel owns long-term memory and decision logic. Modes own domain-specific
mutation, evaluation, import, and execution behavior.

## Proposed control loop

1. Generate or import a claim.
2. Attach assumptions, targets, and eval contracts that must hold for the claim to survive.
3. Run an experiment or evaluation and record structured evidence.
4. Generate attacks against the claim, mutation, assumptions, or evidence.
5. Rank the next action by expected information gain.
6. Repeat until confidence is high enough or the claim is falsified.

## Integration boundaries

### `ml_research`

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

### `skill_optimizer`

`skill_optimizer` is the first non-ML proof that the repo is now a generic
innovation engine.

Expected write-back into the ledger:

- target registration
- eval suite registration
- mutation candidates
- review outcomes
- repeated eval runs
- aggregate candidate scores and pass rates
- promoted winner snapshots

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

The current codebase now contains:

- `src/aieq_core/ledger.py`
  - persistent JSON ledger
- `src/aieq_core/models.py`
  - typed schema for claims, assumptions, evidence, attacks, artifacts, actions, targets, eval suites, mutation candidates, and eval runs
- `src/aieq_core/policy.py`
  - expected-information-gain ranking policy
- `src/aieq_core/modes/`
  - pluggable mode adapters for `ml_research` and `skill_optimizer`
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

Methods and papers are now first-class artifact nodes as well, and prompt
optimization has its own first-class target/eval/mutation graph. That matters
for two reasons:

- Denario imports can update the same method or paper artifact in place on re-import
- controller logic can detect methodology, promoted candidates, or eval readiness without relying only on claim metadata

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
- skill optimization `design_mutation`
- skill optimization `run_eval`
- skill optimization `analyze_failure`
- skill optimization `promote_winner`

The `autoresearch` side can now target either:

- the local machine
- or a remote SSH worker, when configured

That matters because the controller machine and the training machine do not
need to be the same host. The ledger, controller, and imports stay local, while
the bounded GPU-heavy training step can execute remotely and stream its log back
into the same control loop.

It also now has a first semantic bridge between Denario and `autoresearch`:

- if a claim has a Denario method artifact
- the execution plane rewrites `train.py` through an OpenAI-backed bridge
- validates that the generated file still preserves the `autoresearch` contract
- runs a second review pass before any GPU work is launched
- applies that rewritten file only for the current `autoresearch` run
- captures prompt/response/generated-file artifacts in the execution directory
- retries once with a repaired bridge draft if the first bridged run crashes
- then restores the worker's prior `train.py`

It also introduces a capability-aware preflight layer:

- repo launcher detection (`.venv` or `uv`)
- provider-key checks against the configured Denario models
- provider-key checks for the method bridge model
- local or remote GPU and `autoresearch` cache checks
- action-level readiness reporting through `doctor`
- per-mode readiness via `doctor --mode <mode>`

That shifts the UX from “read the controller hint and manually piece together
the environment” to “ask one repo what is runnable right now.”

## Immediate next build steps

1. Automate more controller actions, especially `triage_attack` and `collect_counterevidence`, instead of leaving them manual.
2. Expand the artifact graph to include plots, datasets, and review artifacts instead of only methods and papers.
3. Generalize the eval runners so later modes can optimize product policies, landing pages, and workflow prompts using the same target/eval contract.
4. Build a capability-aware operator UI around the same runtime and controller primitives.
