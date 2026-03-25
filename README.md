# AIEQ-Core

Epistemic ledger and controller for automated research agents. Tracks claims, assumptions, evidence, and attacks as a persistent belief graph. Ranks next actions by expected information gain. Orchestrates external systems (Denario for idea/method/paper generation, autoresearch for GPU experiments) without absorbing them into one monolith.

The name stands for Axiomatic Inversion and Epistemic Quarantine:
- **Axiomatic inversion**: perturb common assumptions to surface hypotheses that ordinary loops would skip.
- **Epistemic quarantine**: reduce the influence of dominant papers and consensus so the system explores neglected directions.

## Status

v0.1.0. The ledger (schema version 7), controller, policy engine, 2 adapters, 3 modes, intake layer, and CLI are implemented. 10 test files cover the core modules. The repo has 0 runtime Python dependencies -- it uses only the standard library. External system integration (Denario, autoresearch) requires those tools to be installed separately.

## Repository layout

```
76 files total

src/aieq_core/
  models.py           -- 20 dataclasses, 11 enums (claims, evidence, attacks, artifacts, etc.)
  ledger.py           -- JSON-backed persistent belief graph, schema v7
  policy.py           -- action ranking by expected information gain
  controller.py       -- chooses next action across modes, tracks decision/execution history
  orchestrator.py     -- guarded execution layer, routes through active mode adapter
  runtime.py          -- config, provider-key detection, capability diagnostics
  intake.py           -- registers inputs, generates hypotheses, compiles protocols
  cli.py              -- CLI for all ledger and orchestration operations
  method_bridge.py    -- rewrites train.py for autoresearch runs via LLM
  _denario_task.py    -- Denario task wrapper
  adapters/
    autoresearch.py   -- parses run logs and results.tsv into ledger evidence
    denario.py        -- imports Denario project dirs into claims, evidence, attacks, artifacts
  modes/
    base.py           -- mode interface
    ml_research.py    -- Denario + autoresearch scientific/ML lab mode
    skill_optimizer.py -- prompt/SKILL.md optimization against eval suites
    repo_benchmark.py -- (stub)

tests/                -- 10 test files
  test_ledger.py
  test_controller.py
  test_orchestrator.py
  test_intake.py
  test_runtime.py
  test_method_bridge.py
  test_autoresearch_adapter.py
  test_denario_adapter.py
  test_cli_modes.py
  test_demo_script.py

examples/
  demo/
    run_demo.py       -- end-to-end demo against fixture data
    fixtures/         -- autoresearch logs, Denario project files
  skill_optimizer/
    support-skill.md  -- sample prompt target
    eval-suite.json   -- sample eval suite

docs/architecture.md
.github/workflows/ci.yml
```

## Core concepts

The ledger stores 14 entity types:

| Entity | Purpose |
|--------|---------|
| Claim | A falsifiable hypothesis to investigate |
| Assumption | Hidden load-bearing belief behind a claim |
| Evidence | Observation that moves belief up or down |
| Attack | Explicit falsification attempt |
| Artifact | Method or paper produced by Denario |
| ArtifactTarget | Mutable artifact registered for optimization |
| InputArtifact | Arbitrary input (repo, document, prompt, text) |
| InnovationHypothesis | Ranked hypothesis generated from an input |
| ProtocolDraft | Grounded protocol compiled from a hypothesis |
| EvalSuite | Test cases and scoring for optimization |
| MutationCandidate | Modified artifact under review |
| EvalRun | Single evaluation of a candidate against a case |
| DecisionRecord | Recorded controller decision |
| ExecutionRecord | Outcome of an executed decision |

## Modes

| Mode | Adapters | What it does |
|------|----------|-------------|
| ml_research | Denario, autoresearch | Generate ideas and methods via Denario, run GPU experiments via autoresearch, import results back into the ledger |
| skill_optimizer | (built-in) | Mutate a prompt or SKILL.md artifact, run eval suites, promote winners |
| repo_benchmark | (stub) | Not yet implemented |

## Controller behavior

The controller reads the full ledger state -- claims, evidence, attacks, execution history, and experiment series -- then ranks actions by expected information gain. It penalizes:
- Repeated failed actions
- Expensive dead ends (using recorded cost and runtime)
- Long experiment stagnation (using autoresearch series rollups)
- Already-completed one-shot stages

When a Denario method artifact exists, the execution plane can bridge it into a concrete `train.py` rewrite, review the draft, and retry once on failure.

## CLI

All operations go through `python -m aieq_core.cli`. Key subcommands:

| Subcommand | Purpose |
|------------|---------|
| init | Create an empty ledger |
| add-claim | Register a new hypothesis |
| add-assumption | Attach an assumption to a claim |
| add-evidence | Record an observation |
| show | Print ledger summary |
| rank-actions | List next actions by information gain |
| decide-next | Controller picks and optionally records a decision |
| run-next | Execute one step via the appropriate adapter |
| run-loop | Execute N iterations of decide+run |
| import-autoresearch-run | Import a single experiment run log |
| import-autoresearch-results | Import aggregate results.tsv series |
| import-denario-project | Import a Denario project directory |
| ingest register | Register arbitrary input for hypothesis generation |
| generate-hypotheses | Generate ranked hypotheses from an input |
| compile-protocol | Compile a hypothesis into a protocol draft |
| materialize-protocol | Turn a protocol into a real claim and target |
| target register | Register an optimization target |
| eval register | Register an eval suite |
| promote-winner | Promote the best evaluated candidate |
| doctor | Check runtime readiness (local or remote) |
| mode list | List registered modes |

## External systems

AIEQ-Core does not bundle Denario or autoresearch. It calls them in their own repos and environments:

- **Denario**: multi-agent idea/method/results/paper pipeline. AIEQ-Core imports its output directories.
- **autoresearch**: single-file experiment engine. AIEQ-Core launches it (locally or via SSH on a remote GPU box) and imports run logs.

## Setup

```bash
git clone https://github.com/jmanhype/AIEQ-Core.git
cd AIEQ-Core
cp .env.example .env          # set API keys and paths
pip install -e .              # zero runtime dependencies
```

Verify readiness:

```bash
PYTHONPATH=src python -m aieq_core.cli doctor
```

Run the demo:

```bash
python examples/demo/run_demo.py
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Configuration

Set in `.env`:

| Variable | Purpose |
|----------|---------|
| AIEQ_AUTORESEARCH_REMOTE_HOST | SSH host for remote GPU experiments |
| AIEQ_AUTORESEARCH_REMOTE_REPO | Path to autoresearch on the remote host |
| AIEQ_METHOD_BRIDGE_ENABLED | Enable LLM-based train.py rewriting |
| AIEQ_METHOD_BRIDGE_MODEL | Model for method bridge (e.g., gpt-4.1) |

## Design decisions

**Zero runtime dependencies**: The core uses only Python stdlib. This keeps the ledger and controller portable. External adapters shell out to Denario and autoresearch in their own environments.

**JSON-backed ledger**: The ledger is a single JSON file. This is simple to inspect, diff, and version-control. It will not scale to millions of entities, but research workflows rarely produce that volume.

**Controller penalizes, does not forbid**: Failed or expensive actions get lower priority, but the controller can still recommend them if the information gain is high enough. This prevents premature pruning of promising but difficult directions.

**Adapters import, not embed**: Denario and autoresearch keep their own repos and runtimes. AIEQ-Core only reads their output artifacts. This avoids version coupling and lets each tool evolve independently.

**Intake can block**: The intake layer is allowed to produce a "blocked" protocol if the input is too abstract to optimize. This is intentional -- it signals that a human needs to extract the concrete artifact before automation can proceed.

## Limitations

- No web UI; all interaction is via CLI
- Ledger is a single JSON file, not a database
- `repo_benchmark` mode is a stub
- Remote autoresearch requires manual SSH key setup and a pre-configured host
- Method bridge depends on OpenAI API access
- No multi-user or concurrent access support for the ledger file
- Branch-level exploration comparison is per-branch rollup only; no direct cross-branch efficiency analysis yet

## License

MIT.
