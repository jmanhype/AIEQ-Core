# Skill Optimizer Example

This folder contains a minimal prompt-optimization example for the
`skill_optimizer` mode.

- `support-skill.md`
  - a small prompt-like artifact to mutate
- `eval-suite.json`
  - one eval suite with two binary criteria

Example flow:

```bash
PYTHONPATH=src python -m aieq_core.cli init data/skill-ledger.json

PYTHONPATH=src python -m aieq_core.cli target register data/skill-ledger.json \
  --mode skill_optimizer \
  --title "Support reply skill" \
  --source-file examples/skill_optimizer/support-skill.md

PYTHONPATH=src python -m aieq_core.cli eval register data/skill-ledger.json \
  --mode skill_optimizer \
  --claim-id <claim-id> \
  --target-id <target-id> \
  --suite-file examples/skill_optimizer/eval-suite.json

PYTHONPATH=src python -m aieq_core.cli run-loop data/skill-ledger.json \
  --mode skill_optimizer \
  --iterations 3
```
