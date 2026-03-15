from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AIEQ-Core end-to-end demo.")
    parser.add_argument(
        "--workspace",
        default=str(repo_root() / "examples" / "demo" / "output"),
        help="Directory where demo outputs should be written.",
    )
    args = parser.parse_args()

    root = repo_root()
    fixtures = root / "examples" / "demo" / "fixtures"
    workspace = Path(args.workspace).resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    ledger_path = workspace / "ledger.json"

    denario_project = fixtures / "denario_project"
    autoresearch_dir = fixtures / "autoresearch"

    write_json(workspace / "init.json", run_cli("init", str(ledger_path)))

    denario_import = run_cli(
        "import-denario-project",
        str(ledger_path),
        "--project-dir",
        str(denario_project),
        "--results-direction",
        "contradict",
        "--novelty",
        "0.78",
        "--falsifiability",
        "0.72",
    )
    write_json(workspace / "denario_import.json", denario_import)
    claim_id = denario_import["claim"]["id"]

    main_series = run_cli(
        "import-autoresearch-results",
        str(ledger_path),
        "--claim-id",
        claim_id,
        "--results-tsv",
        str(autoresearch_dir / "main_results.tsv"),
        "--branch",
        "main",
    )
    radical_series = run_cli(
        "import-autoresearch-results",
        str(ledger_path),
        "--claim-id",
        claim_id,
        "--results-tsv",
        str(autoresearch_dir / "radical_results.tsv"),
        "--branch",
        "radical",
    )
    write_json(workspace / "main_series.json", main_series)
    write_json(workspace / "radical_series.json", radical_series)

    initial_decision = run_cli(
        "decide-next",
        str(ledger_path),
        "--backlog-limit",
        "5",
        "--record",
    )
    write_json(workspace / "initial_decision.json", initial_decision)
    decision_record = initial_decision["decision_record"]
    primary_action = initial_decision["decision"]["primary_action"]

    if primary_action["action_type"] != "run_experiment":
        raise RuntimeError(
            f"Demo expected run_experiment, got {primary_action['action_type']!r} instead."
        )

    run_import = run_cli(
        "import-autoresearch-run",
        str(ledger_path),
        "--claim-id",
        claim_id,
        "--decision-id",
        decision_record["id"],
        "--run-log",
        str(autoresearch_dir / "followup_run.log"),
        "--commit",
        "radical-r4-demo",
        "--branch",
        "radical",
        "--description",
        "Radical branch follow-up",
        "--status",
        "keep",
        "--baseline-bpb",
        "1.000500",
        "--cost-usd",
        "0.42",
    )
    write_json(workspace / "run_import.json", run_import)

    final_snapshot = run_cli("show", str(ledger_path), "--claim-id", claim_id)
    final_decision = run_cli("decide-next", str(ledger_path), "--backlog-limit", "5")
    write_json(workspace / "final_snapshot.json", final_snapshot)
    write_json(workspace / "final_decision.json", final_decision)

    summary = {
        "claim_id": claim_id,
        "claim_title": denario_import["claim"]["title"],
        "workspace": str(workspace),
        "initial_action": primary_action["action_type"],
        "initial_executor": primary_action["executor"],
        "recorded_decision_id": decision_record["id"],
        "followup_evidence_direction": run_import["evidence"]["direction"],
        "final_claim_status": final_snapshot["claim"]["status"],
        "final_confidence": final_snapshot["claim"]["confidence"],
        "final_next_action": final_decision["decision"]["primary_action"]["action_type"],
    }
    write_json(workspace / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_cli(*args: str) -> dict[str, Any]:
    root = repo_root()
    env = os.environ.copy()
    pythonpath_entries = [str(root / "src")]
    existing = env.get("PYTHONPATH", "").strip()
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    completed = subprocess.run(
        [sys.executable, "-m", "aieq_core.cli", *args],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
