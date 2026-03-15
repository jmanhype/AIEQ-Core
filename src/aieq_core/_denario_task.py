from __future__ import annotations

import json
from pathlib import Path
import sys


def _patch_mistralai_compat() -> None:
    """Backfill root-level mistralai exports expected by cmbagent/Denario."""
    try:
        import mistralai  # type: ignore
        from mistralai.client import Mistral  # type: ignore
        from mistralai.client.models import DocumentURLChunk  # type: ignore
    except Exception:
        return

    if not hasattr(mistralai, "Mistral"):
        mistralai.Mistral = Mistral
    if not hasattr(mistralai, "DocumentURLChunk"):
        mistralai.DocumentURLChunk = DocumentURLChunk


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 1:
        raise SystemExit("usage: python _denario_task.py <spec.json>")

    spec_path = Path(args[0]).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    _patch_mistralai_compat()

    from denario import Denario
    from denario.paper_agents.journal import Journal

    project_dir = spec["project_dir"]
    task = spec["task"]
    den = Denario(project_dir=project_dir, clear_project_dir=False)

    if task == "generate_idea":
        den.set_data_description(spec["data_description_input"])
        den.get_idea(mode=spec["mode"], llm=spec["idea_llm"])
    elif task == "generate_method":
        den.get_method(mode=spec["mode"], llm=spec["method_llm"])
    elif task == "synthesize_paper":
        journal_name = str(spec.get("journal", "NONE")).strip().upper() or "NONE"
        journal = getattr(Journal, journal_name, Journal.NONE)
        den.get_paper(journal=journal, llm=spec["paper_llm"])
    else:
        raise ValueError(f"Unsupported Denario task: {task}")

    payload = {
        "task": task,
        "project_dir": project_dir,
        "idea_path": str(Path(project_dir) / "input_files" / "idea.md"),
        "method_path": str(Path(project_dir) / "input_files" / "methods.md"),
        "results_path": str(Path(project_dir) / "input_files" / "results.md"),
        "paper_dir": str(Path(project_dir) / "paper"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
