from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aieq_core.runtime import RuntimeConfig, doctor_report


class RuntimeDoctorTests(unittest.TestCase):
    def test_doctor_reports_ready_runtime_when_launchers_keys_gpu_and_cache_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            autoresearch_repo = root / "external" / "autoresearch"
            denario_repo = root / "external" / "denario"
            for repo in (autoresearch_repo, denario_repo):
                (repo / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
                (repo / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
            (autoresearch_repo / "train.py").write_text("print('train')\n", encoding="utf-8")

            home = root / "home"
            (home / ".cache" / "autoresearch").mkdir(parents=True, exist_ok=True)
            data_description = root / "data_description.md"
            data_description.write_text("Analyze a toy dataset.\n", encoding="utf-8")

            config = RuntimeConfig(
                repo_root=root,
                env_file=None,
                env={
                    "GOOGLE_API_KEY": "test-key",
                    "AIEQ_DATA_DESCRIPTION_FILE": str(data_description),
                },
                runtime_dir=root / ".aieq-runtime",
                denario_projects_dir=root / ".aieq-runtime" / "denario",
                autoresearch_output_dir=root / ".aieq-runtime" / "autoresearch",
                autoresearch_repo=autoresearch_repo,
                denario_repo=denario_repo,
                default_autoresearch_branch="main",
                autoresearch_timeout_seconds=600,
                denario_timeout_seconds=1800,
                denario_mode="fast",
                denario_idea_llm="gemini-2.0-flash",
                denario_method_llm="gemini-2.0-flash",
                denario_paper_llm="gemini-2.5-flash",
                denario_paper_journal="NONE",
                default_data_description_file=str(data_description),
            )

            def fake_which(binary: str) -> str | None:
                mapping = {
                    "git": "/usr/bin/git",
                    "nvidia-smi": "/usr/bin/nvidia-smi",
                }
                return mapping.get(binary)

            with patch("pathlib.Path.home", return_value=home), patch(
                "aieq_core.runtime.shutil.which",
                side_effect=fake_which,
            ):
                report = doctor_report(config)

            self.assertTrue(report["capabilities"]["generate_idea"]["available"])
            self.assertTrue(report["capabilities"]["generate_method"]["available"])
            self.assertTrue(report["capabilities"]["synthesize_paper"]["available"])
            self.assertTrue(report["capabilities"]["run_experiment"]["available"])


if __name__ == "__main__":
    unittest.main()
