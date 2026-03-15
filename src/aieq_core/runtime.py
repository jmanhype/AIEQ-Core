from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any

from .models import ActionType, serialize_dataclass

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE_NAME = ".env"


def parse_env_file(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}

    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def merge_env(defaults: dict[str, str], overrides: dict[str, str] | None = None) -> dict[str, str]:
    merged = dict(defaults)
    merged.update(dict(overrides or os.environ))
    return merged


def parse_bool(value: str, *, default: bool = False) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "claim"


def resolve_path(value: str, *, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def provider_for_model(model_name: str) -> str:
    name = model_name.strip().lower()
    if not name:
        return "unknown"
    if name.startswith("gemini"):
        return "google"
    if name.startswith("claude") or "anthropic" in name:
        return "anthropic"
    if name.startswith("gpt") or name.startswith("o") or "openai" in name:
        return "openai"
    if "sonar" in name or "perplexity" in name:
        return "perplexity"
    return "unknown"


def provider_env_key(provider: str) -> str:
    return {
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
    }.get(provider, "")


def capability_key_for_action(action_type: ActionType) -> str:
    mapping = {
        ActionType.GENERATE_IDEA: "generate_idea",
        ActionType.GENERATE_METHOD: "generate_method",
        ActionType.RUN_EXPERIMENT: "run_experiment",
        ActionType.REPRODUCE_RESULT: "run_experiment",
        ActionType.SYNTHESIZE_PAPER: "synthesize_paper",
    }
    return mapping.get(action_type, "manual_only")


@dataclass(slots=True)
class ProjectLauncher:
    available: bool
    kind: str
    command_prefix: list[str]
    detail: str


@dataclass(slots=True)
class RemoteAutoresearchWorker:
    enabled: bool
    host: str
    repo_path: str
    ssh_path: str
    reachable: bool
    python_available: bool
    uv_available: bool
    gpu_available: bool
    cache_available: bool
    repo_available: bool
    platform: str
    blocked_by: list[str]
    detail: str


@dataclass(slots=True)
class RuntimeConfig:
    repo_root: Path
    env_file: Path | None
    env: dict[str, str]
    runtime_dir: Path
    denario_projects_dir: Path
    autoresearch_output_dir: Path
    autoresearch_repo: Path
    denario_repo: Path
    autoresearch_remote_host: str
    autoresearch_remote_repo: str
    default_autoresearch_branch: str
    autoresearch_timeout_seconds: int
    denario_timeout_seconds: int
    denario_mode: str
    denario_idea_llm: str
    denario_method_llm: str
    denario_paper_llm: str
    denario_paper_journal: str
    default_data_description_file: str
    method_bridge_enabled: bool
    method_bridge_model: str
    method_bridge_timeout_seconds: int

    @classmethod
    def load(cls, *, env_file: str | Path | None = None) -> "RuntimeConfig":
        repo_root = REPO_ROOT
        resolved_env_file: Path | None
        if env_file is None:
            candidate = repo_root / DEFAULT_ENV_FILE_NAME
            resolved_env_file = candidate if candidate.exists() else None
        else:
            resolved_env_file = resolve_path(str(env_file), root=repo_root)

        env_defaults = parse_env_file(resolved_env_file)
        env = merge_env(env_defaults)

        runtime_dir = resolve_path(env.get("AIEQ_RUNTIME_DIR", ".aieq-runtime"), root=repo_root)
        denario_projects_dir = resolve_path(
            env.get("AIEQ_DENARIO_PROJECTS_DIR", str(runtime_dir / "denario")),
            root=repo_root,
        )
        autoresearch_output_dir = resolve_path(
            env.get("AIEQ_AUTORESEARCH_OUTPUT_DIR", str(runtime_dir / "autoresearch")),
            root=repo_root,
        )
        autoresearch_repo = resolve_path(
            env.get("AIEQ_AUTORESEARCH_REPO", "external/autoresearch"),
            root=repo_root,
        )
        denario_repo = resolve_path(
            env.get("AIEQ_DENARIO_REPO", "external/denario"),
            root=repo_root,
        )

        return cls(
            repo_root=repo_root,
            env_file=resolved_env_file,
            env=env,
            runtime_dir=runtime_dir,
            denario_projects_dir=denario_projects_dir,
            autoresearch_output_dir=autoresearch_output_dir,
            autoresearch_repo=autoresearch_repo,
            denario_repo=denario_repo,
            autoresearch_remote_host=env.get("AIEQ_AUTORESEARCH_REMOTE_HOST", "").strip(),
            autoresearch_remote_repo=env.get(
                "AIEQ_AUTORESEARCH_REMOTE_REPO",
                "",
            ).strip(),
            default_autoresearch_branch=env.get("AIEQ_AUTORESEARCH_BRANCH", "main").strip() or "main",
            autoresearch_timeout_seconds=int(env.get("AIEQ_AUTORESEARCH_TIMEOUT_SECONDS", "600")),
            denario_timeout_seconds=int(env.get("AIEQ_DENARIO_TIMEOUT_SECONDS", "1800")),
            denario_mode=env.get("AIEQ_DENARIO_MODE", "fast").strip() or "fast",
            denario_idea_llm=env.get("AIEQ_DENARIO_IDEA_LLM", "gemini-2.0-flash").strip()
            or "gemini-2.0-flash",
            denario_method_llm=env.get("AIEQ_DENARIO_METHOD_LLM", "gemini-2.0-flash").strip()
            or "gemini-2.0-flash",
            denario_paper_llm=env.get("AIEQ_DENARIO_PAPER_LLM", "gemini-2.5-flash").strip()
            or "gemini-2.5-flash",
            denario_paper_journal=env.get("AIEQ_DENARIO_PAPER_JOURNAL", "NONE").strip()
            or "NONE",
            default_data_description_file=env.get("AIEQ_DATA_DESCRIPTION_FILE", "").strip(),
            method_bridge_enabled=parse_bool(
                env.get("AIEQ_METHOD_BRIDGE_ENABLED", "1"),
                default=True,
            ),
            method_bridge_model=env.get("AIEQ_METHOD_BRIDGE_MODEL", "gpt-4.1").strip()
            or "gpt-4.1",
            method_bridge_timeout_seconds=int(
                env.get("AIEQ_METHOD_BRIDGE_TIMEOUT_SECONDS", "120")
            ),
        )

    def subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        for key, value in self.env.items():
            env.setdefault(key, value)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def use_remote_autoresearch(self) -> bool:
        return bool(self.autoresearch_remote_host and self.autoresearch_remote_repo)

    def launcher_for_project(self, project_dir: str | Path) -> ProjectLauncher:
        project_path = Path(project_dir)
        if not project_path.exists():
            return ProjectLauncher(
                available=False,
                kind="missing",
                command_prefix=[],
                detail=f"Project directory does not exist: {project_path}",
            )

        venv_python = project_path / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if venv_python.exists():
            return ProjectLauncher(
                available=True,
                kind="venv",
                command_prefix=[str(venv_python)],
                detail=str(venv_python),
            )

        if shutil.which("uv"):
            return ProjectLauncher(
                available=True,
                kind="uv",
                command_prefix=["uv", "run", "--project", str(project_path), "python"],
                detail="uv run --project",
            )

        return ProjectLauncher(
            available=False,
            kind="missing",
            command_prefix=[],
            detail="No project launcher found. Install uv or create a local .venv.",
        )

    def ensure_runtime_dirs(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.denario_projects_dir.mkdir(parents=True, exist_ok=True)
        self.autoresearch_output_dir.mkdir(parents=True, exist_ok=True)

    def execution_dir(self, decision_id: str) -> Path:
        path = self.runtime_dir / "executions" / decision_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def denario_project_dir_for_decision(
        self,
        *,
        decision_id: str,
        claim_id: str = "",
        claim_title: str = "",
    ) -> Path:
        token = claim_id.strip() or slugify(claim_title or "bootstrap")
        return self.denario_projects_dir / f"{token}-{decision_id}"

    def autoresearch_results_tsv_for_claim(self, *, claim_id: str, branch: str) -> Path:
        claim_dir = self.autoresearch_output_dir / claim_id / branch
        claim_dir.mkdir(parents=True, exist_ok=True)
        return claim_dir / "results.tsv"

    def default_data_description_path(self) -> Path | None:
        if not self.default_data_description_file:
            return None
        return resolve_path(self.default_data_description_file, root=self.repo_root)

    def remote_ssh_base_args(self) -> list[str]:
        if not self.use_remote_autoresearch():
            return []
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
        ]

    def remote_ssh_command(self, remote_command: str) -> list[str]:
        if not self.use_remote_autoresearch():
            return []
        return [
            *self.remote_ssh_base_args(),
            self.autoresearch_remote_host,
            remote_command,
        ]

    def remote_shell_path(self, value: str) -> str:
        if value.startswith("~/"):
            return "$HOME/" + shlex.quote(value[2:])
        return shlex.quote(value)


def probe_remote_autoresearch_worker(
    config: RuntimeConfig,
    *,
    timeout_seconds: int = 5,
) -> RemoteAutoresearchWorker | None:
    if not config.use_remote_autoresearch():
        return None

    ssh_path = shutil.which("ssh") or ""
    blocked_by: list[str] = []
    if not ssh_path:
        blocked_by.append("Local `ssh` binary is not available.")
        return RemoteAutoresearchWorker(
            enabled=True,
            host=config.autoresearch_remote_host,
            repo_path=config.autoresearch_remote_repo,
            ssh_path="",
            reachable=False,
            python_available=False,
            uv_available=False,
            gpu_available=False,
            cache_available=False,
            repo_available=False,
            platform="",
            blocked_by=blocked_by,
            detail="ssh unavailable",
        )

    remote_repo = config.remote_shell_path(config.autoresearch_remote_repo)
    probe_script = f"""
set -e
if command -v python3 >/dev/null 2>&1; then
  echo PYTHON_OK
else
  echo PYTHON_MISSING
fi
if command -v uv >/dev/null 2>&1; then
  echo UV_OK
else
  echo UV_MISSING
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  echo GPU_OK
else
  echo GPU_MISSING
fi
if [ -d ~/.cache/autoresearch ]; then
  echo CACHE_OK
else
  echo CACHE_MISSING
fi
if [ -d {remote_repo} ]; then
  echo REPO_OK
else
  echo REPO_MISSING
fi
uname -s 2>/dev/null || echo PLATFORM_UNKNOWN
"""

    try:
        completed = subprocess.run(
            config.remote_ssh_command(f"bash -lc {shlex.quote(probe_script)}"),
            cwd=config.repo_root,
            env=config.subprocess_env(),
            timeout=timeout_seconds,
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        blocked_by.append(f"Remote worker probe failed: {exc}")
        return RemoteAutoresearchWorker(
            enabled=True,
            host=config.autoresearch_remote_host,
            repo_path=config.autoresearch_remote_repo,
            ssh_path=ssh_path,
            reachable=False,
            python_available=False,
            uv_available=False,
            gpu_available=False,
            cache_available=False,
            repo_available=False,
            platform="",
            blocked_by=blocked_by,
            detail="probe failed",
        )

    stdout = completed.stdout
    reachable = completed.returncode == 0
    python_available = "PYTHON_OK" in stdout
    uv_available = "UV_OK" in stdout
    gpu_available = "GPU_OK" in stdout
    cache_available = "CACHE_OK" in stdout
    repo_available = "REPO_OK" in stdout
    platform = ""
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped in {"Linux", "Darwin"}:
            platform = stripped
        elif stripped and not stripped.endswith(("_OK", "_MISSING")) and stripped not in {"PLATFORM_UNKNOWN"}:
            if stripped in {"Linux", "Darwin"} or stripped.startswith(("MINGW", "MSYS", "CYGWIN")):
                platform = stripped

    if not reachable:
        blocked_by.append(
            completed.stderr.strip() or f"SSH probe exited with code {completed.returncode}."
        )
    if not python_available:
        blocked_by.append("Remote worker is missing Python.")
    if not uv_available:
        blocked_by.append("Remote worker is missing uv.")
    if not gpu_available:
        blocked_by.append("Remote worker has no NVIDIA runtime.")
    if not cache_available:
        blocked_by.append("Remote worker is missing ~/.cache/autoresearch.")
    if not repo_available:
        blocked_by.append(
            f"Remote worker is missing repo path {config.autoresearch_remote_repo}."
        )

    return RemoteAutoresearchWorker(
        enabled=True,
        host=config.autoresearch_remote_host,
        repo_path=config.autoresearch_remote_repo,
        ssh_path=ssh_path,
        reachable=reachable,
        python_available=python_available,
        uv_available=uv_available,
        gpu_available=gpu_available,
        cache_available=cache_available,
        repo_available=repo_available,
        platform=platform,
        blocked_by=blocked_by,
        detail="remote autoresearch worker",
    )


def doctor_report(
    config: RuntimeConfig,
    *,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    config.ensure_runtime_dirs()

    uv_path = shutil.which("uv") or ""
    git_path = shutil.which("git") or ""
    nvidia_path = shutil.which("nvidia-smi") or ""
    autoresearch_launcher = config.launcher_for_project(config.autoresearch_repo)
    denario_launcher = config.launcher_for_project(config.denario_repo)
    default_data_description = config.default_data_description_path()
    autoresearch_cache_dir = Path.home() / ".cache" / "autoresearch"
    remote_autoresearch = probe_remote_autoresearch_worker(config)

    key_presence = {
        "OPENAI_API_KEY": bool(config.env.get("OPENAI_API_KEY")),
        "GOOGLE_API_KEY": bool(config.env.get("GOOGLE_API_KEY")),
        "ANTHROPIC_API_KEY": bool(config.env.get("ANTHROPIC_API_KEY")),
        "PERPLEXITY_API_KEY": bool(config.env.get("PERPLEXITY_API_KEY")),
        "FUTURE_HOUSE_API_KEY": bool(config.env.get("FUTURE_HOUSE_API_KEY")),
        "GOOGLE_APPLICATION_CREDENTIALS": bool(config.env.get("GOOGLE_APPLICATION_CREDENTIALS")),
    }

    method_bridge_provider = provider_for_model(config.method_bridge_model)
    method_bridge_blockers: list[str] = []
    if not config.method_bridge_enabled:
        method_bridge_blockers.append("Method bridge is disabled.")
    if method_bridge_provider != "openai":
        method_bridge_blockers.append(
            f"Only OpenAI-backed method bridging is implemented; got {config.method_bridge_model}."
        )
    if not key_presence["OPENAI_API_KEY"]:
        method_bridge_blockers.append(
            f"Missing OPENAI_API_KEY for bridge model {config.method_bridge_model}."
        )

    def denario_capability(model_name: str, *, action: str) -> dict[str, Any]:
        blocked_by: list[str] = []
        provider = provider_for_model(model_name)
        env_key = provider_env_key(provider)
        if not denario_launcher.available:
            blocked_by.append(denario_launcher.detail)
        if env_key and not key_presence.get(env_key, False):
            blocked_by.append(f"Missing {env_key} for configured model {model_name}.")
        if provider == "unknown":
            blocked_by.append(f"Unknown provider for configured model {model_name}.")
        return {
            "action": action,
            "executor": "denario",
            "available": not blocked_by,
            "blocked_by": blocked_by,
            "model": model_name,
            "provider": provider,
        }

    autoresearch_blockers: list[str] = []
    autoresearch_mode = "remote" if remote_autoresearch is not None else "local"
    if remote_autoresearch is not None:
        autoresearch_blockers.extend(remote_autoresearch.blocked_by)
    else:
        if not autoresearch_launcher.available:
            autoresearch_blockers.append(autoresearch_launcher.detail)
        if not nvidia_path:
            autoresearch_blockers.append("No NVIDIA runtime detected (`nvidia-smi` not found).")
        if not autoresearch_cache_dir.exists():
            autoresearch_blockers.append(
                f"Missing autoresearch cache at {autoresearch_cache_dir}. Run prepare.py first."
            )

    capabilities = {
        "generate_idea": {
            **denario_capability(config.denario_idea_llm, action="generate_idea"),
            "notes": [
                "Requires a data description at run time via AIEQ_DATA_DESCRIPTION_FILE, "
                "`--data-description-file`, or `--data-description`."
            ],
        },
        "generate_method": denario_capability(config.denario_method_llm, action="generate_method"),
        "synthesize_paper": denario_capability(
            config.denario_paper_llm,
            action="synthesize_paper",
        ),
        "run_experiment": {
            "action": "run_experiment",
            "executor": "autoresearch",
            "available": not autoresearch_blockers,
            "blocked_by": autoresearch_blockers,
            "branch": config.default_autoresearch_branch,
            "mode": autoresearch_mode,
            "host": remote_autoresearch.host if remote_autoresearch is not None else "",
            "bridge": {
                "enabled": config.method_bridge_enabled,
                "available": not method_bridge_blockers,
                "blocked_by": method_bridge_blockers,
                "model": config.method_bridge_model,
                "provider": method_bridge_provider,
            },
        },
        "manual_only": {
            "action": "manual_only",
            "executor": "manual",
            "available": False,
            "blocked_by": [
                "challenge_assumption, triage_attack, and collect_counterevidence remain manual "
                "or import-driven in this first execution-plane pass."
            ],
        },
    }

    payload: dict[str, Any] = {
        "repo_root": str(config.repo_root),
        "env_file": {
            "path": str(config.env_file) if config.env_file else "",
            "loaded": bool(config.env_file and config.env_file.exists()),
        },
        "runtime": {
            "runtime_dir": str(config.runtime_dir),
            "denario_projects_dir": str(config.denario_projects_dir),
            "autoresearch_output_dir": str(config.autoresearch_output_dir),
            "default_data_description_file": (
                str(default_data_description) if default_data_description else ""
            ),
            "denario_mode": config.denario_mode,
            "denario_idea_llm": config.denario_idea_llm,
            "denario_method_llm": config.denario_method_llm,
            "denario_paper_llm": config.denario_paper_llm,
            "denario_paper_journal": config.denario_paper_journal,
            "default_autoresearch_branch": config.default_autoresearch_branch,
            "autoresearch_remote_host": config.autoresearch_remote_host,
            "autoresearch_remote_repo": config.autoresearch_remote_repo,
        },
        "tools": {
            "uv": {"available": bool(uv_path), "path": uv_path},
            "git": {"available": bool(git_path), "path": git_path},
            "nvidia_smi": {"available": bool(nvidia_path), "path": nvidia_path},
        },
        "repos": {
            "autoresearch": {
                "path": str(config.autoresearch_repo),
                "exists": config.autoresearch_repo.exists(),
                "train_py": str(config.autoresearch_repo / "train.py"),
                "launcher": serialize_dataclass(autoresearch_launcher),
                "data_cache_exists": autoresearch_cache_dir.exists(),
                "mode": autoresearch_mode,
                "remote_worker": (
                    serialize_dataclass(remote_autoresearch)
                    if remote_autoresearch is not None
                    else None
                ),
            },
            "denario": {
                "path": str(config.denario_repo),
                "exists": config.denario_repo.exists(),
                "launcher": serialize_dataclass(denario_launcher),
                "keys": key_presence,
            },
        },
        "capabilities": capabilities,
    }

    if ledger_path is not None:
        from .controller import ResearchController
        from .ledger import EpistemicLedger

        ledger = EpistemicLedger.load(ledger_path)
        decision = ResearchController().decide(ledger)
        capability_name = capability_key_for_action(decision.primary_action.action_type)
        capability = capabilities.get(capability_name, capabilities["manual_only"])
        payload["next_action"] = {
            "decision": serialize_dataclass(decision),
            "capability": capability_name,
            "ready": bool(capability.get("available")),
            "blocked_by": list(capability.get("blocked_by", [])),
        }

    return payload
