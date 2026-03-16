from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
import urllib.error
import urllib.request


BRIDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {
            "type": "string",
            "description": "Short description of the train.py mutations that were made.",
        },
        "train_py": {
            "type": "string",
            "description": "Full replacement contents for train.py.",
        },
    },
    "required": ["summary", "train_py"],
}

REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "approved": {
            "type": "boolean",
            "description": "Whether the generated train.py is safe enough to execute.",
        },
        "summary": {
            "type": "string",
            "description": "Short explanation of the review decision.",
        },
        "blockers": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete execution blockers that should prevent launch.",
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Non-blocking concerns worth tracking.",
        },
    },
    "required": ["approved", "summary", "blockers", "warnings"],
}

AUTORESEARCH_REQUIRED_SUMMARY_LABELS = (
    "val_bpb:",
    "training_seconds:",
    "total_seconds:",
    "peak_vram_mb:",
    "mfu_percent:",
    "total_tokens_M:",
    "num_steps:",
    "num_params_M:",
    "depth:",
)

AUTORESEARCH_REQUIRED_ASSIGNMENTS = (
    "DEPTH",
    "DEVICE_BATCH_SIZE",
    "EVAL_BATCH_SIZE",
)

SYSTEM_PROMPT = """You rewrite a single Python file named train.py for autoresearch.

Your job is to transform the existing train.py so it implements the supplied research method
while preserving the file's working execution flow, logging shape, and overall script integrity.

Rules:
- Return a full replacement train.py, not a patch.
- Keep the file runnable as a standalone Python script.
- Do not introduce placeholder comments, TODOs, ellipses, or pseudo-code.
- Do not add any new third-party dependencies.
- Preserve the existing training summary/logging behavior unless the method absolutely requires a small adjustment.
- Keep changes tightly scoped to what is needed for the method.
- Preserve already-working memory-safety adaptations unless the method explicitly requires otherwise.
- Prefer clear, local code changes over broad rewrites.
"""

REVIEW_SYSTEM_PROMPT = """You review a generated train.py rewrite for autoresearch before execution.

Your job is to prevent unsafe or obviously broken drafts from reaching the GPU worker.

Treat any of the following as blockers:
- likely runtime errors from missing or undefined names
- accidental shadowing or API misuse that breaks execution flow
- missing autoresearch result-summary labels needed by downstream parsing
- removal of critical configuration assignments without an equivalent replacement
- broad rewrites that no longer preserve the existing training script contract

Prefer concrete blockers over vague style feedback.
"""


class MethodBridgeError(RuntimeError):
    """Raised when the method-to-code bridge cannot produce a valid train.py."""


@dataclass(slots=True)
class MethodBridgeDraft:
    model: str
    prompt: str
    summary: str
    train_py: str
    response_id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "response_id": self.response_id,
            "summary": self.summary,
            "usage": self.usage,
        }


@dataclass(slots=True)
class MethodBridgeReview:
    model: str
    prompt: str
    approved: bool
    summary: str
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    response_id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "response_id": self.response_id,
            "approved": self.approved,
            "summary": self.summary,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "usage": self.usage,
        }


def build_method_bridge_prompt(
    *,
    claim_title: str,
    claim_statement: str,
    method_text: str,
    current_train_py: str,
) -> str:
    return f"""Claim title:
{claim_title}

Claim statement:
{claim_statement}

Denario method:
{method_text}

Current train.py:
```python
{current_train_py}
```

Produce the full replacement train.py implementing the method above.
Keep the script operational, preserve the current execution pattern, and avoid unnecessary edits.
"""


def build_method_bridge_repair_prompt(
    *,
    claim_title: str,
    claim_statement: str,
    method_text: str,
    invalid_train_py: str,
    validation_error: str,
) -> str:
    return f"""Your previous train.py draft was rejected by the validator.

Claim title:
{claim_title}

Claim statement:
{claim_statement}

Denario method:
{method_text}

Validation error:
{validation_error}

Invalid generated train.py:
```python
{invalid_train_py}
```

Return a corrected full replacement train.py that preserves the intended method change
while fixing the validator error above.
"""


def build_method_bridge_runtime_repair_prompt(
    *,
    claim_title: str,
    claim_statement: str,
    method_text: str,
    previous_train_py: str,
    runtime_error: str,
    previous_summary: str,
) -> str:
    return f"""Your previous train.py draft ran but crashed at runtime.

Claim title:
{claim_title}

Claim statement:
{claim_statement}

Denario method:
{method_text}

Previous bridge summary:
{previous_summary}

Runtime failure log excerpt:
{runtime_error}

Failed train.py:
```python
{previous_train_py}
```

Return a corrected full replacement train.py that preserves the intended method change
while fixing the concrete runtime failure above.
"""


def build_method_bridge_review_prompt(
    *,
    claim_title: str,
    claim_statement: str,
    method_text: str,
    current_train_py: str,
    generated_train_py: str,
    generated_summary: str,
) -> str:
    required_labels = ", ".join(AUTORESEARCH_REQUIRED_SUMMARY_LABELS)
    required_assignments = ", ".join(AUTORESEARCH_REQUIRED_ASSIGNMENTS)
    return f"""Claim title:
{claim_title}

Claim statement:
{claim_statement}

Denario method:
{method_text}

Generated bridge summary:
{generated_summary}

Critical autoresearch contract:
- Preserve the training-result summary labels: {required_labels}
- Preserve or equivalently replace these configuration assignments: {required_assignments}
- Keep the script runnable as a standalone training entrypoint
- Preserve the already-working short-budget / memory-safety behavior unless the method explicitly requires a change

Current train.py:
```python
{current_train_py}
```

Generated train.py candidate:
```python
{generated_train_py}
```

Return whether this candidate should be executed. Only approve it if the draft looks operationally safe.
"""


class OpenAIMethodBridge:
    """Generates a full replacement train.py from a Denario method artifact."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: int = 120,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            raise MethodBridgeError("OpenAI API key is required for method bridging.")
        if not self.model:
            raise MethodBridgeError("Bridge model is required for method bridging.")

    def generate(
        self,
        *,
        claim_title: str,
        claim_statement: str,
        method_text: str,
        current_train_py: str,
    ) -> MethodBridgeDraft:
        prompt = build_method_bridge_prompt(
            claim_title=claim_title,
            claim_statement=claim_statement,
            method_text=method_text,
            current_train_py=current_train_py,
        )
        return self._generate_from_prompt(
            prompt=prompt,
            claim_title=claim_title,
            claim_statement=claim_statement,
            method_text=method_text,
        )

    def repair_runtime_failure(
        self,
        *,
        claim_title: str,
        claim_statement: str,
        method_text: str,
        previous_train_py: str,
        runtime_error: str,
        previous_summary: str,
    ) -> MethodBridgeDraft:
        prompt = build_method_bridge_runtime_repair_prompt(
            claim_title=claim_title,
            claim_statement=claim_statement,
            method_text=method_text,
            previous_train_py=previous_train_py,
            runtime_error=runtime_error,
            previous_summary=previous_summary,
        )
        return self._generate_from_prompt(
            prompt=prompt,
            claim_title=claim_title,
            claim_statement=claim_statement,
            method_text=method_text,
        )

    def review(
        self,
        *,
        claim_title: str,
        claim_statement: str,
        method_text: str,
        current_train_py: str,
        generated_train_py: str,
        generated_summary: str,
    ) -> MethodBridgeReview:
        prompt = build_method_bridge_review_prompt(
            claim_title=claim_title,
            claim_statement=claim_statement,
            method_text=method_text,
            current_train_py=current_train_py,
            generated_train_py=generated_train_py,
            generated_summary=generated_summary,
        )
        response = self._post_json("/responses", self._review_payload_for_prompt(prompt))
        output_text = self._extract_output_text(response)
        parsed = self._parse_review_json(output_text)
        return MethodBridgeReview(
            model=self.model,
            prompt=prompt,
            approved=parsed["approved"],
            summary=parsed["summary"].strip(),
            blockers=parsed["blockers"],
            warnings=parsed["warnings"],
            response_id=str(response.get("id", "")).strip(),
            usage=response.get("usage", {}) if isinstance(response.get("usage"), dict) else {},
            raw_response=response,
        )

    def _generate_from_prompt(
        self,
        *,
        prompt: str,
        claim_title: str,
        claim_statement: str,
        method_text: str,
    ) -> MethodBridgeDraft:
        last_error = ""
        current_prompt = prompt

        for attempt in range(2):
            response = self._post_json("/responses", self._payload_for_prompt(current_prompt))
            output_text = self._extract_output_text(response)
            parsed = self._parse_output_json(output_text)
            train_py = parsed["train_py"].replace("\r\n", "\n")

            try:
                self._validate_python(train_py)
            except MethodBridgeError as exc:
                last_error = str(exc)
                if attempt == 1:
                    raise
                current_prompt = build_method_bridge_repair_prompt(
                    claim_title=claim_title,
                    claim_statement=claim_statement,
                    method_text=method_text,
                    invalid_train_py=train_py,
                    validation_error=last_error,
                )
                continue

            return MethodBridgeDraft(
                model=self.model,
                prompt=current_prompt,
                summary=parsed["summary"].strip(),
                train_py=train_py,
                response_id=str(response.get("id", "")).strip(),
                usage=response.get("usage", {}) if isinstance(response.get("usage"), dict) else {},
                raw_response=response,
            )

        raise MethodBridgeError(last_error or "Method bridge failed to produce valid Python.")

    def _payload_for_prompt(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "train_py_bridge",
                    "schema": BRIDGE_SCHEMA,
                    "strict": True,
                }
            },
            "max_output_tokens": 12000,
        }

    def _review_payload_for_prompt(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": REVIEW_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "train_py_bridge_review",
                    "schema": REVIEW_SCHEMA,
                    "strict": True,
                }
            },
            "max_output_tokens": 4000,
        }

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise MethodBridgeError(f"OpenAI bridge request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise MethodBridgeError(f"OpenAI bridge request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise MethodBridgeError("OpenAI bridge returned invalid JSON.") from exc

    @staticmethod
    def _extract_output_text(response: dict[str, Any]) -> str:
        top_level = response.get("output_text")
        if isinstance(top_level, str) and top_level.strip():
            return top_level

        for item in response.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                direct = content.get("text")
                if isinstance(direct, str) and direct.strip():
                    return direct
                nested = content.get("text")
                if isinstance(nested, dict):
                    nested_value = nested.get("value")
                    if isinstance(nested_value, str) and nested_value.strip():
                        return nested_value
                output_text = content.get("output_text")
                if isinstance(output_text, str) and output_text.strip():
                    return output_text

        raise MethodBridgeError("OpenAI bridge response did not contain output text.")

    @staticmethod
    def _parse_output_json(output_text: str) -> dict[str, str]:
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise MethodBridgeError("OpenAI bridge output was not valid JSON.") from exc

        summary = parsed.get("summary")
        train_py = parsed.get("train_py")
        if not isinstance(summary, str) or not summary.strip():
            raise MethodBridgeError("OpenAI bridge output is missing `summary`.")
        if not isinstance(train_py, str) or not train_py.strip():
            raise MethodBridgeError("OpenAI bridge output is missing `train_py`.")
        return {"summary": summary, "train_py": train_py}

    @staticmethod
    def _parse_review_json(output_text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise MethodBridgeError("OpenAI bridge review output was not valid JSON.") from exc

        approved = parsed.get("approved")
        summary = parsed.get("summary")
        blockers = parsed.get("blockers")
        warnings = parsed.get("warnings")

        if not isinstance(approved, bool):
            raise MethodBridgeError("OpenAI bridge review output is missing `approved`.")
        if not isinstance(summary, str) or not summary.strip():
            raise MethodBridgeError("OpenAI bridge review output is missing `summary`.")
        if not isinstance(blockers, list) or not all(isinstance(item, str) for item in blockers):
            raise MethodBridgeError("OpenAI bridge review output is missing `blockers`.")
        if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
            raise MethodBridgeError("OpenAI bridge review output is missing `warnings`.")
        return {
            "approved": approved,
            "summary": summary,
            "blockers": [item.strip() for item in blockers if item.strip()],
            "warnings": [item.strip() for item in warnings if item.strip()],
        }

    @staticmethod
    def _validate_python(source: str) -> None:
        try:
            compile(source, "train.py", "exec")
        except SyntaxError as exc:
            raise MethodBridgeError(f"Generated train.py is not valid Python: {exc}") from exc
        OpenAIMethodBridge._validate_autoresearch_contract(source)

    @staticmethod
    def _validate_autoresearch_contract(source: str) -> None:
        missing_labels = [
            label for label in AUTORESEARCH_REQUIRED_SUMMARY_LABELS if label not in source
        ]
        if missing_labels:
            raise MethodBridgeError(
                "Generated train.py is missing autoresearch summary labels: "
                + ", ".join(missing_labels)
            )

        missing_assignments = []
        for name in AUTORESEARCH_REQUIRED_ASSIGNMENTS:
            if f"{name} =" not in source and f"{name}=" not in source:
                missing_assignments.append(name)
        if missing_assignments:
            raise MethodBridgeError(
                "Generated train.py is missing required autoresearch assignments: "
                + ", ".join(missing_assignments)
            )
