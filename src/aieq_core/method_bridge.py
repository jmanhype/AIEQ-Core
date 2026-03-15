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
        payload = {
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
        response = self._post_json("/responses", payload)
        output_text = self._extract_output_text(response)
        parsed = self._parse_output_json(output_text)
        train_py = parsed["train_py"].replace("\r\n", "\n")
        self._validate_python(train_py)
        return MethodBridgeDraft(
            model=self.model,
            prompt=prompt,
            summary=parsed["summary"].strip(),
            train_py=train_py,
            response_id=str(response.get("id", "")).strip(),
            usage=response.get("usage", {}) if isinstance(response.get("usage"), dict) else {},
            raw_response=response,
        )

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
    def _validate_python(source: str) -> None:
        try:
            compile(source, "train.py", "exec")
        except SyntaxError as exc:
            raise MethodBridgeError(f"Generated train.py is not valid Python: {exc}") from exc
