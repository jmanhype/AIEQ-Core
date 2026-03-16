from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aieq_core.method_bridge import OpenAIMethodBridge


class MethodBridgeTests(unittest.TestCase):
    def test_generate_retries_once_after_invalid_python(self) -> None:
        class FakeBridge(OpenAIMethodBridge):
            def __init__(self) -> None:
                super().__init__(api_key="test-key", model="gpt-4.1")
                self.prompts: list[str] = []
                self.calls = 0

            def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
                self.calls += 1
                self.prompts.append(payload["input"][1]["content"][0]["text"])  # type: ignore[index]
                if self.calls == 1:
                    return {
                        "id": "resp_bad",
                        "output": [
                            {
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": (
                                            '{"summary":"bad","train_py":"def main():\\n'
                                            '    value = 1\\n'
                                            '    global value\\n'
                                            '    print(value)\\n"}'
                                        ),
                                    }
                                ]
                            }
                        ],
                    }
                return {
                    "id": "resp_good",
                    "output": [
                        {
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": (
                                        '{"summary":"fixed","train_py":"BRIDGED = True\\n'
                                        'DEPTH = 8\\n'
                                        'DEVICE_BATCH_SIZE = 32\\n'
                                        'EVAL_BATCH_SIZE = 8\\n\\n'
                                        'def main():\\n'
                                        '    print(\\"val_bpb:          1.234000\\")\\n'
                                        '    print(\\"training_seconds: 1.0\\")\\n'
                                        '    print(\\"total_seconds:    1.0\\")\\n'
                                        '    print(\\"peak_vram_mb:     1.0\\")\\n'
                                        '    print(\\"mfu_percent:      1.0\\")\\n'
                                        '    print(\\"total_tokens_M:   1.0\\")\\n'
                                        '    print(\\"num_steps:        1\\")\\n'
                                        '    print(\\"num_params_M:     1.0\\")\\n'
                                        '    print(\\"depth:            8\\")\\n\\n'
                                        'if __name__ == \\"__main__\\":\\n'
                                        '    main()\\n"}'
                                    ),
                                }
                            ]
                        }
                    ],
                }

        bridge = FakeBridge()
        draft = bridge.generate(
            claim_title="Retry bridge",
            claim_statement="Fix the broken bridge output.",
            method_text="Return valid python.",
            current_train_py="def main():\n    print('hello')\n",
        )

        self.assertEqual(bridge.calls, 2)
        self.assertEqual(draft.response_id, "resp_good")
        self.assertIn("Validation error:", bridge.prompts[1])
        self.assertIn("BRIDGED = True", draft.train_py)

    def test_generate_repairs_missing_autoresearch_contract(self) -> None:
        class FakeBridge(OpenAIMethodBridge):
            def __init__(self) -> None:
                super().__init__(api_key="test-key", model="gpt-4.1")
                self.prompts: list[str] = []
                self.calls = 0

            def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
                self.calls += 1
                self.prompts.append(payload["input"][1]["content"][0]["text"])  # type: ignore[index]
                if self.calls == 1:
                    return {
                        "id": "resp_missing_contract",
                        "output": [
                            {
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": (
                                            '{"summary":"missing summary labels","train_py":"DEPTH = 8\\n'
                                            'DEVICE_BATCH_SIZE = 32\\n'
                                            'EVAL_BATCH_SIZE = 8\\n'
                                            'print(\\"hello\\")\\n"}'
                                        ),
                                    }
                                ]
                            }
                        ],
                    }
                return {
                    "id": "resp_fixed_contract",
                    "output": [
                        {
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": (
                                        '{"summary":"fixed contract","train_py":"DEPTH = 8\\n'
                                        'DEVICE_BATCH_SIZE = 32\\n'
                                        'EVAL_BATCH_SIZE = 8\\n'
                                        'print(\\"val_bpb:          1.234000\\")\\n'
                                        'print(\\"training_seconds: 1.0\\")\\n'
                                        'print(\\"total_seconds:    1.0\\")\\n'
                                        'print(\\"peak_vram_mb:     1.0\\")\\n'
                                        'print(\\"mfu_percent:      1.0\\")\\n'
                                        'print(\\"total_tokens_M:   1.0\\")\\n'
                                        'print(\\"num_steps:        1\\")\\n'
                                        'print(\\"num_params_M:     1.0\\")\\n'
                                        'print(\\"depth:            8\\")\\n"}'
                                    ),
                                }
                            ]
                        }
                    ],
                }

        bridge = FakeBridge()
        draft = bridge.generate(
            claim_title="Repair contract",
            claim_statement="Preserve autoresearch summary labels.",
            method_text="Keep the run-log contract intact.",
            current_train_py="print('hello')\n",
        )

        self.assertEqual(bridge.calls, 2)
        self.assertEqual(draft.response_id, "resp_fixed_contract")
        self.assertIn("autoresearch summary labels", bridge.prompts[1])
        self.assertIn("val_bpb:", draft.train_py)


if __name__ == "__main__":
    unittest.main()
