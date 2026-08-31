"""Tuner agent: proposes HIP kernel modifications.

Deliberately blind to the referee's thresholds. It receives what FAILED,
never what would PASS -- an agent that knows the tolerance can tune to sit
just inside it, which reconstructs the self-grading problem one level up.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import urllib.request
import json as _json


SYSTEM_PROMPT = """You optimize HIP kernels for AMD CDNA3 (MI300X).

You will be given a HIP source file and, if this is not the first attempt,
the reasons your previous attempts were rejected by an independent referee.

Rules:
- Return ONLY the complete modified HIP source file, in a single ```cpp block.
- Preserve the exact stdout JSON format. The referee parses it.
- Preserve the kernel's observable behaviour. It must compute the same result.
- Do not change the problem size, the input generation, or the host reference.

Useful facts about the target:
- A wavefront on CDNA3 is 64 lanes, not 32. Any code assuming warp size 32
  is wrong here.
- Softmax is memory-bound. Reducing passes over memory matters more than
  reducing arithmetic.
- LDS (shared memory) per compute unit differs from NVIDIA. Do not assume
  CUDA occupancy heuristics transfer.
"""


@dataclass
class Attempt:
    source: str
    rejection: str


class TunerAgent:

    def __init__(self, model: str = "claude-sonnet-4-6",
                 api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def propose(self, baseline_source: str,
                history: list[Attempt] | None = None) -> str:
        prompt = self._build_prompt(baseline_source, history or [])
        response = self._call_model(prompt)
        return self._extract_source(response)

    def _build_prompt(self, baseline_source: str,
                      history: list[Attempt]) -> str:
        parts = ["Here is the baseline HIP kernel:\n\n```cpp\n",
                 baseline_source, "\n```\n"]

        if history:
            parts.append("\nYour previous attempts were rejected:\n")
            for i, attempt in enumerate(history, 1):
                parts.append(f"\n--- Attempt {i} ---\n{attempt.rejection}\n")
            parts.append(
                "\nPropose a different approach. Start from the baseline above, "
                "not from your rejected attempts.\n"
            )
        else:
            parts.append(
                "\nPropose an optimized version. Explain nothing; return only "
                "the modified source.\n"
            )

        return "".join(parts)

    def _call_model(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        body = _json.dumps({
            "model": self.model,
            "max_tokens": 16000,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = _json.loads(resp.read())

        return "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        )

    @staticmethod
    def _extract_source(response: str) -> str:
        """Pull the code block. A response with no code block is a failed
        proposal, not something to salvage by guessing."""
        match = re.search(r"```(?:cpp|c\+\+|hip)?\n(.*?)```", response, re.S)
        if not match:
            raise ValueError("model response contained no code block")
        return match.group(1)