from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.llm.client import OpenAIClientBundle
from src.llm.prompts import AGENT_PLANNER_SYSTEM_PROMPT, build_agent_plan_prompt


@dataclass(frozen=True)
class PlannedStep:
    tool: str
    input: str
    reason: str


class AgentPlanner:
    def __init__(self, llm_clients: OpenAIClientBundle, max_steps: int = 4) -> None:
        self.llm_clients = llm_clients
        self.max_steps = max_steps

    def plan(self, question: str) -> list[PlannedStep]:
        heuristic_steps = self._heuristic_plan(question)
        if heuristic_steps:
            return heuristic_steps[: self.max_steps]

        prompt = build_agent_plan_prompt(question=question, max_steps=self.max_steps)
        raw = self.llm_clients.chat(
            messages=[
                {"role": "system", "content": AGENT_PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        parsed = self._parse_steps(raw)
        if parsed:
            return parsed[: self.max_steps]

        return [PlannedStep(tool="retrieve", input=question, reason="fallback retrieve")]

    def _heuristic_plan(self, question: str) -> list[PlannedStep]:
        expr = self._extract_symbolic_expression(question)
        if expr:
            return [
                PlannedStep(tool="retrieve", input=question, reason="collect variable values from docs"),
                PlannedStep(tool="calculate", input=expr, reason="evaluate requested expression"),
            ]
        return []

    @staticmethod
    def _extract_symbolic_expression(question: str) -> str | None:
        candidates = re.findall(
            r"([A-Z_][A-Z0-9_]*(?:\s+[+\-*/]\s+(?:[A-Z_][A-Z0-9_]*|\d+(?:\.\d+)?))+)",
            question,
        )
        for item in candidates:
            expr = " ".join(item.strip().split())
            if re.fullmatch(
                r"[A-Z_][A-Z0-9_]*(?:\s+[+\-*/]\s+(?:[A-Z_][A-Z0-9_]*|\d+(?:\.\d+)?))+",
                expr,
            ):
                return expr
        return None

    def _parse_steps(self, raw: str) -> list[PlannedStep]:
        payload = self._extract_json(raw)
        if payload is None:
            return []

        raw_steps = payload.get("steps", []) if isinstance(payload, dict) else []
        if not isinstance(raw_steps, list):
            return []

        out: list[PlannedStep] = []
        for item in raw_steps:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "")).strip().lower()
            if tool not in {"retrieve", "calculate", "finish"}:
                continue
            text = str(item.get("input", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if tool != "finish" and not text:
                continue
            out.append(PlannedStep(tool=tool, input=text, reason=reason or ""))

        if not out:
            return []

        # Ensure at least one retrieval for knowledge-grounded answering.
        if not any(step.tool == "retrieve" for step in out):
            out.insert(0, PlannedStep(tool="retrieve", input="", reason="force grounding"))

        normalized: list[PlannedStep] = []
        for idx, step in enumerate(out):
            if step.tool == "retrieve" and not step.input:
                normalized.append(PlannedStep(tool="retrieve", input="用户问题", reason=step.reason))
            else:
                normalized.append(step)
            if idx + 1 >= self.max_steps:
                break

        return normalized

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        text = text.strip()
        if not text:
            return None

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()

        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
