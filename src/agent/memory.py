from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.tools.rag_retrieve import RetrievedHit


@dataclass
class AgentMemory:
    turn_count: int = 0
    last_question: str = ""
    last_answer: str = ""

    last_retrieval_query: str = ""
    last_retrieval_text: str = ""
    last_references: list[RetrievedHit] = field(default_factory=list)

    last_calc_expression: str = ""
    last_calc_value: float | None = None

    variables: dict[str, float] = field(default_factory=dict)
    tool_observations: dict[str, str] = field(default_factory=dict)

    last_reranker_applied: bool = False
    last_reranker_message: str = ""

    def reset(self) -> None:
        self.turn_count = 0
        self.last_question = ""
        self.last_answer = ""
        self.last_retrieval_query = ""
        self.last_retrieval_text = ""
        self.last_references = []
        self.last_calc_expression = ""
        self.last_calc_value = None
        self.variables = {}
        self.tool_observations = {}
        self.last_reranker_applied = False
        self.last_reranker_message = ""

    def apply_delta(self, delta: dict[str, Any]) -> None:
        for key, value in delta.items():
            if key == "variables" and isinstance(value, dict):
                casted: dict[str, float] = {}
                for k, v in value.items():
                    try:
                        casted[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
                self.variables.update(casted)
                continue

            if key == "tool_observations" and isinstance(value, dict):
                for k, v in value.items():
                    self.tool_observations[str(k)] = str(v)
                continue

            if hasattr(self, key):
                setattr(self, key, value)

    def summarize(self) -> str:
        vars_text = ", ".join(f"{k}={v}" for k, v in sorted(self.variables.items()))
        if not vars_text:
            vars_text = "<none>"

        calc_text = (
            f"{self.last_calc_expression} = {self.last_calc_value}"
            if self.last_calc_expression and self.last_calc_value is not None
            else "<none>"
        )

        refs_text = "<none>"
        if self.last_references:
            refs_text = "; ".join(f"{r.source}#p{r.page}" for r in self.last_references[:3])

        return (
            f"turn_count={self.turn_count}; "
            f"last_calc={calc_text}; "
            f"variables={vars_text}; "
            f"last_refs={refs_text}"
        )
