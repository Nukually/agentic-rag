from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.agent.memory import AgentMemory
from src.agent.tools.rag_retrieve import RetrievedHit
from src.llm.client import OpenAIClientBundle
from src.retrieval.reranker import OpenAIStyleReranker
from src.retrieval.vector_store import MilvusVectorStore
from src.retrieval.keyword_index import KeywordIndex


@dataclass
class ToolContext:
    question: str
    history: list[dict[str, str]]
    memory: AgentMemory
    run_state: dict[str, Any]

    llm_clients: OpenAIClientBundle
    vector_store: MilvusVectorStore
    reranker: OpenAIStyleReranker
    keyword_index: KeywordIndex | None

    top_k: int
    candidate_k: int
    hybrid_vector_weight: float
    hybrid_keyword_weight: float


@dataclass
class ToolOutput:
    observation: str
    references: list[RetrievedHit] = field(default_factory=list)
    memory_delta: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentTool(Protocol):
    name: str

    def run(self, tool_input: str, context: ToolContext) -> ToolOutput:
        ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def run(self, name: str, tool_input: str, context: ToolContext) -> ToolOutput:
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutput(observation=f"tool_not_found: {name}")
        return tool.run(tool_input=tool_input, context=context)
