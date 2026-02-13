from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.agent.memory import AgentMemory
from src.llm.client import OpenAIClientBundle
from src.llm.prompts import (
    AGENT_PLANNER_SYSTEM_PROMPT,
    AGENT_ROUTER_SYSTEM_PROMPT,
    build_agent_plan_prompt,
    build_agent_router_prompt,
)


@dataclass(frozen=True)
class PlannedStep:
    tool: str
    input: str
    reason: str


class AgentPlanner:
    def __init__(
        self,
        llm_clients: OpenAIClientBundle,
        max_steps: int = 8,
        recent_history_window: int = 20,
    ) -> None:
        self.llm_clients = llm_clients
        self.max_steps = max(1, max_steps)
        self.recent_history_window = max(1, recent_history_window)

    def plan(
        self,
        question: str,
        memory: AgentMemory | None = None,
        history: list[dict[str, str]] | None = None,
        route: str | None = None,
    ) -> list[PlannedStep]:
        history = history or []
        route = route or self._route_question(question)
        if route == "闲聊":
            return []
        heuristic_steps = self._heuristic_plan(question=question, memory=memory)
        if heuristic_steps:
            return heuristic_steps[: self.max_steps]
        if route == "其他":
            return []
        if route is None and self._should_skip_tools(question):
            return []

        prompt = build_agent_plan_prompt(
            question=question,
            max_steps=self.max_steps,
            memory_summary=memory.summarize() if memory is not None else "<none>",
            recent_history=history[-self.recent_history_window :],
        )
        try:
            raw = self.llm_clients.chat(
                messages=[
                    {"role": "system", "content": AGENT_PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            parsed = self._parse_steps(raw, memory=memory)
            if parsed:
                return parsed[: self.max_steps]
        except Exception:
            pass

        return [PlannedStep(tool="retrieve", input=question, reason="fallback retrieve")]

    @staticmethod
    def _normalize_question(question: str) -> str:
        return " ".join(question.strip().split())

    def _should_skip_tools(self, question: str) -> bool:
        q = self._normalize_question(question)
        if not q:
            return True
        if self._has_doc_hints(q):
            return False
        if self._is_smalltalk(q):
            return True
        token_count = len(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", q))
        return token_count <= 8

    def _route_question(self, question: str) -> str | None:
        q = self._normalize_question(question)
        if not q:
            return None
        prompt = build_agent_router_prompt(q)
        try:
            raw = self.llm_clients.chat(
                messages=[
                    {"role": "system", "content": AGENT_ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
        except Exception:
            return None
        return self._parse_route(raw)

    @staticmethod
    def _parse_route(text: str) -> str | None:
        if not text:
            return None
        for label in ("需要查询知识库", "闲聊", "其他"):
            if label in text:
                return label
        return None

    def route_question(self, question: str) -> str | None:
        return self._route_question(question)

    @staticmethod
    def _is_smalltalk(question: str) -> bool:
        normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", question.lower())
        if not normalized:
            return True

        simple = {
            "hi",
            "hello",
            "hey",
            "sup",
            "yo",
            "hola",
            "thanks",
            "thx",
            "你好",
            "您好",
            "嗨",
            "哈喽",
            "哈囉",
            "在吗",
            "在么",
            "在嘛",
            "早上好",
            "下午好",
            "晚上好",
            "谢谢",
            "多谢",
            "感谢",
            "再见",
            "拜拜",
        }
        if normalized in simple:
            return True

        if re.fullmatch(r"(你是谁|你叫什么|你是做什么的|你能做什么|你会什么)", normalized):
            return True

        return False

    @staticmethod
    def _has_doc_hints(question: str) -> bool:
        if AgentPlanner._extract_symbolic_expression(question):
            return True

        if re.search(r"\b[A-Z_][A-Z0-9_]{2,}\b", question):
            return True

        if re.search(r"\b[A-Z0-9]{2,}(?:[-_][A-Z0-9]{2,}){1,}\b", question):
            return True

        if re.search(r"(page\s*\d+|p\.?\s*\d+|第?\s*\d+\s*页)", question, flags=re.IGNORECASE):
            return True

        keywords = (
            "文档",
            "文件",
            "报告",
            "pdf",
            "表",
            "图",
            "章节",
            "附录",
            "引用",
            "来源",
            "根据",
            "检索",
            "查找",
            "搜索",
            "资料",
            "数据",
            "指标",
            "年报",
            "公告",
            "财报",
            "document",
            "report",
            "reference",
            "cite",
        )
        return any(key in question for key in keywords)

    def _heuristic_plan(self, question: str, memory: AgentMemory | None) -> list[PlannedStep]:
        if self._is_budget_analysis_request(question):
            return [
                PlannedStep(tool="retrieve", input=question, reason="collect annual budget data"),
                PlannedStep(tool="budget_analyst", input="用户问题", reason="analyze budget-based rating"),
            ]

        expr = self._extract_symbolic_expression(question)
        if expr:
            vars_in_expr = self._extract_variable_tokens(expr)
            if memory and vars_in_expr and all(v in memory.variables for v in vars_in_expr):
                return [
                    PlannedStep(tool="calculate", input=expr, reason="reuse variables from memory"),
                ]

            return [
                PlannedStep(tool="retrieve", input=question, reason="collect variable values from docs"),
                PlannedStep(tool="calculate", input=expr, reason="evaluate requested expression"),
            ]

        followup_expr = self._extract_followup_expression(question=question, memory=memory)
        if followup_expr:
            return [
                PlannedStep(tool="calculate", input=followup_expr, reason="reuse LAST_RESULT from memory"),
            ]

        return []

    @staticmethod
    def _is_budget_analysis_request(question: str) -> bool:
        q = question.lower()
        has_budget = bool(re.search(r"(年度?预算|budget)", q))
        has_rating = bool(re.search(r"(股价|price|评级|分析师|买入|卖出|增持|减持)", q))
        return has_budget and has_rating

    @staticmethod
    def _extract_variable_tokens(expression: str) -> list[str]:
        names = re.findall(r"\b([A-Z_][A-Z0-9_]*)\b", expression)
        unique: list[str] = []
        seen: set[str] = set()
        for item in names:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

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

    @staticmethod
    def _extract_followup_expression(question: str, memory: AgentMemory | None) -> str | None:
        if memory is None or memory.last_calc_value is None:
            return None

        q = " ".join(question.strip().split())
        has_followup_hint = bool(
            re.search(r"(刚才|上次|上一步|之前|那个结果|这个结果|上个结果|再)" , q)
        )
        if not has_followup_hint:
            return None

        patterns: list[tuple[str, str]] = [
            (r"(?:再|然后)?加(?:上)?\s*(-?\d+(?:\.\d+)?)", "+"),
            (r"(?:再|然后)?减(?:去)?\s*(-?\d+(?:\.\d+)?)", "-"),
            (r"(?:再|然后)?乘(?:以|上)?\s*(-?\d+(?:\.\d+)?)", "*"),
            (r"(?:再|然后)?除(?:以)?\s*(-?\d+(?:\.\d+)?)", "/"),
        ]

        for pattern, op in patterns:
            m = re.search(pattern, q)
            if not m:
                continue
            number = m.group(1)
            return f"LAST_RESULT {op} {number}"

        return None

    def _parse_steps(self, raw: str, memory: AgentMemory | None) -> list[PlannedStep]:
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
            if tool not in {"retrieve", "calculate", "budget_analyst", "finish"}:
                continue
            text = str(item.get("input", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if tool != "finish" and not text:
                if tool in {"retrieve", "budget_analyst"}:
                    text = "用户问题"
                else:
                    continue
            out.append(PlannedStep(tool=tool, input=text, reason=reason or ""))

        if not out:
            return []

        need_grounding = not any(step.tool == "retrieve" for step in out)
        has_memory_context = bool(memory and (memory.last_references or memory.variables or memory.last_calc_value is not None))
        if need_grounding and not has_memory_context:
            out.insert(0, PlannedStep(tool="retrieve", input="用户问题", reason="force grounding"))

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
