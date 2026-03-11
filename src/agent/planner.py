"""planning"""

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
    """One planned tool invocation."""

    tool: str
    input: str
    reason: str


class AgentPlanner:
    """Plan tool steps from question, memory, and conversation history.

    The planner combines LLM planning with heuristic fallbacks to improve
    robustness for common workflows such as retrieval+calculation chains.
    """

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
        replan_feedback: str | None = None,
        previous_steps: list[PlannedStep] | None = None,
        previous_observations: list[str] | None = None,
    ) -> list[PlannedStep]:
        """Return ordered tool steps for the current round.

        Args:
            question: User question for this turn.
            memory: Current conversation memory snapshot.
            history: Multi-turn dialogue history.
            route: Optional precomputed route label from router.
            replan_feedback: Feedback produced by reflection stage.
            previous_steps: Last round's plan.
            previous_observations: Last round's tool observations.

        Returns:
            list[PlannedStep]: Planned tool sequence for execution.
        """

        history = history or []
        previous_steps = previous_steps or []
        previous_observations = previous_observations or []
        route = route or self._route_question(question)

        if replan_feedback:
            reparsed = self._llm_plan(
                question=question,
                memory=memory,
                history=history,
                replan_feedback=replan_feedback,
                previous_steps=previous_steps,
                previous_observations=previous_observations,
            )
            if reparsed:
                return reparsed[: self.max_steps]

            heuristic_retry = self._heuristic_replan(question=question, memory=memory, feedback=replan_feedback)
            if heuristic_retry:
                return heuristic_retry[: self.max_steps]

            return [PlannedStep(tool="retrieve", input=question, reason="retry fallback retrieve")]

        heuristic_steps = self._heuristic_plan(question=question, memory=memory)
        if route == "闲聊" and not heuristic_steps:
            return []
        if heuristic_steps:
            return heuristic_steps[: self.max_steps]
        if route == "其他":
            return []
        if route is None and self._should_skip_tools(question):
            return []

        parsed = self._llm_plan(
            question=question,
            memory=memory,
            history=history,
            replan_feedback=None,
            previous_steps=[],
            previous_observations=[],
        )
        if parsed:
            return parsed[: self.max_steps]

        return [PlannedStep(tool="retrieve", input=question, reason="fallback retrieve")]

    def _llm_plan(
        self,
        question: str,
        memory: AgentMemory | None,
        history: list[dict[str, str]],
        replan_feedback: str | None,
        previous_steps: list[PlannedStep],
        previous_observations: list[str],
    ) -> list[PlannedStep]:
        """Generate tool steps by prompting the planning model."""

        prompt = build_agent_plan_prompt(
            question=question,
            max_steps=self.max_steps,
            memory_summary=memory.summarize() if memory is not None else "<none>",
            recent_history=history[-self.recent_history_window :],
            replan_feedback=replan_feedback,
            previous_steps=previous_steps,
            previous_observations=previous_observations,
        )
        try:
            raw = self.llm_clients.chat(
                messages=[
                    {"role": "system", "content": AGENT_PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return self._parse_steps(raw, memory=memory, question=question)
        except Exception:
            return []

    def _heuristic_replan(
        self,
        question: str,
        memory: AgentMemory | None,
        feedback: str,
    ) -> list[PlannedStep]:
        """Fallback replan strategy when model planning is unavailable."""

        lower_feedback = feedback.lower()
        expr = self._extract_symbolic_expression(question)

        if "unknown variable" in lower_feedback:
            if expr:
                return [
                    PlannedStep(tool="retrieve", input=question, reason="retry: fetch variables"),
                    PlannedStep(tool="calculate", input=expr, reason="retry: recompute expression"),
                ]
            return [PlannedStep(tool="retrieve", input=question, reason="retry: fetch missing context")]

        if "no hits" in lower_feedback or "missing references" in lower_feedback:
            return [PlannedStep(tool="retrieve", input=question, reason="retry: broaden retrieval")]

        if memory and memory.last_references and expr:
            return [PlannedStep(tool="calculate", input=expr, reason="retry: reuse memory references")]

        return []

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
        """Classify a question into chat/knowledge/general route labels."""

        q = self._normalize_question(question)
        if not q:
            return None
        if self._is_coverage_feedback_text(q):
            return "需要查询知识库"
        if self._has_doc_hints(q):
            return "需要查询知识库"
        if self._is_smalltalk(q):
            return "闲聊"
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
        normalized = text.strip()
        if not normalized:
            return None

        # Remove common wrappers from LLM outputs: quotes, markdown fences, punctuation, spaces.
        normalized = re.sub(r"^```(?:text|json)?", "", normalized, flags=re.IGNORECASE).strip()
        normalized = re.sub(r"```$", "", normalized).strip()
        normalized = re.sub(r"\s+", "", normalized)
        normalized = normalized.strip("`'\"：:。,.，；;!?！？()[]{}")

        aliases = {
            "闲聊": "闲聊",
            "聊天": "闲聊",
            "chitchat": "闲聊",
            "需要查询知识库": "需要查询知识库",
            "查询知识库": "需要查询知识库",
            "知识库": "需要查询知识库",
            "其他": "其他",
            "通用": "其他",
            "general": "其他",
        }
        if normalized in aliases:
            return aliases[normalized]

        labels = ("需要查询知识库", "闲聊", "其他")
        hits: list[tuple[int, str]] = []
        for label in labels:
            idx = normalized.find(label)
            if idx < 0:
                continue
            prefix = normalized[max(0, idx - 4) : idx]
            # Avoid false positive such as "不需要查询知识库".
            if label == "需要查询知识库" and (
                "不需要" in prefix
                or "无需" in prefix
                or "不用" in prefix
                or "不必" in prefix
                or "不是" in prefix
                or "并非" in prefix
                or prefix.endswith("不")
                or prefix.endswith("非")
            ):
                continue
            if label in {"闲聊", "其他"} and (
                prefix.endswith("不")
                or prefix.endswith("非")
                or prefix.endswith("别")
                or prefix.endswith("不是")
                or prefix.endswith("并非")
            ):
                continue
            hits.append((idx, label))

        if hits:
            hits.sort(key=lambda item: item[0])
            return hits[0][1]
        return None

    def route_question(self, question: str) -> str | None:
        """Public wrapper for question routing."""

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
        """Fast-path planning for frequent deterministic patterns.

        Example:
            - symbolic math expression -> retrieve + calculate
            - follow-up arithmetic -> calculate with LAST_RESULT
        """

        if self._is_coverage_followup(question=question, memory=memory):
            coverage_query = self._build_coverage_followup_query(question=question, memory=memory)
            return [
                PlannedStep(
                    tool="retrieve",
                    input=coverage_query,
                    reason="补全遗漏信息，扩大覆盖范围后重新检索",
                ),
            ]

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
        q = " ".join((question or "").strip().lower().split())
        if not q:
            return False

        has_budget = bool(re.search(r"(年度?预算|budget)", q))
        if not has_budget:
            return False

        has_price_target = bool(re.search(r"(股价|price|stock\s*price|股票|股市)", q))
        has_rating_intent = bool(
            re.search(r"(评级|rating|买入|卖出|增持|减持|中性|投资建议|recommend)", q)
        )
        has_explicit_budget_judgement = bool(
            re.search(
                r"((根据|基于|结合).{0,8}(预算|budget).{0,12}(分析|判断|评估))"
                r"|((分析|判断|评估).{0,8}(股价|price|股票|评级).{0,12}(预算|budget))",
                q,
            )
        )

        # Only trigger the specialized tool when the user explicitly asks for
        # budget-driven stock-price or rating judgement, not generic budget analysis.
        return (has_price_target and has_rating_intent) or (has_explicit_budget_judgement and has_price_target)

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

    @staticmethod
    def _is_coverage_followup(question: str, memory: AgentMemory | None) -> bool:
        """Detect follow-up messages indicating previous answer missed entities.

        Example:
            - 你漏掉了一些公司
            - 请补充遗漏公司并重新回答
        """

        if memory is None:
            return False
        if not (memory.last_question or memory.last_references):
            return False

        q = " ".join((question or "").strip().split()).lower()
        if not q:
            return False

        omission_hints = (
            "漏掉",
            "漏下",
            "遗漏",
            "不全",
            "缺少",
            "没覆盖",
            "没写全",
            "补充",
            "还有",
            "少了",
            "missing",
            "omitted",
            "incomplete",
        )
        target_hints = (
            "公司",
            "企业",
            "标的",
            "全部",
            "所有",
            "每家",
            "all companies",
            "all firms",
        )
        has_omission = any(token in q for token in omission_hints)
        has_target = any(token in q for token in target_hints)
        return has_omission and (has_target or bool(memory.last_references))

    @staticmethod
    def _is_coverage_feedback_text(question: str) -> bool:
        """Detect omission-feedback text without relying on memory state."""

        q = " ".join((question or "").strip().lower().split())
        if not q:
            return False

        omission_hints = (
            "漏掉",
            "漏下",
            "遗漏",
            "不全",
            "缺少",
            "没覆盖",
            "补充",
            "少了",
            "missing",
            "omitted",
            "incomplete",
        )
        target_hints = (
            "公司",
            "企业",
            "标的",
            "全部",
            "所有",
            "all companies",
            "all firms",
        )
        return any(token in q for token in omission_hints) and any(token in q for token in target_hints)

    @staticmethod
    def _build_coverage_followup_query(question: str, memory: AgentMemory | None) -> str:
        """Construct a retrieval query that emphasizes complete coverage."""

        base = ""
        if memory is not None:
            base = " ".join((memory.last_question or "").strip().split())
        if not base:
            base = " ".join((question or "").strip().split())
        if not base:
            return "用户问题"
        return f"{base} 请覆盖所有公司并避免遗漏，按公司逐一给出依据"

    def _parse_steps(self, raw: str, memory: AgentMemory | None, question: str) -> list[PlannedStep]:
        """Parse model JSON into validated, normalized planned steps."""

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
            if tool == "finish":
                break
            if tool == "budget_analyst" and not self._is_budget_analysis_request(question):
                continue
            text = str(item.get("input", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if not text:
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
