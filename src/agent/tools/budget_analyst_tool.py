from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.agent.tools.registry import ToolContext, ToolOutput


@dataclass(frozen=True)
class BudgetItem:
    year: int | None
    value: float
    raw: str


class BudgetAnalystTool:
    name = "budget_analyst"

    def run(self, tool_input: str, context: ToolContext) -> ToolOutput:
        text_input = (tool_input or "").strip()
        payload = _parse_json_payload(text_input)

        combined_text = _build_combined_text(
            tool_input=text_input,
            question=context.question,
            retrieval_text=str(
                context.run_state.get("latest_retrieval_text")
                or context.memory.last_retrieval_text
                or ""
            ),
        )

        budgets: list[BudgetItem] = []
        stock_price: float | None = None

        if payload:
            budgets = _parse_budgets_from_json(payload)
            stock_price = _parse_stock_price_from_json(payload)

        if not budgets:
            budgets = _extract_budgets_from_text(combined_text)

        if stock_price is None:
            stock_price = _extract_stock_price_from_text(combined_text)

        analysis = _analyze_budget(budgets=budgets, stock_price=stock_price, raw_text=combined_text)
        return ToolOutput(
            observation=analysis.observation,
            memory_delta=analysis.memory_delta,
            metadata=analysis.metadata,
        )


@dataclass(frozen=True)
class AnalysisResult:
    observation: str
    memory_delta: dict[str, object]
    metadata: dict[str, object]


def _parse_json_payload(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _build_combined_text(tool_input: str, question: str, retrieval_text: str) -> str:
    parts = []
    if tool_input and tool_input != "用户问题":
        parts.append(tool_input)
    if question:
        parts.append(question)
    if retrieval_text:
        parts.append(retrieval_text)
    return "\n".join(p for p in parts if p).strip()


def _parse_budgets_from_json(payload: dict) -> list[BudgetItem]:
    budgets: list[BudgetItem] = []
    raw_items = payload.get("budgets") or payload.get("budget") or []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        return budgets

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        year = item.get("year")
        amount = item.get("amount") or item.get("value")
        unit = item.get("unit") or ""
        try:
            value = float(amount) * _unit_multiplier(str(unit))
        except (TypeError, ValueError):
            continue
        year_val: int | None
        try:
            year_val = int(year) if year is not None else None
        except (TypeError, ValueError):
            year_val = None
        raw = f"{amount}{unit}".strip()
        budgets.append(BudgetItem(year=year_val, value=value, raw=raw))
    return budgets


def _parse_stock_price_from_json(payload: dict) -> float | None:
    for key in ("stock_price", "price", "股价"):
        if key in payload:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                return None
    return None


def _extract_budgets_from_text(text: str) -> list[BudgetItem]:
    if not text:
        return []

    budgets: list[BudgetItem] = []
    seen: set[tuple[int | None, float]] = set()

    year_first = re.compile(
        r"(20\d{2})[^0-9]{0,6}(?:年度|年)?预算[^0-9]{0,6}([0-9]+(?:\.[0-9]+)?)\s*([^\s,，。；;]{0,6})"
    )
    budget_first = re.compile(
        r"(?:年度|年)?预算[^0-9]{0,6}(20\d{2})[^0-9]{0,6}([0-9]+(?:\.[0-9]+)?)\s*([^\s,，。；;]{0,6})"
    )
    for pattern in (year_first, budget_first):
        for match in pattern.finditer(text):
            year = int(match.group(1))
            amount = float(match.group(2))
            unit = match.group(3) or ""
            value = amount * _unit_multiplier(unit)
            key = (year, value)
            if key in seen:
                continue
            seen.add(key)
            budgets.append(BudgetItem(year=year, value=value, raw=f"{amount}{unit}".strip()))

    if budgets:
        return budgets

    no_year = re.compile(
        r"(?:年度|年)?预算[^0-9]{0,6}([0-9]+(?:\.[0-9]+)?)\s*([^\s,，。；;]{0,6})"
    )
    for match in no_year.finditer(text):
        amount = float(match.group(1))
        unit = match.group(2) or ""
        value = amount * _unit_multiplier(unit)
        key = (None, value)
        if key in seen:
            continue
        seen.add(key)
        budgets.append(BudgetItem(year=None, value=value, raw=f"{amount}{unit}".strip()))

    return budgets


def _extract_stock_price_from_text(text: str) -> float | None:
    if not text:
        return None
    price_pattern = re.compile(
        r"(?:股价|股价为|股价是|price|stock\s*price)[^0-9]{0,6}([0-9]+(?:\.[0-9]+)?)",
        flags=re.IGNORECASE,
    )
    match = price_pattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _unit_multiplier(unit: str) -> float:
    if not unit:
        return 1.0
    u = unit.strip().lower()
    for token in ("人民币", "美元", "元", "圆", "rmb", "cny", "usd", "¥", "$"):
        u = u.replace(token, "")
    u = u.strip()

    if not u:
        return 1.0
    if "万亿" in u:
        return 1e12
    if "十亿" in u:
        return 1e9
    if "亿" in u:
        return 1e8
    if "千万" in u:
        return 1e7
    if "百万" in u:
        return 1e6
    if "万" in u:
        return 1e4
    if "千" in u:
        return 1e3
    if "百" in u:
        return 1e2

    if "billion" in u or u in {"b", "bn"}:
        return 1e9
    if "million" in u or u == "m":
        return 1e6
    if "thousand" in u or u == "k":
        return 1e3

    return 1.0


def _format_amount(value: float) -> str:
    if value >= 1e12:
        return f"{value / 1e12:.2f}万亿"
    if value >= 1e8:
        return f"{value / 1e8:.2f}亿"
    if value >= 1e4:
        return f"{value / 1e4:.2f}万"
    return f"{value:.2f}"


def _detect_budget_tone(text: str) -> int:
    if not text:
        return 0
    if re.search(r"(下调|削减|减少|缩减|压缩).{0,6}预算|预算.{0,6}(下调|削减|减少|缩减|压缩)", text):
        return -1
    if re.search(r"(上调|增加|提升|扩张).{0,6}预算|预算.{0,6}(上调|增加|提升|扩张)", text):
        return 1
    return 0


def _select_latest_and_prev(budgets: list[BudgetItem]) -> tuple[BudgetItem | None, BudgetItem | None]:
    with_year = [item for item in budgets if item.year is not None]
    if with_year:
        with_year.sort(key=lambda item: item.year or 0)
        latest = with_year[-1]
        prev = with_year[-2] if len(with_year) >= 2 else None
        return latest, prev

    if budgets:
        return budgets[-1], None
    return None, None


def _analyze_budget(budgets: list[BudgetItem], stock_price: float | None, raw_text: str) -> AnalysisResult:
    latest, prev = _select_latest_and_prev(budgets)
    growth_pct: float | None = None
    notes: list[str] = []

    if latest and prev and prev.value > 0:
        growth_pct = (latest.value - prev.value) / prev.value * 100.0
        notes.append("已计算年度预算增速")
    elif latest:
        notes.append("仅找到单一年份预算，无法计算增速")
    else:
        notes.append("未找到年度预算数据")

    score = 0
    if growth_pct is not None:
        if growth_pct >= 15:
            score += 2
        elif growth_pct >= 5:
            score += 1
        elif growth_pct <= -15:
            score -= 2
        elif growth_pct <= -5:
            score -= 1

    tone = _detect_budget_tone(raw_text)
    if tone != 0:
        score += tone
        notes.append("文本提示预算上/下调")

    if latest is None:
        rating = "无法评级"
    else:
        if score >= 2:
            rating = "买入"
        elif score == 1:
            rating = "增持"
        elif score == 0:
            rating = "中性"
        elif score == -1:
            rating = "减持"
        else:
            rating = "卖出"

    budget_lines: list[str] = []
    for item in sorted(budgets, key=lambda x: (x.year is None, x.year or 0)):
        label = str(item.year) if item.year is not None else "年度预算"
        budget_lines.append(f"{label}={_format_amount(item.value)}")
    budget_summary = ", ".join(budget_lines) if budget_lines else "<none>"

    latest_text = (
        f"{latest.year if latest and latest.year is not None else '最新'}={_format_amount(latest.value)}"
        if latest
        else "<none>"
    )
    prev_text = (
        f"{prev.year if prev and prev.year is not None else '上期'}={_format_amount(prev.value)}"
        if prev
        else "<none>"
    )
    growth_text = f"{growth_pct:.2f}%" if growth_pct is not None else "<none>"
    price_text = f"{stock_price:.4f}" if stock_price is not None else "<none>"

    observation = (
        "budget_analyst:"
        f" rating={rating}; score={score}; "
        f"budget_latest={latest_text}; budget_prev={prev_text}; "
        f"budget_growth_pct={growth_text}; stock_price={price_text}; "
        f"budgets={budget_summary}; notes={' | '.join(notes)}"
    )

    variables: dict[str, float] = {}
    if latest:
        variables["BUDGET_LATEST"] = latest.value
    if prev:
        variables["BUDGET_PREV"] = prev.value
    if growth_pct is not None:
        variables["BUDGET_GROWTH_PCT"] = growth_pct
    if stock_price is not None:
        variables["STOCK_PRICE"] = stock_price
    variables["BUDGET_ANALYST_SCORE"] = float(score)

    memory_delta = {"variables": variables, "tool_observations": {"budget_analyst": observation}}
    metadata = {
        "rating": rating,
        "score": score,
        "budget_growth_pct": growth_pct,
        "stock_price": stock_price,
    }
    return AnalysisResult(observation=observation, memory_delta=memory_delta, metadata=metadata)
