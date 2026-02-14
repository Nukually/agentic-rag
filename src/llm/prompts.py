RAG_SYSTEM_PROMPT = """你是一个严谨的 RAG 助手。
请仅基于给定的检索上下文回答问题，禁止编造事实。
如果上下文不足以回答，请明确说明“我在当前知识库中没有找到足够信息”。
回答时优先简洁，并在关键结论后标注引用，如 [ref:1]。
"""

AGENT_PLANNER_SYSTEM_PROMPT = """你是一个任务规划器。你要把用户问题拆成工具步骤。
仅输出 JSON，不要输出其他文字。JSON 格式：
{"steps":[{"tool":"retrieve|calculate|budget_analyst|finish","input":"...","reason":"..."}]}
规则：
1) 需要事实依据时优先 retrieve。
2) 需要算术计算时使用 calculate，input 必须是可执行表达式（例如 A + B - C 或 12.5*3）。
3) 需要基于年度预算分析股价并给出评级时使用 budget_analyst（通常先 retrieve）。
4) 步骤总数不超过 4。
"""

AGENT_ROUTER_SYSTEM_PROMPT = """你是一个分类助手。
请判断用户的问题是“闲聊”、“需要查询知识库”还是“其他”。
只输出类别名称，不要输出多余文字。
"""

AGENT_GENERAL_SYSTEM_PROMPT = """你是一个可靠的通用助理。
回答应简洁清晰；如果问题需要外部资料但未提供，说明无法确认。
"""

AGENT_CHITCHAT_SYSTEM_PROMPT = """你是一个友好的聊天助手。
保持简短自然，直接回应用户即可。
"""

AGENT_FINAL_SYSTEM_PROMPT = """你是一个严谨的 Agentic RAG 助手。
你会收到工具执行轨迹与检索上下文，请基于这些信息回答用户。
禁止编造；若信息不足请明确说明。关键结论后标注引用 [ref:n]。
"""


def build_user_prompt(question: str, contexts: list[dict[str, str]]) -> str:
    if not contexts:
        context_text = "<NO_CONTEXT>"
    else:
        blocks: list[str] = []
        for i, item in enumerate(contexts, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"[ref:{i}] source={item['source']} page={item['page']}",
                        item["text"],
                    ]
                )
            )
        context_text = "\n\n".join(blocks)

    return (
        "请使用以下检索上下文回答问题。\n\n"
        f"=== 检索上下文开始 ===\n{context_text}\n=== 检索上下文结束 ===\n\n"
        f"用户问题：{question}"
    )


def build_agent_plan_prompt(
    question: str,
    max_steps: int,
    memory_summary: str = "<none>",
    recent_history: list[dict[str, str]] | None = None,
    replan_feedback: str | None = None,
    previous_steps: list[object] | None = None,
    previous_observations: list[str] | None = None,
) -> str:
    recent_history = recent_history or []
    if not recent_history:
        history_text = "<none>"
    else:
        lines: list[str] = []
        for item in recent_history:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            lines.append(f"{role}: {content}")
        history_text = "\n".join(lines)

    replan_text = "<none>"
    if replan_feedback:
        replan_text = replan_feedback.strip() or "<none>"

    prev_steps_text = "<none>"
    if previous_steps:
        lines: list[str] = []
        for i, step in enumerate(previous_steps, start=1):
            tool = getattr(step, "tool", "")
            tool_input = getattr(step, "input", "")
            reason = getattr(step, "reason", "")
            lines.append(f"{i}. tool={tool} input={tool_input} reason={reason}")
        prev_steps_text = "\n".join(lines) if lines else "<none>"

    prev_obs_text = "<none>"
    if previous_observations:
        lines = [obs for obs in previous_observations if str(obs or "").strip()]
        prev_obs_text = "\n".join(lines) if lines else "<none>"

    return (
        f"用户问题：{question}\n\n"
        f"记忆摘要：{memory_summary}\n\n"
        f"最近对话：\n{history_text}\n\n"
        f"重规划反馈：\n{replan_text}\n\n"
        f"上一次计划：\n{prev_steps_text}\n\n"
        f"上一次观察：\n{prev_obs_text}\n\n"
        f"请输出不超过 {max_steps} 步的工具计划，仅 JSON。"
        "如果这是重规划，请避免重复导致失败的步骤顺序。"
    )


def build_agent_router_prompt(question: str) -> str:
    return f"用户问题：{question}\n\n请输出类别名称。"


def build_agent_answer_prompt(
    question: str,
    tool_traces: list[object],
    contexts: list[dict[str, str]],
) -> str:
    trace_lines: list[str] = []
    for step in tool_traces:
        idx = getattr(step, "step_no", "")
        tool = getattr(step, "tool", "")
        tool_input = getattr(step, "tool_input", "")
        obs = getattr(step, "observation", "")
        trace_lines.append(f"[step:{idx}] tool={tool} input={tool_input}\nobs={obs}")
    trace_text = "\n\n".join(trace_lines) if trace_lines else "<NO_TRACE>"

    if not contexts:
        ctx_text = "<NO_CONTEXT>"
    else:
        blocks: list[str] = []
        for i, item in enumerate(contexts, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"[ref:{i}] source={item['source']} page={item['page']}",
                        item["text"],
                    ]
                )
            )
        ctx_text = "\n\n".join(blocks)

    return (
        f"用户问题：{question}\n\n"
        f"=== 工具执行轨迹 ===\n{trace_text}\n=== 轨迹结束 ===\n\n"
        f"=== 检索上下文 ===\n{ctx_text}\n=== 上下文结束 ==="
    )
