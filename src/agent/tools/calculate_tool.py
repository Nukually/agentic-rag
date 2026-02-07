from __future__ import annotations

from src.agent.tools.calculator import SafeCalculator
from src.agent.tools.registry import ToolContext, ToolOutput


class CalculateTool:
    name = "calculate"

    def __init__(self, calculator: SafeCalculator | None = None) -> None:
        self.calculator = calculator or SafeCalculator()

    def run(self, tool_input: str, context: ToolContext) -> ToolOutput:
        expression = " ".join((tool_input or "").strip().split())
        if not expression:
            return ToolOutput(observation="calc_failed: empty expression")

        known_vars = dict(context.memory.variables)
        if context.memory.last_calc_value is not None:
            known_vars.setdefault("LAST_RESULT", float(context.memory.last_calc_value))

        retrieval_text = str(
            context.run_state.get("latest_retrieval_text")
            or context.memory.last_retrieval_text
            or ""
        )

        try:
            calc = self.calculator.evaluate(
                expression=expression,
                context_text=retrieval_text,
                additional_variables=known_vars,
            )
            vars_text = ", ".join(f"{k}={v}" for k, v in sorted(calc.variables.items())) or "<none>"
            observation = f"expression={calc.expression}; value={calc.value}; vars={vars_text}"
            return ToolOutput(
                observation=observation,
                memory_delta={
                    "last_calc_expression": calc.expression,
                    "last_calc_value": calc.value,
                    "variables": calc.variables,
                    "tool_observations": {"calculate": observation},
                },
                metadata={
                    "calc_value": calc.value,
                    "calc_expression": calc.expression,
                },
            )
        except Exception as exc:
            return ToolOutput(observation=f"calc_failed: {exc}")
