"""Safe expression calculator based on AST node allow-listing."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CalcResult:
    """Calculation result with resolved variables."""

    expression: str
    value: float
    variables: dict[str, float]


class SafeCalculator:
    """Evaluate arithmetic expressions with strict AST safety checks.

    Supported operators include +, -, *, /, //, %, and **.
    Variable values can be extracted from text patterns like `A=123`.
    """

    _allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Mod,
        ast.FloorDiv,
    )

    def evaluate(
        self,
        expression: str,
        context_text: str = "",
        additional_variables: dict[str, float] | None = None,
    ) -> CalcResult:
        """Evaluate an expression using variables from text and explicit inputs.

        Args:
            expression: Arithmetic expression (e.g. `A + B - 3`).
            context_text: Retrieval text used for variable extraction.
            additional_variables: Pre-supplied variables (e.g. memory values).

        Returns:
            CalcResult: Normalized expression, numeric value, and variables.

        Example:
            >>> calc = SafeCalculator()
            >>> calc.evaluate("A + B", context_text="A=1 B=2").value
            3.0
        """

        normalized = " ".join(expression.strip().split())
        if not normalized:
            raise ValueError("empty expression")

        variables = self.extract_variables(context_text)
        if additional_variables:
            for key, value in additional_variables.items():
                try:
                    variables[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        value = self._eval_ast(normalized, variables)
        return CalcResult(expression=normalized, value=float(value), variables=dict(variables))

    @staticmethod
    def extract_variables(text: str) -> dict[str, float]:
        """Extract uppercase variable assignments from free text."""

        if not text:
            return {}

        mapping: dict[str, float] = {}
        for m in re.finditer(r"\b([A-Z_][A-Z0-9_]*)\b\s*(?:=|:|：)\s*(-?\d+(?:\.\d+)?)", text):
            key = m.group(1).strip()
            value = float(m.group(2))
            mapping[key] = value
        return mapping

    def _eval_ast(self, expression: str, variables: dict[str, float]) -> float:
        """Parse and evaluate expression AST after allow-list validation."""

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"invalid expression: {expression}") from exc

        for node in ast.walk(tree):
            if not isinstance(node, self._allowed_nodes):
                raise ValueError(f"unsupported operation: {type(node).__name__}")

        return float(self._eval_node(tree.body, variables))

    def _eval_node(self, node: ast.AST, variables: dict[str, float]) -> float:
        """Recursively evaluate one AST node."""

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("constant must be int/float")

        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"unknown variable: {node.id}")
            return float(variables[node.id])

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, variables)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("unsupported unary operator")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, variables)
            right = self._eval_node(node.right, variables)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError("unsupported binary operator")

        raise ValueError(f"unsupported expression node: {type(node).__name__}")
