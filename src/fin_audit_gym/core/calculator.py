"""Safe AST-based arithmetic calculator without eval() or exec()."""

import ast
import decimal
from decimal import Decimal

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)


def _eval_ast(node: ast.AST) -> Decimal:
    """Recursively evaluate allowed AST nodes using exact Decimal arithmetic."""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")  # noqa: TRY004
        return Decimal(str(node.value))

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("Division by zero")
            try:
                return left / right
            except (decimal.DivisionByZero, decimal.InvalidOperation) as exc:
                raise ValueError("Division by zero") from exc
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ValueError("Division by zero")
            try:
                return left % right
            except (decimal.DivisionByZero, decimal.InvalidOperation) as exc:
                raise ValueError("Division by zero") from exc
        raise ValueError(f"Disallowed binary operator: {type(node.op).__name__}")

    raise ValueError(f"Disallowed AST node: {type(node).__name__}")


def safe_calculate(expression: str) -> str:
    """Safely calculate an arithmetic expression string using AST validation.

    Only basic arithmetic operations (+, -, *, /, %) and numbers are allowed.
    All functions, variable names, attributes, imports, and statement executions
    are strictly rejected with ValueError.
    """
    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty")

    cleaned = expression.strip()
    if len(cleaned) > 1000:
        raise ValueError(
            f"Expression length ({len(cleaned)}) exceeds maximum limit of 1000 characters"
        )

    try:
        tree = ast.parse(cleaned, mode="eval")
    except (SyntaxError, RecursionError, MemoryError) as exc:
        raise ValueError(f"Invalid arithmetic expression syntax: {exc}") from exc

    # Strictly whitelist AST nodes across the entire tree
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Disallowed expression element: {type(node).__name__}")  # noqa: TRY004

    try:
        res = _eval_ast(tree)
    except RecursionError as exc:
        raise ValueError("Expression recursion limit exceeded") from exc

    # Clean formatting
    try:
        if res == res.to_integral():
            return str(int(res))
    except (decimal.InvalidOperation, OverflowError):
        pass

    formatted = format(res, "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted
