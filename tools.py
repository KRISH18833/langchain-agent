# Custom tools for the LangChain agent project.
import ast
import operator
import subprocess
import sys
from ddgs import DDGS
from langchain_core.tools import tool

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_UNARY_OPERATORS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise TypeError(f"Unsupported constant type: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise TypeError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPERATORS:
            raise TypeError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_ast(node.operand)
        return _UNARY_OPERATORS[op_type](operand)
    raise TypeError(f"Unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> str:
    try:
        parsed = ast.parse(expression.strip(), mode="eval")
        result = _eval_ast(parsed.body)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError as e:
        return f"Error: Invalid syntax ({e.msg if hasattr(e, 'msg') else 'syntax error'})"
    except Exception as e:
        return f"Error: {e}"


def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="us-en", safesearch="off", max_results=5)
        if not results:
            return f"No results found for: {query}"
        formatted = []
        for i, item in enumerate(results[:5], 1):
            title = item.get("title", "").strip()
            if len(title) > 120:
                title = title[:120] + "..."
            snippet = item.get("body", "").strip()
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            url = item.get("href", "").strip()
            formatted.append(f"{i}. {title}\n   Snippet: {snippet}\n   URL: {url}")
        return "\n".join(formatted)
    except Exception as e:
        if "no results found" in str(e).lower():
            return f"No results found for: {query}"
        return f"Error: {e}"


def python_repl(code: str) -> str:
    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            out = res.stdout
            if len(out) > 2000:
                out = out[:2000] + "... [truncated]"
            return out
        err = res.stderr.strip() or f"Process exited with code {res.returncode}"
        if len(err) > 2000:
            err = err[:2000] + "... [truncated]"
        return f"Error: {err}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out after 10 seconds"
    except Exception as e:
        return f"Error: {e}"


@tool("calculator")
def calculator_tool(expression: str) -> str:
    """Exact arithmetic on numbers. Use for any math calculation. The result is final and exact; do not verify it with another tool. Supports + - * / ** % // and parentheses. Input example: '123456789 * 987654321'."""
    return calculator(expression)


@tool("web_search")
def web_search_tool(query: str) -> str:
    """Search the web for current or real-time information (news, recent versions, prices, events, facts that may have changed). Do NOT use for general knowledge you already know. Input: a short search query."""
    return web_search(query)


@tool("python_repl")
def python_repl_tool(code: str) -> str:
    """Execute Python code for tasks that need real code execution (loops, algorithms, data processing, text manipulation). Do NOT use for simple arithmetic; use calculator instead. Always use print() to show results."""
    return python_repl(code)


ALL_TOOLS = [calculator_tool, web_search_tool, python_repl_tool]


if __name__ == "__main__":
    queries = [
        "latest Python release",
        "current Python version",
        "population of India 2026",
        "zxqwvk99281jjh",
    ]
    for q in queries:
        print(f"=== Query: '{q}' ===")
        print(web_search(q))
        print()
