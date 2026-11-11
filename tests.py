# Test runner for the LangChain agent.
import time
import tools
from agent import run_agent

TESTS = [
    (
        "Single tool (calculator)",
        "What is 2 ** 100 + 987654321 * 123456789?",
        None,
    ),
    (
        "Multi-step (search + calculate)",
        (
            "Search for the current population of India, then calculate how many people "
            "that is per square kilometer given India's area is 3,287,263 km2."
        ),
        None,
    ),
    (
        "Tool failure (calculator)",
        "Use the calculator tool to compute 100 / 0 and report what it returns.",
        None,
    ),
    (
        "Tool failure (search)",
        "What is the latest Python release?",
        "network_down",  # signals monkeypatch
    ),
]


def _network_down(query: str) -> str:
    raise RuntimeError("network down")


for n, (label, question, patch) in enumerate(TESTS, 1):
    print(f"=== TEST {n}: {label} ===")

    if patch == "network_down":
        original = tools.web_search
        tools.web_search = _network_down
        # Also patch through the tool wrapper's underlying call.
        # The @tool wrapper calls tools.web_search via the module-level name,
        # but since Python closures capture the global, we patch the module attribute.

    try:
        run_agent(question)
    finally:
        if patch == "network_down":
            tools.web_search = original

    print()

    if n < len(TESTS):
        print("[sleeping 5s to avoid rate limits]")
        time.sleep(5)
        print()

