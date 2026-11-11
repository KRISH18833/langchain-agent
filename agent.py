# LangChain tool-calling agent definition and initialization.
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import ALL_TOOLS

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = (
    "You are a helpful assistant with three tools: calculator (exact math), "
    "web_search (current information), python_repl (run code). Use a tool only "
    "when needed. Answer simple or general-knowledge questions directly without tools. "
    "For multi-step tasks, call tools one after another and use the real results from earlier steps. "
    "Trust a tool result that does not start with 'Error:'. Never re-run or re-verify a successful "
    "result with another tool; give the final answer immediately. If a tool returns an error, "
    "explain it briefly and try a different approach or give your best answer. "
    "Before each tool call, briefly state why you are using it."
)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def build_agent():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is empty or not set in .env")

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, api_key=api_key)
    agent = create_agent(model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
    return agent


def _force_final_answer(llm, messages: list) -> str:
    """Call the LLM once with no tools, asking it to give a final answer."""
    prompt_msgs = messages + [
        HumanMessage(content="Give your final answer now using the tool results above.")
    ]
    response = llm.invoke(prompt_msgs)
    return _extract_text(response.content).strip()


def run_agent(question: str) -> str:
    try:
        agent = build_agent()
        api_key = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, api_key=api_key)

        printed_count = 0
        final_answer = ""

        # Loop guard state
        seen_calls: set[tuple] = set()          # (tool_name, frozen_args)
        calculator_success: str | None = None   # stores result if calculator succeeded

        stream = agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
            config={"recursion_limit": 10},
        )

        all_messages: list = []

        for step in stream:
            messages = step.get("messages", [])
            all_messages = messages  # keep latest snapshot

            for msg in messages[printed_count:]:
                if isinstance(msg, HumanMessage):
                    text = _extract_text(msg.content).strip()
                    print(f"[USER] {text}")

                elif isinstance(msg, AIMessage):
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    text = _extract_text(msg.content).strip()

                    if tool_calls:
                        if text:
                            print(f"[THOUGHT] {text}")

                        for tc in tool_calls:
                            name = tc.get("name", "tool")
                            args = tc.get("args", {})
                            if isinstance(args, dict):
                                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                                frozen = tuple(sorted(args.items()))
                            else:
                                args_str = str(args)
                                frozen = (str(args),)

                            call_key = (name, frozen)

                            # Guard 1: exact duplicate call
                            if call_key in seen_calls:
                                print(f"[LOOP GUARD] Repeated call to {name}, stopping and forcing final answer")
                                answer = _force_final_answer(llm, all_messages)
                                print(f"[FINAL ANSWER] {answer}")
                                return answer

                            # Guard 2: python_repl after a successful calculator result
                            if name == "python_repl" and calculator_success is not None:
                                print(f"[LOOP GUARD] Repeated call to {name}, stopping and forcing final answer")
                                answer = _force_final_answer(llm, all_messages)
                                print(f"[FINAL ANSWER] {answer}")
                                return answer

                            seen_calls.add(call_key)
                            print(f"[TOOL CALL] {name}({args_str})")
                    else:
                        print(f"[FINAL ANSWER] {text}")
                        final_answer = text

                elif isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, "name", None) or "tool"
                    content = _extract_text(msg.content).strip()
                    if len(content) > 500:
                        content = content[:500] + "..."
                    print(f"[TOOL RESULT] {tool_name} -> {content}")

                    # Record a successful calculator result
                    if tool_name == "calculator" and not content.startswith("Error:"):
                        calculator_success = content

            printed_count = len(messages)

        return final_answer
    except Exception as e:
        err_msg = f"[AGENT ERROR] {e}"
        print(err_msg)
        return err_msg
