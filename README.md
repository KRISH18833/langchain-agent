# LangChain Tool-Calling Agent

A robust, safeguard-protected autonomous agent built with LangChain and Google Gemini that solves multi-step queries using custom tools for exact math, web search, and Python code execution.

## Features

- **Deterministic Math & Code Execution**: Offloads multi-step arithmetic and algorithms to dedicated Python AST and REPL environments.
- **Live Web Retrieval**: Integrates `ddgs` for real-time information with automated retries and resilient query handling.
- **Defensive Tool Wrapping**: All tool wrappers enforce a strict no-raise guarantee, returning structured error messages instead of crashing.
- **Multi-Tier Safeguards & Loop Guards**: Prevents infinite tool-calling loops, redundant cross-tool verification, error cascading, and recursion limit exhaustion.
- **Step-by-Step Trajectory Logging**: Clean, formatted terminal output capturing user input, model thoughts, tool calls, tool results, and final answers.
- **Graceful Error Recovery**: Falls back to internal parametric knowledge when external tools encounter persistent network or environment failures.

## Tools

| Tool Name | What It Does | Why the LLM Can't Do It Reliably |
| :--- | :--- | :--- |
| `calculator` | Evaluates arithmetic expressions safely using Python's `ast` module (supporting `+`, `-`, `*`, `/`, `**`, `%`, `//`, parentheses, and large integers). | LLMs predict tokens probabilistically and frequently hallucinate digits when performing complex or multi-digit arithmetic (e.g. $2^{100}$). |
| `web_search` | Queries DuckDuckGo via `ddgs` with search retry logic, returning structured titles, URLs, and summaries. | LLMs have fixed training cutoff dates and lack access to real-time events, current statistics, or up-to-date documentation. |
| `python_repl` | Executes arbitrary Python code in an isolated subprocess with a 10-second timeout and character truncation. | LLMs cannot execute live algorithms, data processing, iterative loops, or string manipulation that require runtime validation. |

## Architecture

- **Framework**: LangChain `create_agent` (built on LangGraph).
- **Model**: `gemini-3.5-flash-lite` (via `langchain-google-genai`).
- **No-Raise Guarantee**: Every `@tool` wrapper (`calculator_tool`, `web_search_tool`, `python_repl_tool`) wraps execution in a `try...except` block, ensuring tools return formatted string errors (`Error: <reason>`) rather than raising exceptions into the LangGraph engine.
- **Safeguards (`run_agent`)**:
  1. **Repeated Call Guard**: Detects and halts when the exact same tool and arguments are called consecutively, forcing an immediate final answer using the prior result.
  2. **Calculator-then-Python REPL Guard**: Blocks attempts to re-verify exact calculator results with `python_repl`, passing calculated digits verbatim.
  3. **Per-Tool Limit (`MAX_CALLS_PER_TOOL = 3`)**: Prevents any single tool from being invoked more than 3 times within a single run.
  4. **Error Streak Guard**: Halts execution if a tool returns `"Error:"` twice in a row, prompting the model to use existing knowledge and report the failure.
  5. **Total Call Ceiling**: Triggers force-final-answer if total tool calls reach 5 across the entire trajectory.
- **Recursion Limit**: Set to `10` graph steps.

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd langchain-agent
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and provide your Google Gemini API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## Usage

Run with default question:
```bash
python main.py
```

Run with a custom question:
```bash
python main.py "What is 123456789 * 987654321?"
```

Run the automated test suite:
```bash
python tests.py
```

Or run the runner that pipes output to `test_output.txt`:
```bash
python run_tests.py
```

## Example Runs

Below are the verbatim outputs from the automated test suite (`test_output.txt`):

- **TEST 0 proves the agent can answer simple general-knowledge questions directly without unnecessary tool calls.**
- **TEST 1 proves the calculator handles exact high-precision integer arithmetic and that the loop guard halts redundant re-verification.**
- **TEST 2 proves multi-step tool chaining where demographic data retrieved via web search is piped directly into the calculator.**
- **TEST 3 proves the agent safely traps tool-level division by zero and reports the error cleanly.**
- **TEST 4 proves the error streak guard detects persistent network failures after 2 consecutive errors and provides a graceful fallback answer without crashing.**

```
=== TEST 0: No tool needed ===
[USER] What is the capital of France?
D:\Valentius Kryptix\langchain-agent\venv\Lib\site-packages\langchain_google_genai\chat_models.py:3935: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[FINAL ANSWER] The capital of France is Paris.

[sleeping 5s to avoid rate limits]

=== TEST 1: Single tool (calculator) ===
[USER] What is 2 ** 100 + 987654321 * 123456789?
D:\Valentius Kryptix\langchain-agent\venv\Lib\site-packages\langchain_google_genai\chat_models.py:3935: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[TOOL CALL] calculator(expression='2**100 + 987654321 * 123456789')
[TOOL RESULT] calculator -> 1267650600228351334127815840645
D:\Valentius Kryptix\langchain-agent\venv\Lib\site-packages\langchain_google_genai\chat_models.py:3935: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[LOOP GUARD] Repeated call to python_repl, stopping and forcing final answer
D:\Valentius Kryptix\langchain-agent\venv\Lib\site-packages\langchain_google_genai\chat_models.py:3935: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[FINAL ANSWER] To find the exact value of $2^{100} + 987654321 \times 123456789$, we compute both parts:

1. $2^{100} = 1,267,650,600,228,229,401,496,703,205,376$
2. $987,654,321 \times 123456789 = 121,932,631,112,635,269$

Adding them together gives the exact integer:

**1267650600228351334127815840645**

[sleeping 5s to avoid rate limits]

=== TEST 2: Multi-step (search + calculate) ===
[USER] Search for the current population of India, then calculate how many people that is per square kilometer given India's area is 3,287,263 km2.
D:\Valentius Kryptix\langchain-agent\venv\Lib\site-packages\langchain_google_genai\chat_models.py:3935: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[TOOL CALL] web_search(query='current population of India 2026 or 2025')
[TOOL RESULT] web_search -> 1. India Population (2026) - Worldometer
   Snippet: Population of India: current, historical, and projected population, growth rate, immigration, median age, total fertility rate (TFR), population density, urbanization ...
   URL: https://www.worldometers.info/world-population/india-population/
2. India Population 2026 (Live) | State & District Wise Demographics
   Snippet: 16 hours ago · Current Population of India 2026 estimates and official Census 2011 data. View state-wise population, demographics, literacy rates, and sex ratio statistics.
   URL: http://indiacensus.in/
3. India Population 2026
   Snippet: Details and statistics about India, a country in Asia known for the Taj Mahal and diverse cultural heritage.
   URL: https://worldpopulationreview.com/countries/india
4. India Population In November 2026: 1.47 Billion (36 States ...
   Snippet: 16 hours ago · India's population stands at 1.47 billion in 2026, making it the world's most populous country. Track live counts across 36 states & UT's, fertility, and sex ratio.
   URL: https://worldpopulationclock.net/india-live-population-clock-by-state/
5. India Population (1950-2026) - Macrotrends
   Snippet: Total current population for India in 2026 is 1,467,231,210, a 0.87% increase from 2025. Total population for India in 2025 was 1,454,606,724, a 0.89% increase from 2024.
   URL: https://www.macrotrends.net/global-metrics/countries/ind/india/population
D:\Valentius Kryptix\langchain-agent\venv\Lib\site-packages\langchain_google_genai\chat_models.py:3935: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[TOOL CALL] calculator(expression='1467231210 / 3287263')
[TOOL RESULT] calculator -> 446.3382485672731
D:\Valentius Kryptix\langchain-agent\venv\Lib\site-packages\langchain_google_genai\chat_models.py:3935: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[FINAL ANSWER] According to current estimates, India's population is approximately **1,467,231,210** (about 1.47 billion). 

Given India's area of 3,287,263 km², the population density is approximately **446 people per square kilometer**.

[sleeping 5s to avoid rate limits]

=== TEST 3: Tool failure (calculator) ===
[USER] Use the calculator tool to compute 100 / 0 and report what it returns.
D:\Valentius Kryptix\langchain-agent\venv\Lib\site-packages\langchain_google_genai\chat_models.py:3935: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[TOOL CALL] calculator(expression='100 / 0')
[TOOL RESULT] calculator -> Error: Division by zero
D:\Valentius Kryptix\langchain-agent\venv\Lib\site-packages\langchain_google_genai\chat_models.py:3935: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[FINAL ANSWER] The calculator tool returned an error when attempting to divide by zero: "Error: Division by zero".

[sleeping 5s to avoid rate limits]

=== TEST 4: Tool failure (search) ===
[USER] What is the latest Python release?
D:\Valentius Kryptix\langchain-agent\venv\Lib\site-packages\langchain_google_genai\chat_models.py:3935: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[TOOL CALL] web_search(query='latest Python release version python.org')
[TOOL RESULT] web_search -> Error: network down
D:\Valentius Kryptix\langchain-agent\venv\Lib\site-packages\langchain_google_genai\chat_models.py:3935: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[TOOL CALL] web_search(query='latest python release version')
[TOOL RESULT] web_search -> Error: network down
[LOOP GUARD] web_search failed twice, stopping and forcing final answer
D:\Valentius Kryptix\langchain-agent\venv\Lib\site-packages\langchain_google_genai\chat_models.py:3935: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[FINAL ANSWER] The tool failed to retrieve live data, so my answer is based on my knowledge cutoff of January 2025 and may be out of date. 

As of my last update, the latest stable Python releases were in the **Python 3.13** series (such as Python 3.13.0, which was released in October 2024), with Python 3.14 currently in development. 

To check for the absolute newest release, please visit the official [Python Downloads page](https://www.python.org/downloads/).
```

## Notes

- **TEST 1 Re-verification**: In TEST 1, the model attempted to re-verify the calculator result with `python_repl`. The loop guard successfully intercepted this redundant call and forced the final answer using the exact calculated digits verbatim.
- **TEST 4 Fallback**: In TEST 4, with `web_search` monkeypatched to simulate network failure, the error streak guard halted execution after two consecutive failures. The fallback answer was drawn directly from the model's internal parametric knowledge, clearly noting potential staleness.
- **Model Compatibility**: The default configured model is `gemini-3.5-flash-lite`. Depending on your Google AI Studio tier or regional API access, this can be switched to `gemini-2.5-flash` or `gemini-1.5-flash` in `agent.py` by modifying `MODEL_NAME`.
