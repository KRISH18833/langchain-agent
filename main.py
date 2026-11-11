# Entry point for running the LangChain agent.
import os
import sys
from dotenv import load_dotenv
from agent import run_agent

load_dotenv()


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY is not set or empty in .env. Please configure your API key.", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is 123456789 * 987654321?"

    run_agent(question)


if __name__ == "__main__":
    main()
