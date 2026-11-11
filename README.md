# LangChain Tool-Using Agent

A modular Python project implementing a LangChain tool-using agent powered by Google Gemini.

## Project Structure

```
langchain-agent/
├── tools.py          # Custom tools for the agent
├── agent.py          # Agent initialization and tool configuration
├── main.py           # Application entry point
├── requirements.txt  # Project dependencies
├── .env.example      # Example environment variables template
├── .env              # Environment variables (API keys)
├── .gitignore        # Git ignore rules
└── README.md         # Project documentation
```

## Setup

1. **Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**:
   Copy `.env.example` to `.env` (or update `.env`) with your Google Gemini API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## Running the Agent

Run with default prompt:
```bash
python main.py
```

Run with custom query:
```bash
python main.py "Search for the latest Python version"
```

## Running Tests

Run the automated test suite across single-tool, multi-step, and tool-failure scenarios:
```bash
python run_tests.py
```
Output is saved to `test_output.txt`.

