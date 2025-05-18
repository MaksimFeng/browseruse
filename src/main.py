# """browser_agent.py
# OpenAI‑ or Ollama‑powered Playwright agent with **budget guard**.

# Key points
# ──────────
# * Keeps the Playwright browser **open for the entire agent run** (fixes
#   `Browser.new_context: Target page, context or browser has been closed`).
# * Uses OpenAI first; falls back to local Ollama (`llama3`) if quota is exhausted.
# * Tracks cumulative spend in `~/.browser_agent_budget.json` and blocks when the
#   user‑defined USD cap is reached.
# * Silences LangSmith tracing unless opted in.

# Run:
#     python -m browser_agent "https://example.com" "Return the page title"
# """
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List
from warnings import filterwarnings

from playwright.async_api import async_playwright
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.playwright import (
    PlayWrightBrowserToolkit,
)
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama  # pip install -U langchain-ollama
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from openai.error import OpenAIError  # SDK ≥1.0
except ImportError:
    from openai import OpenAIError

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BUDGET_LIMIT_USD = float(os.getenv("BROWSER_AGENT_BUDGET", "10.0"))
BUDGET_FILE = Path.home() / ".browser_agent_budget.json"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-0125")
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

# Hard‑coded token price table (USD / 1K)
_PRICES = {
    "gpt-4o-mini": 0.0005,
    "gpt-4o": 0.005,
}

# Disable LangSmith spam unless the user really wants tracing
filterwarnings("ignore", category=UserWarning, module="langsmith")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RunResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    cost: float


def _load_spend() -> float:
    if BUDGET_FILE.exists():
        try:
            return json.loads(BUDGET_FILE.read_text()).get("spent", 0.0)
        except Exception:
            return 0.0
    return 0.0


def _save_spend(spent: float) -> None:
    BUDGET_FILE.write_text(json.dumps({"spent": spent}))


def _price(model: str, prompt_t: int, compl_t: int) -> float:
    rate = _PRICES.get(model, 0.002)
    return rate * (prompt_t + compl_t) / 1000


# ──────────────────────────────────────────────────────────────────────────────
# Core logic (single Playwright context)
# ──────────────────────────────────────────────────────────────────────────────
async def run_task(url: str, user_task: str, model: str = DEFAULT_MODEL, max_tokens: int = MAX_TOKENS) -> RunResult:
    spent_so_far = _load_spend()
    if spent_so_far >= BUDGET_LIMIT_USD:
        raise RuntimeError(f"Budget cap ({BUDGET_LIMIT_USD}$) exhausted. Spent: {spent_so_far}$")

    # Open Playwright *once* and keep it until we finish all agent calls
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        tools = toolkit.get_tools()
        from langchain import hub
        react_prompt = hub.pull("hwchase17/react")

        # Helper to build an executor for any LLM instance
        def _executor_for(llm) -> AgentExecutor:
            agent = create_react_agent(llm, tools, react_prompt)
            return AgentExecutor(agent=agent, tools=tools, verbose=False)

        # First try OpenAI
        ai_llm = ChatOpenAI(model=model, max_tokens=max_tokens, temperature=TEMPERATURE)
        executor = _executor_for(ai_llm)

        async def _invoke(exec_: AgentExecutor):
            return await exec_.ainvoke({"input": f"Go to {url}. {user_task}"})

        try:
            result = await _invoke(executor)
        except OpenAIError as e:
            # Explicit quota error triggers fallback
            if "insufficient_quota" in str(e):
                print("[Info] OpenAI quota exhausted – switching to local Ollama model.")
                # Build Ollama executor (if daemon ready)
                try:
                    local_llm = ChatOllama(model="llama3")
                    _ = local_llm.invoke("ping")  # health‑check
                    executor = _executor_for(local_llm)
                    result = await _invoke(executor)
                except Exception as ollama_err:
                    print(f"[Warn] Ollama unavailable – {ollama_err}. Returning stub answer.")
                    result = {
                        "output": "[Fallback‑echo] No local LLM available.",
                        "__llm_output__": {"usage": {"prompt_tokens": 0, "completion_tokens": 0}},
                    }
            else:
                raise

        usage = result.get("__llm_output__", {}).get("usage", {})
        p_t, c_t = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        cost = _price(model, p_t, c_t)
        if cost:
            _save_spend(spent_so_far + cost)

        return RunResult(result["output"], p_t, c_t, cost)


# ──────────────────────────────────────────────────────────────────────────────
# CLI wrapper
# ──────────────────────────────────────────────────────────────────────────────
async def _cli_entry(argv: List[str] | None = None):
    global BUDGET_LIMIT_USD

    parser = argparse.ArgumentParser(description="Playwright‑controlled LLM agent with budget guard")
    parser.add_argument("url")
    parser.add_argument("task", help="Instruction, e.g. 'Return the page title'")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--budget", type=float, default=BUDGET_LIMIT_USD, help="Monthly USD cap")
    args = parser.parse_args(argv)

    BUDGET_LIMIT_USD = args.budget

    try:
        res = await run_task(args.url, args.task, args.model, args.max_tokens)
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Answer: {res.answer}\n"
        f"Tokens: {res.prompt_tokens}+{res.completion_tokens}  Cost: {res.cost:.6f} USD"
        f"  |  Spent total: {_load_spend():.4f} USD"
    )


def main() -> None:
    try:
        asyncio.run(_cli_entry())
    except RuntimeError as e:
        # Accommodate Jupyter / Streamlit
        if "asyncio.run()" in str(e) and "running event loop" in str(e):
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.get_event_loop().run_until_complete(_cli_entry())
        else:
            raise


if __name__ == "__main__":
    main()
