"""Ask the agent a question from the terminal.

Usage:  python -m rdagent.cli "Does the OV-600P meet the fan noise requirement?"
"""

from __future__ import annotations

import argparse
import json

from rdagent.agent import BaselineRAG, RDAgent
from rdagent.config import get_settings
from rdagent.llm import OllamaClient
from rdagent.tools import KnowledgeBase


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--model", default=settings.model)
    parser.add_argument("--host", default=settings.ollama_host)
    parser.add_argument("--baseline", action="store_true", help="single-shot RAG without tools")
    args = parser.parse_args(argv)

    kb = KnowledgeBase.from_dir(settings.data_dir)
    llm = OllamaClient(model=args.model, host=args.host)
    system = BaselineRAG(llm, kb) if args.baseline else RDAgent(llm, kb, settings.max_steps)
    result = system.run(args.question)

    for i, step in enumerate(result.steps, 1):
        status = f"error: {step.error}" if step.error else f"sources: {', '.join(step.sources) or '-'}"
        print(f"  step {i}: {step.tool}({json.dumps(step.arguments)}) -> {status}")
    print()
    print(result.answer)
    print()
    print(
        f"citations: {', '.join(result.citations) or 'none'} | "
        f"invalid: {', '.join(result.invalid_citations) or 'none'} | "
        f"{result.latency_s:.1f} s, {result.llm_calls} model calls"
    )


if __name__ == "__main__":
    main()
