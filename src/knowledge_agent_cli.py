import argparse
import asyncio
import json

from knowledge_agent import build_knowledge_agent
from knowledge_agent.factory import build_llm_from_env


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OneKE Structured RAG Knowledge Agent")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--no-web", action="store_true", help="Disable Tavily MCP fallback")
    args = parser.parse_args()

    agent = build_knowledge_agent(build_llm_from_env())
    result = await agent.ainvoke(args.question, allow_web=not args.no_web)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

