import asyncio
from typing import List

from app.config import config
from app.tool.base import BaseTool, ToolResult
from app.tool.search import (
    BaiduSearchEngine,
    DuckDuckGoSearchEngine,
    GoogleSearchEngine,
    WebSearchEngine,
)


class WebSearch(BaseTool):
    name: str = "web_search"
    description: str = """Perform a web search and return relevant results with titles, URLs, and snippets.
Use this tool when you need to find information on the web, get up-to-date data, or research specific topics."""
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to submit to the search engine.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    }
    # Order matters: first is preferred, rest are fallbacks.
    _engine_order: List[str] = ["duckduckgo", "google", "baidu"]
    _search_engine: dict = {
        "google": GoogleSearchEngine(),
        "baidu": BaiduSearchEngine(),
        "duckduckgo": DuckDuckGoSearchEngine(),
    }

    async def execute(self, query: str, num_results: int = 10) -> ToolResult:
        loop = asyncio.get_event_loop()

        for engine_name in self._ordered_engines():
            engine = self._search_engine.get(engine_name)
            if engine is None:
                continue
            try:
                raw = await loop.run_in_executor(
                    None,
                    lambda e=engine: e.perform_search(query, num_results=num_results),
                )
            except Exception as exc:
                # Try the next engine instead of failing the whole tool call.
                continue

            results = self._normalize(raw)
            if results:
                formatted = self._format(query, engine_name, results)
                return ToolResult(output=formatted)

        # Every engine returned nothing -> make the failure EXPLICIT (not "no output").
        return ToolResult(
            error=(
                f"Web search for '{query}' returned no results from any engine "
                f"({', '.join(self._ordered_engines())}). The search backend may be "
                f"blocked or rate-limited."
            )
        )

    def _ordered_engines(self) -> List[str]:
        """Configured engine first, then the rest as fallbacks."""
        if config.search_config is not None:
            preferred = config.search_config.engine.lower()
            return [preferred] + [e for e in self._engine_order if e != preferred]
        return list(self._engine_order)

    @staticmethod
    def _normalize(raw) -> List[dict]:
        """Accept either list[str] (URLs) or list[dict] (ddgs) and unify to dicts."""
        out = []
        for item in raw or []:
            if isinstance(item, dict):
                out.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("href") or item.get("url", ""),
                        "body": item.get("body", ""),
                    }
                )
            else:  # bare URL string
                out.append({"title": "", "url": str(item), "body": ""})
        return out

    @staticmethod
    def _format(query: str, engine: str, results: List[dict]) -> str:
        lines = [f"Search results for '{query}' (via {engine}):", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}".rstrip())
            lines.append(f"   {r['url']}")
            if r["body"]:
                lines.append(f"   {r['body']}")
            lines.append("")
        return "\n".join(lines).strip()
