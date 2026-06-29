from ddgs import DDGS

from app.tool.search.base import WebSearchEngine


class DuckDuckGoSearchEngine(WebSearchEngine):
    def perform_search(self, query, num_results=10, *args, **kwargs):
        """DuckDuckGo search engine (sync). Returns a list of result dicts."""
        with DDGS() as ddgs:
            # returns list[dict] with keys: title, href, body
            return ddgs.text(query, max_results=num_results)
