"""Research providers for podcast topic gathering."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable

from ..state import SourceInfo

logger = logging.getLogger(__name__)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _http_get_json(url: str, headers: dict[str, str] | None = None, timeout: int = 20) -> object:
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_post_json(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: int = 20,
) -> object:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _truncate(text: str, max_chars: int = 1200) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rsplit(" ", 1)[0] + "..."


def search_wikipedia(topic: str, max_results: int) -> list[SourceInfo]:
    """Fetch intro summaries from Wikipedia."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "format": "json",
            "srlimit": max_results,
            "utf8": 1,
        }
    )
    search_url = f"https://en.wikipedia.org/w/api.php?{params}"
    payload = _http_get_json(
        search_url,
        headers={"User-Agent": "PodcastGen-Agent/0.2 (research bot)"},
    )
    search_hits = payload.get("query", {}).get("search", [])
    if not search_hits:
        return []

    titles = [hit["title"] for hit in search_hits[:max_results]]
    extract_params = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 1,
            "titles": "|".join(titles),
            "format": "json",
            "utf8": 1,
        }
    )
    extract_url = f"https://en.wikipedia.org/w/api.php?{extract_params}"
    extract_payload = _http_get_json(
        extract_url,
        headers={"User-Agent": "PodcastGen-Agent/0.2 (research bot)"},
    )
    pages = extract_payload.get("query", {}).get("pages", {})

    sources: list[SourceInfo] = []
    for page in pages.values():
        title = page.get("title", "")
        body = _truncate(page.get("extract", ""))
        if not title or not body:
            continue
        encoded_title = urllib.parse.quote(title.replace(" ", "_"))
        sources.append(
            SourceInfo(
                title=f"Wikipedia: {title}",
                body=body,
                url=f"https://en.wikipedia.org/wiki/{encoded_title}",
            )
        )
    return sources


def search_arxiv(topic: str, max_results: int) -> list[SourceInfo]:
    """Fetch recent arXiv paper summaries for technical topics."""
    query = urllib.parse.quote(f"all:{topic}")
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={query}&start=0&max_results={max_results}&sortBy=relevance"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PodcastGen-Agent/0.2 (research bot)"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        xml_text = response.read().decode("utf-8")

    root = ET.fromstring(xml_text)
    sources: list[SourceInfo] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
        link = ""
        for link_node in entry.findall("atom:link", ATOM_NS):
            if link_node.attrib.get("rel") == "alternate":
                link = link_node.attrib.get("href", "")
                break
        if title and summary:
            sources.append(
                SourceInfo(
                    title=f"arXiv: {title}",
                    body=_truncate(summary),
                    url=link,
                )
            )
    return sources


def search_duckduckgo(topic: str, max_results: int) -> list[SourceInfo]:
    """Fallback web search through DuckDuckGo."""
    from duckduckgo_search import DDGS

    time.sleep(1.5)
    ddgs = DDGS()
    results = list(ddgs.text(topic, max_results=max_results))
    sources: list[SourceInfo] = []
    for result in results:
        title = result.get("title", "")
        body = _truncate(result.get("body", ""))
        url = result.get("href") or result.get("link") or ""
        if title or body:
            sources.append(SourceInfo(title=title, body=body, url=url))
    return sources


def search_tavily(topic: str, max_results: int, api_key: str) -> list[SourceInfo]:
    """Search with Tavily (requires API key)."""
    payload = _http_post_json(
        "https://api.tavily.com/search",
        {
            "api_key": api_key,
            "query": topic,
            "max_results": max_results,
            "include_answer": False,
            "search_depth": "basic",
        },
    )
    sources: list[SourceInfo] = []
    for result in payload.get("results", []):
        title = result.get("title", "")
        body = _truncate(result.get("content", ""))
        url = result.get("url", "")
        if title or body:
            sources.append(SourceInfo(title=title, body=body, url=url))
    return sources


def search_brave(topic: str, max_results: int, api_key: str) -> list[SourceInfo]:
    """Search with Brave Search API (requires API key)."""
    params = urllib.parse.urlencode({"q": topic, "count": max_results})
    url = f"https://api.search.brave.com/res/v1/web/search?{params}"
    payload = _http_get_json(
        url,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
    )
    sources: list[SourceInfo] = []
    for result in payload.get("web", {}).get("results", []):
        title = result.get("title", "")
        body = _truncate(result.get("description", ""))
        url = result.get("url", "")
        if title or body:
            sources.append(SourceInfo(title=title, body=body, url=url))
    return sources


def _dedupe_sources(sources: list[SourceInfo]) -> list[SourceInfo]:
    seen: set[tuple[str, str]] = set()
    unique: list[SourceInfo] = []
    for source in sources:
        key = (source["title"].strip().lower(), source["body"][:120].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def format_research_text(topic: str, sources: list[SourceInfo]) -> str:
    if not sources:
        return f"Topic: {topic}\nNo additional research available."

    lines = [f"Topic: {topic}", "", "Key Information:"]
    for index, source in enumerate(sources, 1):
        lines.append(f"\n{index}. {source['title']}")
        lines.append(source["body"])
        if source.get("url"):
            lines.append(f"Source: {source['url']}")
    return "\n".join(lines)


def gather_research(
    topic: str,
    *,
    max_results: int,
    providers: list[str],
    tavily_api_key: str = "",
    brave_api_key: str = "",
) -> tuple[str, list[SourceInfo]]:
    """Run configured research providers and merge their results."""
    provider_fns: dict[str, Callable[[], list[SourceInfo]]] = {
        "wikipedia": lambda: search_wikipedia(topic, max_results),
        "arxiv": lambda: search_arxiv(topic, max_results),
        "duckduckgo": lambda: search_duckduckgo(topic, max_results),
    }
    if tavily_api_key:
        provider_fns["tavily"] = lambda: search_tavily(topic, max_results, tavily_api_key)
    if brave_api_key:
        provider_fns["brave"] = lambda: search_brave(topic, max_results, brave_api_key)

    collected: list[SourceInfo] = []
    for provider in providers:
        search_fn = provider_fns.get(provider)
        if search_fn is None:
            continue
        try:
            results = search_fn()
            logger.info("Research provider %s returned %d result(s)", provider, len(results))
            collected.extend(results)
        except Exception as exc:
            logger.warning("Research provider %s failed: %s", provider, exc)

    sources = _dedupe_sources(collected)[:max_results]
    return format_research_text(topic, sources), sources
