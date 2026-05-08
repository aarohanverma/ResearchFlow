"""arXiv RSS source — default ingestion transport.

Rate-limited: 3-second delay between requests.
Exponential backoff with jitter on 503.
Never re-requests a category more than once per run.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone

import feedparser
import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.adapters.sources.base import BaseSource, RawPaper

log = logging.getLogger(__name__)

_RSS_BASE = "https://rss.arxiv.org/rss/"
_DELAY_BETWEEN_REQUESTS = 3.0   # seconds — stay well below arXiv's rate limit


class ArXivRssSource(BaseSource):
    """Ingestion source that reads arXiv papers from the public RSS feed.

    Applies a 3-second inter-request delay and exponential-backoff retry
    (up to 3 attempts) on HTTP 503 errors. Each category is fetched at most
    once per instance lifetime.
    """

    source_name = "arxiv_rss"

    def __init__(self) -> None:
        """Initialise the source with an empty per-run fetch guard."""
        self._fetched: set[str] = set()   # prevent double-fetch in one run

    async def fetch(self, external_category_key: str) -> list[RawPaper]:
        """Fetch recent papers for an arXiv RSS category.

        Skips the fetch if this category was already fetched in the current
        run. Applies a 3-second delay before each HTTP request and retries
        on 503 responses with exponential backoff.

        Args:
            external_category_key: arXiv category string (e.g. ``"cs.AI"``).

        Returns:
            A list of ``RawPaper`` objects parsed from the RSS feed. Returns
            an empty list if the category was already fetched this run.
        """
        if external_category_key in self._fetched:
            return []
        self._fetched.add(external_category_key)

        url = f"{_RSS_BASE}{external_category_key}"
        log.info("arxiv_rss fetching category=%s", external_category_key)

        await asyncio.sleep(_DELAY_BETWEEN_REQUESTS)

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=5, max=60),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(url)
                    if resp.status_code == 503:
                        raise httpx.HTTPStatusError("503", request=resp.request, response=resp)
                    resp.raise_for_status()
                    raw_xml = resp.text

        feed = feedparser.parse(raw_xml)
        papers: list[RawPaper] = []

        for entry in feed.entries:
            arxiv_id = self._extract_arxiv_id(entry.get("id", ""))
            if not arxiv_id:
                continue

            authors = [a.name for a in entry.get("authors", [])] or [
                entry.get("author", "Unknown")
            ]
            published = self._parse_date(entry.get("published"))

            papers.append(RawPaper(
                external_id=arxiv_id,
                title=entry.get("title", "").strip(),
                authors=authors,
                abstract=entry.get("summary", "").strip(),
                source_url=entry.get("link", ""),
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                published_at=published,
                namespace_key=external_category_key,
            ))

        log.info("arxiv_rss fetched category=%s count=%d", external_category_key, len(papers))
        return papers

    def _extract_arxiv_id(self, url: str) -> str | None:
        """Extract the bare arXiv ID (e.g. '2401.12345') from an arXiv URL, stripping version suffix."""
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?$", url)
        return m.group(1) if m else None

    def _parse_date(self, date_str: str | None) -> datetime | None:
        """Parse an RFC 2822 date string to a UTC-aware ``datetime``, returning ``None`` on failure."""
        if not date_str:
            return None
        try:
            import email.utils
            parsed = email.utils.parsedate_to_datetime(date_str)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None
