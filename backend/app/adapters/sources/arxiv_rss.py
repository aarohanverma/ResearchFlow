"""arXiv RSS source — default ingestion transport.

Rate-limited: 3-second delay between requests.
Exponential backoff with jitter on 503.
Never re-requests a category more than once per run.

Weekend fallback: arXiv RSS is empty Sat/Sun (no daily announcements).
When RSS returns 0 entries, falls back to the arXiv export API which
returns recent papers for a category regardless of day.
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
_API_BASE = "https://export.arxiv.org/api/query"
_DELAY_BETWEEN_REQUESTS = 3.0   # seconds — stay well below arXiv's rate limit


class ArXivRssSource(BaseSource):
    """Ingestion source that reads arXiv papers from the public RSS feed.

    Applies a 3-second inter-request delay and exponential-backoff retry
    (up to 3 attempts) on HTTP 503 errors. Each category is fetched at most
    once per instance lifetime.

    Weekend fallback: when RSS returns 0 entries (Sat/Sun), automatically
    falls back to the arXiv export API to fetch the most recent 50 papers.
    """

    source_name = "arxiv_rss"

    def __init__(self) -> None:
        """Initialise the source with an empty set of fetched category keys."""
        self._fetched: set[str] = set()

    async def fetch(self, external_category_key: str) -> list[RawPaper]:
        """Fetch new arXiv papers for the given category key.

        Tries the daily RSS feed first. If RSS returns zero entries (typical
        on Saturday/Sunday when arXiv makes no announcements), falls back to
        the arXiv export API which always returns recent submissions.

        Each category key is fetched at most once per instance lifetime —
        a second call for the same key returns an empty list immediately.

        Args:
            external_category_key: arXiv category identifier (e.g. ``"cs.AI"``).

        Returns:
            List of normalised ``RawPaper`` objects. Empty when the category
            was already fetched or when all fetch attempts fail.
        """
        if external_category_key in self._fetched:
            return []

        # Mark fetched BEFORE the attempt so a second concurrent call for the
        # same category is short-circuited immediately.  Both _fetch_rss and
        # _fetch_api catch all exceptions and return [], so the only "failure"
        # scenario is an empty result — which still correctly triggers the
        # weekend fallback below.
        self._fetched.add(external_category_key)

        papers = await self._fetch_rss(external_category_key)

        if not papers:
            log.info(
                "arxiv_rss: RSS empty for %s (weekend?), falling back to export API",
                external_category_key,
            )
            papers = await self._fetch_api(external_category_key)

        log.info(
            "arxiv_rss: final count category=%s count=%d",
            external_category_key, len(papers),
        )
        return papers

    async def _fetch_rss(self, category: str) -> list[RawPaper]:
        """Fetch papers from the arXiv daily RSS feed.

        Applies a 3-second pre-request delay to stay well below arXiv's rate
        limit. Retries up to 3 times with exponential-jitter backoff on HTTP
        503 errors. Returns an empty list on any non-retryable failure.

        Args:
            category: arXiv category key (e.g. ``"cs.LG"``).

        Returns:
            Parsed ``RawPaper`` list, or empty list when the feed returns no
            entries (common on weekends) or when a network error occurs.
        """
        url = f"{_RSS_BASE}{category}"
        log.info("arxiv_rss: fetching RSS category=%s", category)

        await asyncio.sleep(_DELAY_BETWEEN_REQUESTS)

        try:
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
        except Exception as exc:
            log.warning("arxiv_rss: RSS fetch failed category=%s err=%s", category, exc)
            return []

        feed = feedparser.parse(raw_xml)
        papers: list[RawPaper] = []

        for entry in feed.entries:
            arxiv_id = self._extract_arxiv_id(entry.get("id", ""))
            if not arxiv_id:
                continue
            authors = [a.name for a in entry.get("authors", [])] or [entry.get("author", "Unknown")]
            papers.append(RawPaper(
                external_id=arxiv_id,
                title=entry.get("title", "").strip(),
                authors=authors,
                abstract=entry.get("summary", "").strip(),
                source_url=entry.get("link", ""),
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                published_at=self._parse_date(entry.get("published")),
                namespace_key=category,
            ))

        log.info("arxiv_rss: RSS fetched category=%s count=%d", category, len(papers))
        return papers

    async def _fetch_api(self, category: str, max_results: int = 50) -> list[RawPaper]:
        """Fetch recent papers from the arXiv Atom export API.

        Used as a fallback when the daily RSS feed is empty (weekends). The
        export API returns the most recent submissions for a category regardless
        of the day of the week.

        Args:
            category: arXiv category key (e.g. ``"cs.AI"``).
            max_results: Maximum number of papers to retrieve.

        Returns:
            Parsed ``RawPaper`` list, or empty list on network failure.
        """
        params = {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": "0",
            "max_results": str(max_results),
        }
        log.info("arxiv_rss: fetching export API category=%s max=%d", category, max_results)

        await asyncio.sleep(_DELAY_BETWEEN_REQUESTS)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(_API_BASE, params=params)
                resp.raise_for_status()
                raw_xml = resp.text
        except Exception as exc:
            log.warning("arxiv_rss: export API fetch failed category=%s err=%s", category, exc)
            return []

        feed = feedparser.parse(raw_xml)
        papers: list[RawPaper] = []

        for entry in feed.entries:
            arxiv_id = self._extract_arxiv_id(entry.get("id", ""))
            if not arxiv_id:
                continue
            authors = [a.get("name", "Unknown") for a in entry.get("authors", [])]
            if not authors:
                authors = ["Unknown"]
            # Atom summary contains the abstract
            abstract = entry.get("summary", "").strip()
            # Published date in Atom is ISO 8601
            published = self._parse_iso_date(entry.get("published", ""))
            papers.append(RawPaper(
                external_id=arxiv_id,
                title=re.sub(r"\s+", " ", entry.get("title", "")).strip(),
                authors=authors,
                abstract=abstract,
                source_url=entry.get("link", f"https://arxiv.org/abs/{arxiv_id}"),
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                published_at=published,
                namespace_key=category,
            ))

        log.info("arxiv_rss: export API fetched category=%s count=%d", category, len(papers))
        return papers

    def _extract_arxiv_id(self, url: str) -> str | None:
        """Extract a bare arXiv ID from a URL or raw ID string.

        Strips version suffixes (e.g. ``v3``) so only the canonical ID is
        returned (e.g. ``"2301.07041"`` from
        ``"https://arxiv.org/abs/2301.07041v3"``).

        Args:
            url: A string containing an arXiv ID, possibly embedded in a URL.

        Returns:
            The bare arXiv ID (``YYMM.NNNNN``), or ``None`` if not found.
        """
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?$", url)
        return m.group(1) if m else None

    def _parse_date(self, date_str: str | None) -> datetime | None:
        """Parse an RFC 2822 date string (RSS ``<pubDate>`` format) to UTC datetime.

        Args:
            date_str: Date string in RFC 2822 format, or ``None``.

        Returns:
            Timezone-aware UTC ``datetime``, or ``None`` on parse failure.
        """
        if not date_str:
            return None
        try:
            import email.utils
            parsed = email.utils.parsedate_to_datetime(date_str)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    def _parse_iso_date(self, date_str: str | None) -> datetime | None:
        """Parse an ISO 8601 date string (Atom ``<published>`` format) to UTC datetime.

        Args:
            date_str: Date string in ISO 8601 format, or ``None``.

        Returns:
            Timezone-aware UTC ``datetime``, or ``None`` on parse failure.
        """
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None
