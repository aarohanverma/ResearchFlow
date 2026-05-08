"""Unit tests for ArXivRssSource — RSS parsing and pure-Python helpers."""

from datetime import timezone

import pytest

from app.adapters.sources.arxiv_rss import ArXivRssSource

SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>cs.AI updates on arXiv.org</title>
    <item>
      <title>Test Paper One</title>
      <id>http://arxiv.org/abs/2401.00001v1</id>
      <link>http://arxiv.org/abs/2401.00001v1</link>
      <summary>This is the abstract for paper one about deep learning.</summary>
      <author>Smith, Alice; Jones, Bob</author>
      <published>Mon, 01 Jan 2024 00:00:00 GMT</published>
    </item>
    <item>
      <title>Test Paper Two</title>
      <id>http://arxiv.org/abs/2401.00002v2</id>
      <link>http://arxiv.org/abs/2401.00002v2</link>
      <summary>Abstract for paper two about reinforcement learning.</summary>
      <author>Doe, Jane</author>
      <published>Tue, 02 Jan 2024 00:00:00 GMT</published>
    </item>
  </channel>
</rss>"""


class TestExtractArxivId:
    def _src(self):
        return ArXivRssSource()

    def test_standard_id(self):
        src = self._src()
        assert src._extract_arxiv_id("http://arxiv.org/abs/2401.00001v1") == "2401.00001"

    def test_id_with_5_digits(self):
        src = self._src()
        assert src._extract_arxiv_id("http://arxiv.org/abs/2401.12345v3") == "2401.12345"

    def test_no_version_suffix(self):
        src = self._src()
        assert src._extract_arxiv_id("http://arxiv.org/abs/2401.00001") == "2401.00001"

    def test_invalid_url_returns_none(self):
        src = self._src()
        assert src._extract_arxiv_id("http://example.com/not-arxiv") is None

    def test_empty_string_returns_none(self):
        src = self._src()
        assert src._extract_arxiv_id("") is None

    def test_strips_version_from_higher_number(self):
        src = self._src()
        assert src._extract_arxiv_id("https://arxiv.org/abs/2312.99999v10") == "2312.99999"


class TestParseDate:
    def _src(self):
        return ArXivRssSource()

    def test_valid_rfc_date(self):
        src = self._src()
        result = src._parse_date("Mon, 01 Jan 2024 00:00:00 GMT")
        assert result is not None
        assert result.tzinfo is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_none_input_returns_none(self):
        src = self._src()
        assert src._parse_date(None) is None

    def test_empty_string_returns_none(self):
        src = self._src()
        assert src._parse_date("") is None

    def test_garbage_string_returns_none(self):
        src = self._src()
        assert src._parse_date("not-a-date") is None

    def test_result_is_utc(self):
        src = self._src()
        result = src._parse_date("Mon, 01 Jan 2024 12:00:00 +0000")
        assert result is not None
        assert result.tzinfo.utcoffset(result).total_seconds() == 0


class TestFetchWithMockedHTTP:
    @pytest.mark.asyncio
    async def test_fetch_parses_papers(self, monkeypatch):
        import httpx
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = SAMPLE_RSS
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        src = ArXivRssSource()

        with patch("app.adapters.sources.arxiv_rss.asyncio.sleep", new_callable=AsyncMock), \
             patch("app.adapters.sources.arxiv_rss.httpx.AsyncClient", return_value=mock_client):
            papers = await src.fetch("cs.AI")

        assert len(papers) == 2
        assert papers[0].external_id == "2401.00001"
        assert papers[0].title == "Test Paper One"
        assert "deep learning" in papers[0].abstract
        assert papers[1].external_id == "2401.00002"

    @pytest.mark.asyncio
    async def test_no_double_fetch_same_category(self, monkeypatch):
        """Source tracks fetched categories and skips duplicates in the same run."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_RSS
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        src = ArXivRssSource()

        with patch("app.adapters.sources.arxiv_rss.asyncio.sleep", new_callable=AsyncMock), \
             patch("app.adapters.sources.arxiv_rss.httpx.AsyncClient", return_value=mock_client):
            first = await src.fetch("cs.AI")
            second = await src.fetch("cs.AI")

        assert len(first) == 2
        assert second == []  # de-duplicated
        assert mock_client.get.call_count == 1  # HTTP only called once

    @pytest.mark.asyncio
    async def test_pdf_url_constructed_from_id(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_RSS
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        src = ArXivRssSource()

        with patch("app.adapters.sources.arxiv_rss.asyncio.sleep", new_callable=AsyncMock), \
             patch("app.adapters.sources.arxiv_rss.httpx.AsyncClient", return_value=mock_client):
            papers = await src.fetch("cs.AI")

        assert papers[0].pdf_url == "https://arxiv.org/pdf/2401.00001.pdf"
