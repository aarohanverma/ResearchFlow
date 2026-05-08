"""Integration-style tests for feed endpoints."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_token(user_id: str | None = None) -> str:
    from app.core.security import create_access_token
    return create_access_token(user_id or str(uuid.uuid4()))


def _auth(user_id: str | None = None) -> dict:
    return {"Authorization": f"Bearer {_make_token(user_id)}"}


@pytest.fixture(scope="module")
def client():
    with (
        patch("app.db.session.create_all_tables", new_callable=AsyncMock),
        patch("app.scheduler.jobs.start_scheduler"),
        patch("app.scheduler.jobs.stop_scheduler"),
    ):
        from main import app
        from app.core.deps import get_db

        async def override_db():
            yield AsyncMock()

        app.dependency_overrides[get_db] = override_db

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

        app.dependency_overrides.clear()


class TestGetFeed:
    def test_requires_auth(self, client):
        resp = client.get("/api/v1/feed?namespace_key=cs.AI")
        assert resp.status_code == 401

    def test_returns_feed_structure(self, client):
        uid = str(uuid.uuid4())
        mock_user = MagicMock()
        mock_user.orientation = MagicMock()
        mock_user.orientation.value = "both"
        mock_user.orientation = __import__("app.models.user", fromlist=["Orientation"]).Orientation.both

        mock_profile = MagicMock()
        mock_profile.hot_subtopics = []
        mock_profile.cold_subtopics = []

        with (
            patch("app.repositories.user.UserRepository.get_by_id", return_value=mock_user),
            patch("app.repositories.user.UserRepository.get_interest_profile", return_value=mock_profile),
            patch("app.services.scoring.ScoringService.score_papers_for_user", return_value=[]),
        ):
            resp = client.get(
                "/api/v1/feed?namespace_key=cs.AI",
                headers=_auth(uid),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "papers" in body
        assert "total" in body
        assert body["namespace_key"] == "cs.AI"

    def test_requires_namespace_key(self, client):
        resp = client.get("/api/v1/feed", headers=_auth())
        assert resp.status_code == 422


class TestSubmitFeedback:
    def test_requires_auth(self, client):
        resp = client.post("/api/v1/feed/feedback", json={
            "paper_id": str(uuid.uuid4()),
            "signal": "like",
        })
        assert resp.status_code == 401

    def test_valid_feedback_returns_204(self, client):
        with (
            patch("app.repositories.paper.PaperRepository.add_feedback", new_callable=AsyncMock),
            patch("app.repositories.paper.PaperRepository.get_by_id", new_callable=AsyncMock, return_value=None),
        ):
            resp = client.post(
                "/api/v1/feed/feedback",
                json={"paper_id": str(uuid.uuid4()), "signal": "like"},
                headers=_auth(),
            )
        assert resp.status_code == 204

    def test_invalid_signal_rejected(self, client):
        resp = client.post(
            "/api/v1/feed/feedback",
            json={"paper_id": str(uuid.uuid4()), "signal": "invalid_signal"},
            headers=_auth(),
        )
        assert resp.status_code == 422

    def test_valid_signals_accepted(self, client):
        for signal in ("like", "dismiss", "more_like_this"):
            with (
                patch("app.repositories.paper.PaperRepository.add_feedback", new_callable=AsyncMock),
                patch("app.repositories.paper.PaperRepository.get_by_id", new_callable=AsyncMock, return_value=None),
            ):
                resp = client.post(
                    "/api/v1/feed/feedback",
                    json={"paper_id": str(uuid.uuid4()), "signal": signal},
                    headers=_auth(),
                )
            assert resp.status_code == 204, f"Signal '{signal}' should be accepted"


class TestRefreshFeed:
    def test_requires_auth(self, client):
        resp = client.post("/api/v1/feed/refresh?namespace_key=cs.AI")
        assert resp.status_code == 401

    def test_returns_triggered_true(self, client):
        mock_mappings = [MagicMock()]

        with patch("app.repositories.graph.GraphRepository.get_source_mappings", return_value=mock_mappings):
            resp = client.post(
                "/api/v1/feed/refresh?namespace_key=cs.AI",
                headers=_auth(),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["triggered"] is True
        assert body["namespace_key"] == "cs.AI"

    def test_requires_namespace_key(self, client):
        resp = client.post("/api/v1/feed/refresh", headers=_auth())
        assert resp.status_code == 422


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
