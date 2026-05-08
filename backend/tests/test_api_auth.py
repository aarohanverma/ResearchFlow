"""Integration-style tests for auth endpoints using FastAPI TestClient."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """TestClient with DB and scheduler patched out."""
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


class TestRegisterEndpoint:
    def test_register_success(self, client):
        uid = uuid.uuid4()

        with patch("app.repositories.user.UserRepository.get_by_email", return_value=None), \
             patch("app.repositories.user.UserRepository.create") as mock_create:
            mock_user = MagicMock()
            mock_user.id = uid
            mock_create.return_value = mock_user

            resp = client.post("/api/v1/auth/register", json={
                "email": "new@example.com",
                "password": "StrongPass123!",
                "display_name": "New User",
            })

        assert resp.status_code == 201
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    def test_register_duplicate_email_returns_409(self, client):
        existing = MagicMock()
        existing.id = uuid.uuid4()

        with patch("app.repositories.user.UserRepository.get_by_email", return_value=existing):
            resp = client.post("/api/v1/auth/register", json={
                "email": "existing@example.com",
                "password": "StrongPass123!",
                "display_name": "Dupe User",
            })

        assert resp.status_code == 409

    def test_register_short_password_rejected(self, client):
        resp = client.post("/api/v1/auth/register", json={
            "email": "user@example.com",
            "password": "short",
            "display_name": "User",
        })
        assert resp.status_code == 422

    def test_register_invalid_email_rejected(self, client):
        resp = client.post("/api/v1/auth/register", json={
            "email": "not-an-email",
            "password": "StrongPass123!",
            "display_name": "User",
        })
        assert resp.status_code == 422


class TestLoginEndpoint:
    def test_login_success(self, client):
        uid = uuid.uuid4()
        from app.core.security import hash_password

        mock_user = MagicMock()
        mock_user.id = uid
        mock_user.hashed_password = hash_password("ValidPass123!")

        with patch("app.repositories.user.UserRepository.get_by_email", return_value=mock_user):
            resp = client.post("/api/v1/auth/login", json={
                "email": "user@example.com",
                "password": "ValidPass123!",
            })

        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_login_wrong_password_returns_401(self, client):
        from app.core.security import hash_password

        mock_user = MagicMock()
        mock_user.id = uuid.uuid4()
        mock_user.hashed_password = hash_password("RealPass123!")

        with patch("app.repositories.user.UserRepository.get_by_email", return_value=mock_user):
            resp = client.post("/api/v1/auth/login", json={
                "email": "user@example.com",
                "password": "WrongPassword!",
            })

        assert resp.status_code == 401

    def test_login_unknown_email_returns_401(self, client):
        with patch("app.repositories.user.UserRepository.get_by_email", return_value=None):
            resp = client.post("/api/v1/auth/login", json={
                "email": "nobody@example.com",
                "password": "AnyPass123!",
            })
        assert resp.status_code == 401


class TestMeEndpoint:
    def _auth_header(self) -> dict:
        from app.core.security import create_access_token
        token = create_access_token(str(uuid.uuid4()))
        return {"Authorization": f"Bearer {token}"}

    def test_me_returns_user(self, client):
        mock_user = MagicMock()
        mock_user.id = uuid.uuid4()
        mock_user.email = "me@example.com"
        mock_user.display_name = "Me"
        mock_user.expertise_level = "practitioner"
        mock_user.orientation = "both"
        mock_user.onboarding_complete = True

        with patch("app.repositories.user.UserRepository.get_by_id", return_value=mock_user):
            resp = client.get("/api/v1/auth/me", headers=self._auth_header())

        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == "me@example.com"

    def test_me_without_token_returns_401(self, client):
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    def test_me_with_invalid_token_returns_401(self, client):
        resp = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer garbage"})
        assert resp.status_code == 401
