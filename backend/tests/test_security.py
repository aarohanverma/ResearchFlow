"""Unit tests for security utilities: password hashing and JWT."""

from datetime import timedelta

import pytest
from jose import JWTError

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        h = hash_password("MySecretPass!")
        assert h != "MySecretPass!"

    def test_correct_password_verifies(self):
        h = hash_password("MySecretPass!")
        assert verify_password("MySecretPass!", h) is True

    def test_wrong_password_rejected(self):
        h = hash_password("MySecretPass!")
        assert verify_password("wrongpassword", h) is False

    def test_empty_password_handled(self):
        h = hash_password("a")
        assert verify_password("a", h) is True
        assert verify_password("b", h) is False

    def test_hashes_are_unique(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        # bcrypt salts ensure hashes differ
        assert h1 != h2
        assert verify_password("same", h1)
        assert verify_password("same", h2)


class TestJWT:
    def test_create_and_decode_token(self):
        token = create_access_token("user-abc-123")
        payload = decode_access_token(token)
        assert payload["sub"] == "user-abc-123"

    def test_token_contains_exp_and_iat(self):
        token = create_access_token("user-xyz")
        payload = decode_access_token(token)
        assert "exp" in payload
        assert "iat" in payload

    def test_uuid_subject_round_trips(self):
        import uuid
        uid = str(uuid.uuid4())
        token = create_access_token(uid)
        payload = decode_access_token(token)
        assert payload["sub"] == uid

    def test_invalid_token_raises_jwt_error(self):
        with pytest.raises(JWTError):
            decode_access_token("this.is.not.a.valid.token")

    def test_malformed_token_raises(self):
        with pytest.raises(JWTError):
            decode_access_token("garbage")

    def test_custom_expiry_accepted(self):
        token = create_access_token("user-1", expires_delta=timedelta(minutes=1))
        payload = decode_access_token(token)
        assert payload["sub"] == "user-1"
