# © 2024 Thoughtworks, Inc. | Licensed under the Apache License, Version 2.0  | See LICENSE.md file for permissions.
import json
import unittest
from unittest.mock import MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from api.api_key_management import ApiKeyManagementAPI


class TestApiKeyManagementAPI(ApiKeyManagementAPI):
    """Subclass that overrides get_user_email to avoid requiring a real session."""

    def __init__(self, app, api_key_service, config_service, user_email):
        self._test_user_email = user_email
        super().__init__(app, api_key_service, config_service)

    def get_user_email(self, request):
        return self._test_user_email


class TestApiKeyManagementEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.add_middleware(SessionMiddleware, secret_key="some-random-string")
        self.client = TestClient(self.app)
        self.user_email = "testuser@example.com"
        self.mock_config = MagicMock()

    def _make_api(self, mock_service, user_email=None):
        TestApiKeyManagementAPI(
            self.app,
            mock_service,
            self.mock_config,
            user_email or self.user_email,
        )

    def test_should_return_list_of_keys_when_user_has_api_keys(self):
        mock_service = MagicMock()
        mock_service.list_keys_for_user.return_value = {
            "abc123hash": {
                "name": "my-key",
                "user_id": "pseudonymized_user_id",
                "created_at": "2024-01-01T00:00:00",
                "expires_at": "2024-02-01T00:00:00",
                "last_used": None,
                "usage_count": 3,
            }
        }
        self._make_api(mock_service)

        response = self.client.get("/api/apikeys")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["keys"]) == 1
        assert data["keys"][0]["key_hash"] == "abc123hash"
        assert data["keys"][0]["name"] == "my-key"
        mock_service.list_keys_for_user.assert_called_with(self.user_email)

    def test_should_return_empty_list_when_user_has_no_api_keys(self):
        mock_service = MagicMock()
        mock_service.list_keys_for_user.return_value = {}
        self._make_api(mock_service)

        response = self.client.get("/api/apikeys")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["keys"] == []

    def test_should_generate_api_key_and_return_key_info_when_request_is_valid(self):
        mock_service = MagicMock()
        mock_service.generate_api_key.return_value = "generated_api_key_value_xyz"
        self._make_api(mock_service)

        response = self.client.post(
            "/api/apikeys/generate",
            json={"name": "my-integration-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["api_key"] == "generated_api_key_value_xyz"
        assert data["name"] == "my-integration-key"
        assert data["expires_days"] == 30
        mock_service.generate_api_key.assert_called_with(
            name="my-integration-key",
            user_id=self.user_email,
            expires_days=30,
        )

    def test_should_generate_api_key_with_custom_expiry_when_expires_days_is_valid(
        self,
    ):
        mock_service = MagicMock()
        mock_service.generate_api_key.return_value = "custom_expiry_key"
        self._make_api(mock_service)

        response = self.client.post(
            "/api/apikeys/generate",
            json={"name": "short-lived-key", "expires_days": 7},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["expires_days"] == 7
        mock_service.generate_api_key.assert_called_with(
            name="short-lived-key",
            user_id=self.user_email,
            expires_days=7,
        )

    def test_should_return_400_when_expires_days_exceeds_maximum(self):
        mock_service = MagicMock()
        self._make_api(mock_service)

        response = self.client.post(
            "/api/apikeys/generate",
            json={"name": "long-key", "expires_days": 31},
        )

        assert response.status_code == 400
        assert "maximum expiry is 30 days" in response.json()["detail"]
        mock_service.generate_api_key.assert_not_called()

    def test_should_return_400_when_expires_days_is_less_than_one(self):
        mock_service = MagicMock()
        self._make_api(mock_service)

        response = self.client.post(
            "/api/apikeys/generate",
            json={"name": "zero-expiry-key", "expires_days": 0},
        )

        assert response.status_code == 400
        assert "expiry must be at least 1 day" in response.json()["detail"]
        mock_service.generate_api_key.assert_not_called()

    def test_should_revoke_api_key_when_key_belongs_to_user(self):
        key_hash = "key_to_revoke_hash"
        mock_service = MagicMock()
        mock_service.list_keys_for_user.return_value = {
            key_hash: {"name": "my-key", "user_id": "pseudonymized_user_id"}
        }
        mock_service.revoke_key.return_value = True
        self._make_api(mock_service)

        response = self.client.post(
            "/api/apikeys/revoke",
            json={"key_hash": key_hash},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_service.revoke_key.assert_called_with(key_hash)

    def test_should_return_404_when_revoking_key_not_owned_by_user(self):
        mock_service = MagicMock()
        mock_service.list_keys_for_user.return_value = {}
        self._make_api(mock_service)

        response = self.client.post(
            "/api/apikeys/revoke",
            json={"key_hash": "nonexistent_hash"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_should_return_usage_statistics_when_user_has_keys(self):
        mock_service = MagicMock()
        mock_service.list_keys_for_user.return_value = {
            "hash1": {
                "name": "key-one",
                "user_id": "user",
                "created_at": "2024-01-01T00:00:00",
                "expires_at": "2024-02-01T00:00:00",
                "last_used": "2024-01-15T10:00:00",
                "usage_count": 10,
            },
            "hash2": {
                "name": "key-two",
                "user_id": "user",
                "created_at": "2024-01-02T00:00:00",
                "expires_at": "2024-02-02T00:00:00",
                "last_used": None,
                "usage_count": 5,
            },
        }
        self._make_api(mock_service)

        response = self.client.get("/api/apikeys/usage")

        assert response.status_code == 200
        data = response.json()
        assert data["total_keys"] == 2
        assert data["total_usage"] == 15
        mock_service.list_keys_for_user.assert_called_with(self.user_email)

    def test_should_return_401_when_user_is_not_authenticated(self):
        mock_service = MagicMock()

        class UnauthenticatedApiKeyManagementAPI(ApiKeyManagementAPI):
            def get_user_email(self, request):
                raise HTTPException(
                    status_code=401,
                    detail="User not authenticated.",
                )

        UnauthenticatedApiKeyManagementAPI(self.app, mock_service, self.mock_config)

        response = self.client.get("/api/apikeys")

        assert response.status_code == 401
        mock_service.list_keys_for_user.assert_not_called()

    def test_should_return_401_when_unauthenticated_user_tries_to_generate_key(self):
        mock_service = MagicMock()

        class UnauthenticatedApiKeyManagementAPI(ApiKeyManagementAPI):
            def get_user_email(self, request):
                raise HTTPException(
                    status_code=401,
                    detail="User not authenticated. You must be logged in to generate or manage API keys, even in developer mode.",
                )

        UnauthenticatedApiKeyManagementAPI(self.app, mock_service, self.mock_config)

        response = self.client.post(
            "/api/apikeys/generate",
            json={"name": "my-key"},
        )

        assert response.status_code == 401
        assert "User not authenticated" in response.json()["detail"]
        mock_service.generate_api_key.assert_not_called()
