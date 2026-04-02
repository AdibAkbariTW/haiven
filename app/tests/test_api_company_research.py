# © 2024 Thoughtworks, Inc. | Licensed under the Apache License, Version 2.0  | See LICENSE.md file for permissions.
import unittest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from api.api_company_research import ApiCompanyResearch
from llms.model_config import ModelConfig


class TestApiCompanyResearch(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.add_middleware(SessionMiddleware, secret_key="some-random-string")
        self.client = TestClient(self.app)

        self.mock_chat_session = MagicMock()
        self.mock_chat_session.run.return_value = iter(
            [
                "some research response",
                {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 200,
                        "total_tokens": 300,
                    }
                },
            ]
        )

        self.mock_chat_manager = MagicMock()
        self.mock_chat_manager.json_chat.return_value = (
            "some_key",
            self.mock_chat_session,
        )

        self.mock_prompt_list = MagicMock()
        self.mock_prompt_list.render_prompt.return_value = (
            "rendered company overview prompt",
            None,
        )

        self.mock_model_config = MagicMock()

    def test_should_return_streaming_response_when_valid_company_research_request(self):
        ApiCompanyResearch(
            self.app,
            self.mock_chat_manager,
            self.mock_model_config,
            self.mock_prompt_list,
        )

        response = self.client.post(
            "/api/research",
            json={"userinput": "Thoughtworks", "config": "company"},
        )

        assert response.status_code == 200
        streamed_content = response.content.decode("utf-8")
        assert "some research response" in streamed_content

    def test_should_use_company_overview_prompt_when_config_is_company(self):
        ApiCompanyResearch(
            self.app,
            self.mock_chat_manager,
            self.mock_model_config,
            self.mock_prompt_list,
        )

        self.client.post(
            "/api/research",
            json={"userinput": "Thoughtworks", "config": "company"},
        )

        self.mock_prompt_list.render_prompt.assert_called_with(
            prompt_choice="company-overview",
            user_input="Thoughtworks",
        )

    def test_should_use_ai_tool_prompt_when_config_is_ai_tool(self):
        self.mock_chat_session.run.return_value = iter(["ai tool research response"])

        ApiCompanyResearch(
            self.app,
            self.mock_chat_manager,
            self.mock_model_config,
            self.mock_prompt_list,
        )

        self.client.post(
            "/api/research",
            json={"userinput": "some ai tool", "config": "ai-tool"},
        )

        self.mock_prompt_list.render_prompt.assert_called_with(
            prompt_choice="company-overview-ai-tool",
            user_input="some ai tool",
        )

    def test_should_use_default_company_overview_prompt_when_config_is_unknown(self):
        self.mock_chat_session.run.return_value = iter(["default response"])

        ApiCompanyResearch(
            self.app,
            self.mock_chat_manager,
            self.mock_model_config,
            self.mock_prompt_list,
        )

        self.client.post(
            "/api/research",
            json={"userinput": "some company", "config": "unknown-config"},
        )

        self.mock_prompt_list.render_prompt.assert_called_with(
            prompt_choice="company-overview",
            user_input="some company",
        )

    def test_should_use_perplexity_model_config_when_calling_research_endpoint(self):
        ApiCompanyResearch(
            self.app,
            self.mock_chat_manager,
            self.mock_model_config,
            self.mock_prompt_list,
        )

        self.client.post(
            "/api/research",
            json={"userinput": "Thoughtworks"},
        )

        self.mock_chat_manager.json_chat.assert_called_once()
        actual_model_config = self.mock_chat_manager.json_chat.call_args[1][
            "model_config"
        ]
        expected_model_config = ModelConfig("perplexity", "perplexity", "Perplexity")
        assert actual_model_config.provider == expected_model_config.provider

    def test_should_return_error_in_stream_when_llm_raises_exception(self):
        self.mock_chat_session.run.side_effect = Exception("Model not available")

        ApiCompanyResearch(
            self.app,
            self.mock_chat_manager,
            self.mock_model_config,
            self.mock_prompt_list,
        )

        response = self.client.post(
            "/api/research",
            json={"userinput": "Thoughtworks"},
        )

        assert response.status_code == 200
        streamed_content = response.content.decode("utf-8")
        assert "[ERROR]" in streamed_content

    def test_should_return_streaming_response_when_userinput_is_empty(self):
        self.mock_chat_session.run.return_value = iter(["response for empty input"])

        ApiCompanyResearch(
            self.app,
            self.mock_chat_manager,
            self.mock_model_config,
            self.mock_prompt_list,
        )

        response = self.client.post(
            "/api/research",
            json={"userinput": ""},
        )

        assert response.status_code == 200
        self.mock_prompt_list.render_prompt.assert_called_with(
            prompt_choice="company-overview",
            user_input="",
        )
