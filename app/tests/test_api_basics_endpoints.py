# © 2024 Thoughtworks, Inc. | Licensed under the Apache License, Version 2.0  | See LICENSE.md file for permissions.
import io
import json
import unittest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from api.api_basics import ApiBasics


def _make_api_basics(app, chat_manager=None, prompts_chat=None, prompts_guided=None):
    """Helper to construct ApiBasics with sensible MagicMock defaults."""
    return ApiBasics(
        app,
        chat_manager=chat_manager or MagicMock(),
        model_config=MagicMock(),
        prompts_guided=prompts_guided or MagicMock(),
        knowledge_manager=MagicMock(),
        prompts_chat=prompts_chat or MagicMock(),
        image_service=MagicMock(),
        config_service=MagicMock(),
        disclaimer_and_guidelines=MagicMock(),
        inspirations_manager=MagicMock(),
    )


class TestApiBasicsIterateEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.add_middleware(SessionMiddleware, secret_key="some-random-string")
        self.client = TestClient(self.app)

        self.mock_chat_session = MagicMock()
        self.mock_chat_session.run.return_value = iter(
            [
                "iterated response",
                {
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 100,
                        "total_tokens": 150,
                    }
                },
            ]
        )
        self.mock_chat_manager = MagicMock()
        self.mock_chat_manager.json_chat.return_value = (
            "session-key",
            self.mock_chat_session,
        )

    def test_should_return_streaming_response_when_iterate_request_is_valid(self):
        _make_api_basics(self.app, chat_manager=self.mock_chat_manager)

        response = self.client.post(
            "/api/prompt/iterate",
            json={
                "chatSessionId": "session-123",
                "userinput": "make it more concise",
                "scenarios": '[{"id": 1, "title": "London"}]',
            },
        )

        assert response.status_code == 200
        streamed_content = response.content.decode("utf-8")
        assert "iterated response" in streamed_content

    def test_should_pass_existing_session_id_when_iterating(self):
        _make_api_basics(self.app, chat_manager=self.mock_chat_manager)

        self.client.post(
            "/api/prompt/iterate",
            json={
                "chatSessionId": "existing-session-456",
                "userinput": "refine these",
                "scenarios": '[{"id": 2, "title": "Paris"}]',
            },
        )

        call_kwargs = self.mock_chat_manager.json_chat.call_args[1]
        assert call_kwargs["session_id"] == "existing-session-456"

    def test_should_return_400_when_chat_session_id_is_missing(self):
        _make_api_basics(self.app, chat_manager=self.mock_chat_manager)

        response = self.client.post(
            "/api/prompt/iterate",
            json={
                "userinput": "make it better",
                "scenarios": '[{"id": 1}]',
            },
        )

        assert response.status_code == 400
        assert "chatSessionId is required" in response.json()["detail"]


class TestApiBasicsRenderPromptEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.add_middleware(SessionMiddleware, secret_key="some-random-string")
        self.client = TestClient(self.app)

    def test_should_return_rendered_prompt_when_render_request_is_valid(self):
        mock_prompts_chat = MagicMock()
        mock_template = MagicMock()
        mock_template.template = "Analyze {{userinput}} for a company overview."
        mock_prompts_chat.render_prompt.return_value = (
            "Analyze Thoughtworks for a company overview.",
            mock_template,
        )

        _make_api_basics(self.app, prompts_chat=mock_prompts_chat)

        response = self.client.post(
            "/api/prompt/render",
            json={"promptid": "some-prompt-id", "userinput": "Thoughtworks"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["prompt"] == "Analyze Thoughtworks for a company overview."
        assert data["template"] == "Analyze {{userinput}} for a company overview."

    def test_should_render_guided_prompt_when_promptid_starts_with_guided(self):
        mock_prompts_guided = MagicMock()
        mock_template = MagicMock()
        mock_template.template = "Guided template {{userinput}}"
        mock_prompts_guided.render_prompt.return_value = (
            "Guided rendered output",
            mock_template,
        )

        _make_api_basics(self.app, prompts_guided=mock_prompts_guided)

        response = self.client.post(
            "/api/prompt/render",
            json={"promptid": "guided-requirements", "userinput": "some input"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["prompt"] == "Guided rendered output"
        mock_prompts_guided.render_prompt.assert_called_with(
            prompt_choice="guided-requirements",
            user_input="some input",
        )

    def test_should_return_500_when_promptid_is_missing(self):
        _make_api_basics(self.app)

        response = self.client.post(
            "/api/prompt/render",
            json={"userinput": "some input"},
        )

        assert response.status_code == 500
        assert "promptid is required" in response.json()["detail"]


class TestApiBasicsImageEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.add_middleware(SessionMiddleware, secret_key="some-random-string")
        self.client = TestClient(self.app)

        self.mock_image_service = MagicMock()
        self.mock_image_service.prompt_with_image.return_value = iter(
            ["A photo showing a mountain landscape."]
        )

    def test_should_return_streaming_response_when_valid_image_is_uploaded(self):
        with patch("api.api_basics.Image") as mock_pil_image:
            mock_pil_image.open.return_value = MagicMock()

            api = ApiBasics(
                self.app,
                chat_manager=MagicMock(),
                model_config=MagicMock(),
                prompts_guided=MagicMock(),
                knowledge_manager=MagicMock(),
                prompts_chat=MagicMock(),
                image_service=self.mock_image_service,
                config_service=MagicMock(),
                disclaimer_and_guidelines=MagicMock(),
                inspirations_manager=MagicMock(),
            )

            image_bytes = io.BytesIO(b"fake-image-bytes")
            response = self.client.post(
                "/api/prompt/image",
                data={"prompt": "What is in this image?"},
                files={"file": ("test.png", image_bytes, "image/png")},
            )

        assert response.status_code == 200
        streamed_content = response.content.decode("utf-8")
        assert "mountain landscape" in streamed_content
        self.mock_image_service.prompt_with_image.assert_called_once()


class TestApiBasicsDisclaimerEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.add_middleware(SessionMiddleware, secret_key="some-random-string")
        self.client = TestClient(self.app)

    def _make_api_with_disclaimer(self, mock_disclaimer):
        return ApiBasics(
            self.app,
            chat_manager=MagicMock(),
            model_config=MagicMock(),
            prompts_guided=MagicMock(),
            knowledge_manager=MagicMock(),
            prompts_chat=MagicMock(),
            image_service=MagicMock(),
            config_service=MagicMock(),
            disclaimer_and_guidelines=mock_disclaimer,
            inspirations_manager=MagicMock(),
        )

    def test_should_return_disclaimer_content_when_disclaimer_is_enabled(self):
        mock_disclaimer = MagicMock()
        mock_disclaimer.is_enabled = True
        mock_disclaimer.fetch_disclaimer_and_guidelines.return_value = json.dumps(
            {"title": "Usage Guidelines", "content": "Please use responsibly."}
        )
        self._make_api_with_disclaimer(mock_disclaimer)

        response = self.client.get("/api/disclaimer-guidelines")

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True
        assert data["title"] == "Usage Guidelines"
        assert data["content"] == "Please use responsibly."

    def test_should_return_disabled_response_when_disclaimer_is_not_enabled(self):
        mock_disclaimer = MagicMock()
        mock_disclaimer.is_enabled = False
        self._make_api_with_disclaimer(mock_disclaimer)

        response = self.client.get("/api/disclaimer-guidelines")

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False
        assert data["title"] == ""
        assert data["content"] == ""


class TestApiBasicsKnowledgeSnippetsEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.add_middleware(SessionMiddleware, secret_key="some-random-string")
        self.client = TestClient(self.app)

    def _make_api_with_knowledge_manager(self, mock_knowledge_manager):
        return ApiBasics(
            self.app,
            chat_manager=MagicMock(),
            model_config=MagicMock(),
            prompts_guided=MagicMock(),
            knowledge_manager=mock_knowledge_manager,
            prompts_chat=MagicMock(),
            image_service=MagicMock(),
            config_service=MagicMock(),
            disclaimer_and_guidelines=MagicMock(),
            inspirations_manager=MagicMock(),
        )

    def test_should_return_knowledge_snippets_when_contexts_exist(self):
        mock_knowledge_manager = MagicMock()
        mock_context = MagicMock()
        mock_context.metadata = {"title": "Architecture Context"}
        mock_context.content = "Architecture decision records and guidelines."
        mock_knowledge_manager.knowledge_base_markdown.get_all_contexts.return_value = {
            "architecture": mock_context
        }
        self._make_api_with_knowledge_manager(mock_knowledge_manager)

        response = self.client.get("/api/knowledge/snippets")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["context"] == "architecture"
        assert data[0]["title"] == "Architecture Context"
        assert data[0]["snippets"]["context"] == "Architecture decision records and guidelines."

    def test_should_return_empty_list_when_no_contexts_exist(self):
        mock_knowledge_manager = MagicMock()
        mock_knowledge_manager.knowledge_base_markdown.get_all_contexts.return_value = {}
        self._make_api_with_knowledge_manager(mock_knowledge_manager)

        response = self.client.get("/api/knowledge/snippets")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_should_return_snippets_sorted_by_context_key(self):
        mock_knowledge_manager = MagicMock()

        def make_context(title, content):
            ctx = MagicMock()
            ctx.metadata = {"title": title}
            ctx.content = content
            return ctx

        mock_knowledge_manager.knowledge_base_markdown.get_all_contexts.return_value = {
            "zebra": make_context("Zebra Context", "z content"),
            "alpha": make_context("Alpha Context", "a content"),
            "middle": make_context("Middle Context", "m content"),
        }
        self._make_api_with_knowledge_manager(mock_knowledge_manager)

        response = self.client.get("/api/knowledge/snippets")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["context"] == "alpha"
        assert data[1]["context"] == "middle"
        assert data[2]["context"] == "zebra"
