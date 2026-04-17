# © 2024 Thoughtworks, Inc. | Licensed under the Apache License, Version 2.0  | See LICENSE.md file for permissions.
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from llms.chats import ChatManager, ChatOptions, StreamingChat
from llms.model_config import ModelConfig
from prompts.prompts import PromptList

from logger import HaivenLogger
import json
from auth import auth_util


class PromptRequestBody(BaseModel):
    userinput: Optional[str] = None
    promptid: Optional[str] = None
    chatSessionId: Optional[str] = None
    contexts: Optional[List[str]] = None
    document: Optional[List[str]] = None
    json: bool = False
    userContext: Optional[str] = None


class IterateRequest(PromptRequestBody):
    scenarios: str
    contexts: Optional[List[str]] = None
    user_context: Optional[str] = None


def streaming_media_type() -> str:
    return "text/event-stream"


def streaming_headers(chat_session_key_value=None):
    headers = {
        "Connection": "keep-alive",
        "Content-Encoding": "none",
        "Access-Control-Expose-Headers": "X-Chat-ID",
    }
    if chat_session_key_value:
        headers["X-Chat-ID"] = chat_session_key_value

    return headers


class HaivenBaseApi:
    def __init__(
        self,
        app,
        chat_manager: ChatManager,
        model_config: ModelConfig,
        prompt_list: PromptList,
    ):
        self.chat_manager = chat_manager
        self.model_config = model_config
        self.prompt_list = prompt_list

    def _is_api_key_auth(self, request):
        """Check if the request is using API key authentication."""
        return auth_util.is_api_key_auth(request)

    def _get_request_source(self, request):
        """Get the source of the request (mcp, ui, or unknown)."""
        return auth_util.get_request_source(request)

    def get_hashed_user_id(self, request):
        """Get the hashed user ID from the request session."""
        return auth_util.get_hashed_user_id(request)

    def stream_json_chat(
        self,
        prompt,
        chat_category,
        chat_session_key_value=None,
        document_keys=None,
        prompt_id=None,
        user_identifier=None,
        contexts=None,
        origin_url=None,
        model_config=None,
        userContext=None,
    ):
        """Stream JSON chat with simplified event handling"""
        try:

            def stream_with_events(chat_session, prompt):
                try:
                    # Pass through the formatted events from the chat session (original behavior)
                    for event_str in chat_session.run(prompt):
                        # Ensure we're yielding strings, not dicts
                        if isinstance(event_str, dict):
                            yield json.dumps(event_str)
                        else:
                            yield str(event_str)

                except Exception as error:
                    error_msg = (
                        str(error).strip()
                        or "Error while the model was processing the input"
                    )
                    print(f"[ERROR]: {error_msg}")
                    # Send error in JSON format for JSON chat
                    error_response = {"data": f"[ERROR]: {error_msg}"}
                    yield json.dumps(error_response)

            chat_session_key_value, chat_session = self.chat_manager.json_chat(
                model_config=model_config or self.model_config,
                session_id=chat_session_key_value,
                options=ChatOptions(in_chunks=True, category=chat_category),
                contexts=contexts or [],
                user_context=userContext,
            )

            self.log_run(
                chat_session,
                origin_url,
                user_identifier,
                chat_session_key_value,
                prompt_id,
                contexts,
                userContext,
            )

            return StreamingResponse(
                stream_with_events(chat_session, prompt),
                media_type=streaming_media_type(),
                headers=streaming_headers(chat_session_key_value),
            )

        except Exception as error:
            raise Exception(error)

    def log_run(
        self,
        chat_session,
        origin_url,
        user_identifier,
        chat_session_key_value,
        prompt_id,
        context,
        userContext,
    ):
        return chat_session.log_run(
            {
                "url": origin_url,
                "user_id": user_identifier,
                "session": chat_session_key_value,
                "prompt_id": prompt_id,
                "context": ",".join(context or []),
                "is_user_context_included": True if userContext else False,
            }
        )

    def stream_text_chat(
        self,
        prompt,
        chat_category,
        chat_session_key_value=None,
        document_keys=[],
        prompt_id=None,
        user_identifier=None,
        contexts=None,
        origin_url=None,
        userContext=None,
        model_config=None,
    ):
        """Stream text chat with simplified event handling"""
        try:

            def stream_with_events(chat_session: StreamingChat, prompt):
                try:
                    if document_keys:
                        # Handle document-based streaming
                        for (
                            event_str,
                            sources_markdown,
                        ) in chat_session.run_with_document(document_keys, prompt):
                            # Ensure we're yielding strings, not dicts
                            if isinstance(event_str, dict):
                                yield json.dumps(event_str)
                            else:
                                yield str(event_str)
                    else:
                        # Handle regular streaming
                        for event_str in chat_session.run(prompt):
                            # Ensure we're yielding strings, not dicts
                            if isinstance(event_str, dict):
                                yield json.dumps(event_str)
                            else:
                                yield str(event_str)

                except Exception as error:
                    error_msg = (
                        str(error).strip()
                        or "Error while the model was processing the input"
                    )
                    print(f"[ERROR]: {error_msg}")
                    yield f"[ERROR]: {error_msg}"

            chat_session_key_value, chat_session = self.chat_manager.streaming_chat(
                model_config=model_config or self.model_config,
                session_id=chat_session_key_value,
                options=ChatOptions(in_chunks=True, category=chat_category),
                contexts=contexts or [],
                user_context=userContext,
            )

            self.log_run(
                chat_session,
                origin_url,
                user_identifier,
                chat_session_key_value,
                prompt_id,
                contexts,
                userContext,
            )

            return StreamingResponse(
                stream_with_events(chat_session, prompt),
                media_type=streaming_media_type(),
                headers=streaming_headers(chat_session_key_value),
            )

        except Exception as error:
            raise Exception(error)
