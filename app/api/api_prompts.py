# © 2024 Thoughtworks, Inc. | Licensed under the Apache License, Version 2.0  | See LICENSE.md file for permissions.
import io
import re
from typing import List, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import File, Form, UploadFile
from PIL import Image

from embeddings.documents import KnowledgeDocument
from knowledge_manager import KnowledgeManager
from llms.chats import ChatManager
from llms.model_config import ModelConfig
from llms.image_description_service import ImageDescriptionService
from prompts.prompts import PromptList, filter_downloadable_prompts
from prompts.inspirations import InspirationsManager

from api.api_base import HaivenBaseApi, PromptRequestBody, IterateRequest, streaming_media_type, streaming_headers
from config_service import ConfigService
from disclaimer_and_guidelines import DisclaimerAndGuidelinesService
from logger import HaivenLogger
from loguru import logger
import json


_ITERATE_PROMPT_TEMPLATE = """
                    ### Output format: JSON with at least the "id" property repeated
                    Here is my current working state of the data, iterate on those objects based on that request,
                    and only return your new list of the objects in JSON format, nothing else.
                    Be sure to repeat back to me the JSON that I already have, and only update it with my new request.
                    Definitely repeat back to me the "id" property, so I can track your changes back to my original data.
                    For example, if I give you
                    [ { "title": "Paris", "id": 1 }, { "title": "London", "id": 2 } ]
                    and ask you to add information about what you know about each of these cities, then return to me
                    [ { "summary": "capital of France", "id": 1 }, { "summary": "Capital of the UK", "id": 2 } ]
"""


class ApiBasics(HaivenBaseApi):
    def __init__(
        self,
        app: FastAPI,
        chat_manager: ChatManager,
        model_config: ModelConfig,
        prompts_guided: PromptList,
        knowledge_manager: KnowledgeManager,
        prompts_chat: PromptList,
        image_service: ImageDescriptionService,
        config_service: ConfigService,
        disclaimer_and_guidelines: DisclaimerAndGuidelinesService,
        inspirations_manager: InspirationsManager,
    ):
        super().__init__(app, chat_manager, model_config, prompts_guided)
        self.knowledge_manager = knowledge_manager
        self.prompts_chat = prompts_chat
        self.image_service = image_service
        self.config_service = config_service
        self.disclaimer_and_guidelines = disclaimer_and_guidelines
        self.inspirations_manager = inspirations_manager
        self._register_endpoints(app, prompts_guided, prompts_chat, knowledge_manager,
                                 image_service, config_service, disclaimer_and_guidelines)

    def _register_endpoints(self, app, prompts_guided, prompts_chat, knowledge_manager,
                            image_service, config_service, disclaimer_and_guidelines):
        @app.get("/api/models")
        @logger.catch(reraise=True)
        def get_models(request: Request):
            try:
                embeddings = config_service.load_embedding_model()
                vision = config_service.get_image_model()
                chat = config_service.get_chat_model()
                return JSONResponse({"chat": {"id": chat.id, "name": chat.name},
                                     "vision": {"id": vision.id, "name": vision.name},
                                     "embeddings": {"id": embeddings.id, "name": embeddings.name}})
            except Exception as error:
                HaivenLogger.get().error(str(error))
                raise HTTPException(status_code=500, detail=f"Server error: {str(error)}")

        @app.get("/api/prompts")
        @logger.catch(reraise=True)
        def get_prompts(request: Request):
            try:
                return JSONResponse(prompts_chat.get_prompts_with_follow_ups())
            except Exception as error:
                HaivenLogger.get().error(str(error))
                raise HTTPException(status_code=500, detail=f"Server error: {str(error)}")

        @app.get("/api/disclaimer-guidelines")
        @logger.catch(reraise=True)
        def get_disclaimer_and_guidelines(request: Request):
            try:
                if not disclaimer_and_guidelines.is_enabled:
                    return JSONResponse({"enabled": False, "title": "", "content": ""})
                response_data = json.loads(disclaimer_and_guidelines.fetch_disclaimer_and_guidelines())
                response_data["enabled"] = True
                return JSONResponse(response_data)
            except Exception as error:
                HaivenLogger.get().error(str(error))
                raise HTTPException(status_code=500, detail=f"Server error: {str(error)}")

        @app.get("/api/knowledge/snippets")
        @logger.catch(reraise=True)
        def get_knowledge_snippets(request: Request):
            try:
                all_contexts = knowledge_manager.knowledge_base_markdown.get_all_contexts()
                response_data = [{"context": k, "title": v.metadata["title"],
                                  "snippets": {"context": v.content}}
                                 for k, v in all_contexts.items()]
                return JSONResponse(sorted(response_data, key=lambda x: x["context"]))
            except Exception as error:
                HaivenLogger.get().error(str(error))
                raise HTTPException(status_code=500, detail=f"Server error: {str(error)}")

        @app.get("/api/knowledge/documents")
        @logger.catch(reraise=True)
        def get_knowledge_documents(request: Request):
            try:
                documents: List[KnowledgeDocument] = knowledge_manager.knowledge_base_documents.get_documents()
                return JSONResponse([{"key": d.key, "title": d.title, "description": d.description,
                                      "source": d.get_source_title_link()} for d in documents])
            except Exception as error:
                HaivenLogger.get().error(str(error))
                raise HTTPException(status_code=500, detail=f"Server error: {str(error)}")

        @app.post("/api/prompt")
        @logger.catch(reraise=True)
        def chat(request: Request, prompt_data: PromptRequestBody):
            origin_url = request.headers.get("referer")
            try:
                rendered_prompt, stream_fn, model_cfg = self._select_model_and_stream_fn(
                    prompt_data, prompts_guided, prompts_chat)
                return stream_fn(prompt=rendered_prompt, model_config=model_cfg,
                                 chat_category="boba-chat",
                                 chat_session_key_value=prompt_data.chatSessionId,
                                 document_keys=prompt_data.document,
                                 prompt_id=prompt_data.promptid,
                                 user_identifier=self.get_hashed_user_id(request),
                                 contexts=prompt_data.contexts,
                                 userContext=prompt_data.userContext,
                                 origin_url=origin_url)
            except Exception as error:
                HaivenLogger.get().error(str(error))
                raise HTTPException(status_code=500, detail=f"Server error: {str(error)}")

        @app.post("/api/prompt/iterate")
        def iterate(prompt_data: IterateRequest):
            try:
                if not prompt_data.chatSessionId:
                    raise HTTPException(status_code=400, detail="chatSessionId is required")
                return self.stream_json_chat(prompt=self._build_iterate_prompt(prompt_data),
                                             chat_category="boba-chat",
                                             chat_session_key_value=prompt_data.chatSessionId,
                                             contexts=prompt_data.contexts,
                                             userContext=prompt_data.user_context)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

        @app.post("/api/prompt/render")
        @logger.catch(reraise=True)
        def render_prompt(prompt_data: PromptRequestBody):
            if not prompt_data.promptid:
                raise HTTPException(status_code=500, detail="Server error: promptid is required")
            prompts = prompts_guided if prompt_data.promptid.startswith("guided-") else prompts_chat
            rendered_prompt, template = prompts.render_prompt(
                prompt_choice=prompt_data.promptid, user_input=prompt_data.userinput)
            return JSONResponse({"prompt": rendered_prompt, "template": template.template})

        @app.post("/api/prompt/image")
        @logger.catch(reraise=True)
        async def describe_image(prompt: str = Form(...), file: UploadFile = File(...)):
            try:
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                return StreamingResponse(
                    (chunk for chunk in image_service.prompt_with_image(image, prompt)),
                    media_type=streaming_media_type(), headers=streaming_headers(None))
            except Exception as error:
                HaivenLogger.get().error(str(error))
                raise HTTPException(status_code=500, detail=f"Server error: {str(error)}")

        @app.get("/api/inspirations")
        @logger.catch(reraise=True)
        def get_inspirations(request: Request):
            try:
                return JSONResponse(self.inspirations_manager.get_inspirations())
            except Exception as error:
                HaivenLogger.get().error(str(error))
                raise HTTPException(status_code=500, detail=f"Server error: {str(error)}")

        @app.get("/api/inspirations/{inspiration_id}")
        @logger.catch(reraise=True)
        def get_inspiration_by_id(request: Request, inspiration_id: str):
            try:
                inspiration = self.inspirations_manager.get_inspiration_by_id(inspiration_id)
                if inspiration is None:
                    raise HTTPException(status_code=404, detail="Inspiration not found")
                return JSONResponse(inspiration)
            except HTTPException:
                raise
            except Exception as error:
                HaivenLogger.get().error(str(error))
                raise HTTPException(status_code=500, detail=f"Server error: {str(error)}")

        @app.get("/api/download-prompt")
        @logger.catch(reraise=True)
        def download_prompt(request: Request, prompt_id: str = None, category: str = None):
            user_id = self.get_hashed_user_id(request)
            source = self._get_request_source(request)
            try:
                return JSONResponse(self._get_downloadable_prompts(
                    prompts_chat, prompt_id, category, user_id, source))
            except HTTPException:
                raise
            except Exception as error:
                HaivenLogger.get().error(str(error), extra={"ERROR": "Downloading prompts failed",
                    "user_id": user_id, "prompt_id": prompt_id, "category": category, "source": source})
                raise HTTPException(status_code=500, detail=f"Server error: {str(error)}")

    def _select_model_and_stream_fn(self, prompt_data: PromptRequestBody,
                                    prompts_guided: PromptList, prompts_chat: PromptList) -> Tuple:
        """Select the appropriate stream function and model config based on prompt data."""
        stream_fn = self.stream_text_chat
        selected_model_config = self.model_config
        rendered_prompt = prompt_data.userinput

        if prompt_data.promptid:
            prompts = prompts_guided if prompt_data.promptid.startswith("guided-") else prompts_chat
            rendered_prompt, _ = prompts.render_prompt(
                prompt_choice=prompt_data.promptid, user_input=prompt_data.userinput)
            if prompts.produces_json_output(prompt_data.promptid):
                stream_fn = self.stream_json_chat
            prompt_obj = prompts.get(prompt_data.promptid)
            if prompt_obj and prompt_obj.metadata.get("grounded", True):
                selected_model_config = ModelConfig("perplexity", "perplexity", "Perplexity")

        if prompt_data.json is True:
            stream_fn = self.stream_json_chat

        return rendered_prompt, stream_fn, selected_model_config

    def _build_iterate_prompt(self, prompt_data: IterateRequest) -> str:
        """Build the iterate prompt from request data using the standard template."""
        return (f"\n\n                    My new request:\n                    {prompt_data.userinput}\n                    "
                + _ITERATE_PROMPT_TEMPLATE
                + f"\n                    ### Current JSON data\n                    {prompt_data.scenarios}\n"
                  "                    Please iterate on this data based on my request. Apply my request to ALL of the objects.\n                ")

    def _get_downloadable_prompts(self, prompts_chat: PromptList, prompt_id: str,
                                  category: str, user_id: str, source: str):
        """Retrieve and return downloadable prompts, logging analytics."""
        def _valid(val):
            return bool(val) and re.match(r"^[a-zA-Z0-9_-]{1,100}$", val)

        if prompt_id is not None:
            if not _valid(prompt_id):
                raise HTTPException(status_code=400, detail="Invalid prompt_id")
            prompt = prompts_chat.get_a_prompt_with_follow_ups(prompt_id, download_prompt=True)
            if not prompt:
                raise Exception("Prompt not found")
            if prompt.get("download_restricted", False):
                HaivenLogger.get().analytics("Download restricted prompt attempted",
                    {"user_id": user_id, "prompt_id": prompt_id, "category": "Individual Prompt", "source": source})
                raise HTTPException(status_code=403, detail="This prompt is not available for download")
            HaivenLogger.get().analytics("Download prompt",
                {"user_id": user_id, "prompt_id": prompt_id, "category": "Individual Prompt", "source": source})
            return [prompt]

        cat_label = category if (category and category.strip()) else "all"
        if category and category.strip():
            if not _valid(category):
                raise HTTPException(status_code=400, detail="Invalid category")
            prompts = prompts_chat.get_prompts_with_follow_ups(download_prompt=True, category=category)
        else:
            prompts = prompts_chat.get_prompts_with_follow_ups(download_prompt=True)

        downloadable_prompts = filter_downloadable_prompts(prompts)
        for p in downloadable_prompts:
            HaivenLogger.get().analytics("Download prompt",
                {"user_id": user_id, "prompt_id": p.get("identifier"), "category": cat_label, "source": source})
        return downloadable_prompts
