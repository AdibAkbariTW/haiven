# © 2024 Thoughtworks, Inc. | Licensed under the Apache License, Version 2.0  | See LICENSE.md file for permissions.
import os
import re
from typing import List
from dataclasses import dataclass

import frontmatter


@dataclass
class KnowledgeMarkdown:
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata


@dataclass
class ContextMetadata:
    key: str
    title: str


class KnowledgeBaseMarkdown:
    _knowledge: dict[str, KnowledgeMarkdown]

    def __init__(self):
        self._knowledge = {}

    def _load_context(self, path: str) -> list[KnowledgeMarkdown]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

        if os.path.isfile(path):
            if not path.endswith(".md") or path.endswith("README.md"):
                return
            try:
                content = frontmatter.load(path)
                if content.content != "":
                    return KnowledgeMarkdown(content.content, content.metadata)

            except Exception as e:
                print(f"Error processing markdown file {path}: {str(e)}")
        else:
            raise ValueError(f"Path must be a file or directory: {path}")

        return None

    def load_for_context(self, context: str, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"The specified path does not exist, no knowledge will be loaded: {path}"
            )

        context_content = self._load_context(path)
        self._knowledge[context] = context_content

    def get_all_contexts(self) -> dict[str, KnowledgeMarkdown]:
        """
        Returns a dictionary containing all available contexts
        """
        return self._knowledge

    _INJECTION_PATTERNS = re.compile(
        r"^\s*(ignore\s|disregard\s|you are now\s|forget\s|override\s|system:\s|assistant:\s|user:\s)",
        re.IGNORECASE | re.MULTILINE,
    )
    _EXCESSIVE_SPECIAL_CHARS = re.compile(r"[^\w\s.,;:!?'\"\-()\[\]{}/\\@#%&*+=<>|~`]{3,}")

    def _sanitise_user_context(self, user_context: str) -> str:
        """
        Strip prompt-injection patterns from user-supplied context before it is
        included in the LLM system prompt.
        """
        if not user_context:
            return user_context

        lines = user_context.splitlines()
        sanitised_lines = []
        for line in lines:
            if self._INJECTION_PATTERNS.match(line):
                continue
            line = self._EXCESSIVE_SPECIAL_CHARS.sub("", line)
            sanitised_lines.append(line)

        return "\n".join(sanitised_lines)

    def aggregate_all_contexts(
        self, contexts: List[str], user_context: str = None
    ) -> str:
        """
        Return all required contexts' contents appended as one string
        """
        knowledgePackContextsAggregated = None
        if contexts:
            knowledgePackContextsAggregated = "\n\n".join(
                self._knowledge[context_key].content for context_key in contexts
            )

        sanitised_user_context = self._sanitise_user_context(user_context)
        return "\n\n".join(
            filter(None, [knowledgePackContextsAggregated, sanitised_user_context])
        )
