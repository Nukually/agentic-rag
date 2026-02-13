from __future__ import annotations

from typing import Sequence

from openai import OpenAI

from src.utils.config import AppConfig


class OpenAIClientBundle:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm_client = OpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_api_url,
            timeout=config.llm_timeout,
        )
        self.embedding_client = OpenAI(
            api_key=config.embedding_api_key,
            base_url=config.embedding_api_url,
            timeout=config.embedding_timeout,
        )

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        resp = self.llm_client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.llm_temperature if temperature is None else temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    def embed_texts(self, texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        normalized = [text.replace("\n", " ").strip() for text in texts]
        actual_batch_size = batch_size if batch_size is not None else self.config.embedding_batch_size
        actual_batch_size = max(1, int(actual_batch_size))

        for i in range(0, len(normalized), actual_batch_size):
            batch = normalized[i : i + actual_batch_size]
            if not batch:
                continue
            resp = self.embedding_client.embeddings.create(
                model=self.config.embedding_model,
                input=batch,
            )
            all_embeddings.extend(item.embedding for item in resp.data)

        return all_embeddings
