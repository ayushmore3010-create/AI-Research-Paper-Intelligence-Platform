"""Provider-neutral LLM interface with mock and OpenAI implementations."""

from typing import Protocol

from app.config import Settings


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str: ...


class MockLLM:
    def generate(self, prompt: str) -> str:
        return "The configured mock provider cannot generate a model answer. The retrieved sources are shown below."


class OpenAIProvider:
    def __init__(self, config: Settings):
        if not config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from openai import OpenAI
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.llm_model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(model=self.model, temperature=0, messages=[
            {"role": "system", "content": "Answer only from the supplied research-paper context. Say the information was not found when context is insufficient."},
            {"role": "user", "content": prompt},
        ])
        return response.choices[0].message.content or "Information not found in the uploaded papers."


def build_llm(config: Settings) -> LLMProvider:
    return OpenAIProvider(config) if config.llm_provider.lower() == "openai" else MockLLM()
