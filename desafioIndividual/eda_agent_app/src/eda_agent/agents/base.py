from __future__ import annotations
import os
from typing import Optional
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-4o-mini"

def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ausente. Configure a variável de ambiente.")
    return api_key

def build_llm(model: Optional[str] = None, temperature: float = 0.0) -> ChatOpenAI:
    """
    Fábrica de LLM para todos os agentes.
    """
    api_key = get_api_key()
    return ChatOpenAI(model=(model or DEFAULT_MODEL), temperature=temperature, api_key=api_key)
