"""
llm_client.py – Unified LLM client supporting OpenAI and Groq.
"""
from __future__ import annotations

from typing import Optional

from config import (
    GROQ_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
)
from logger import get_logger

log = get_logger("llm_client")


def _call_openai(messages: list[dict], model: str, temperature: float, max_tokens: int) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.error("OpenAI call failed: %s", e)
        raise


def _call_groq(messages: list[dict], model: str, temperature: float, max_tokens: int) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.error("Groq call failed: %s", e)
        raise


def llm_complete(
    prompt: str,
    system: str = "You are a helpful assistant.",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Call the configured LLM. Falls back gracefully if no key is set."""
    _provider   = provider or LLM_PROVIDER
    _model      = model or LLM_MODEL
    _temp       = temperature if temperature is not None else LLM_TEMPERATURE
    _max_tokens = max_tokens or LLM_MAX_TOKENS

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]

    log.debug("LLM request | provider=%s model=%s tokens=%d", _provider, _model, _max_tokens)

    if _provider == "groq":
        if not GROQ_API_KEY:
            log.warning("GROQ_API_KEY not set; returning placeholder")
            return "[LLM unavailable – set GROQ_API_KEY in .env]"
        return _call_groq(messages, _model, _temp, _max_tokens)

    elif _provider == "openai":
        if not OPENAI_API_KEY:
            log.warning("OPENAI_API_KEY not set; returning placeholder")
            return "[LLM unavailable – set OPENAI_API_KEY in .env]"
        return _call_openai(messages, _model, _temp, _max_tokens)

    else:
        raise ValueError(f"Unknown LLM provider: {_provider}")
