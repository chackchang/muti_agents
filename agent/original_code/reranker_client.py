"""Small SiliconFlow-compatible document reranker client."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()


def _setting(name: str, fallback: str = "") -> str:
    return str(os.getenv(name) or fallback).strip()


async def rerank_documents(
    query: str,
    documents: List[str],
) -> Optional[List[Tuple[int, float]]]:
    """Return ``(document_index, score)`` pairs, or None when unavailable."""
    if not query or not documents:
        return []

    api_key = _setting("RERANK_API_KEY", _setting("EMBEDDING_API_KEY", _setting("OPENAI_API_KEY")))
    base_url = _setting("RERANK_BASE_URL", _setting("EMBEDDING_BASE_URL", _setting("OPENAI_BASE_URL"))).rstrip("/")
    model = _setting("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    if not api_key or not base_url or not model:
        return None

    payload: Dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
        "return_documents": False,
    }
    try:
        timeout = float(_setting("RERANK_TIMEOUT_SECONDS", "8"))
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/rerank",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            return None
        return [
            (int(item["index"]), float(item["relevance_score"]))
            for item in results
            if isinstance(item, dict) and "index" in item and "relevance_score" in item
        ]
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        return None
