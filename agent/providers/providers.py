"""LLM 供应商配置。

嵌入不在这里：它是本地的 GE-M3 运行时，统一由 ``utils`` 门面导出
（``from utils import aembed_texts``），与远端 OpenAI 兼容接口无关。
"""

import os
import asyncio
from typing import Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIChatModel
import openai
from dotenv import load_dotenv

load_dotenv()


def get_llm_model(model_choice: Optional[str] = None) -> OpenAIChatModel:
    llm_choice = model_choice or os.getenv("LLM_CHOICE", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    provider = OpenAIProvider(
        api_key=api_key,
        base_url=base_url,
    )
    return OpenAIChatModel(llm_choice, provider=provider)


async def test_llm_connection() -> tuple[bool, str | None]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    llm_choice = os.getenv("LLM_CHOICE", "").strip()

    if not api_key:
        return False, "OPENAI_API_KEY is missing"
    if not base_url:
        return False, "OPENAI_BASE_URL is missing"
    if not llm_choice:
        return False, "LLM_CHOICE is missing"

    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    try:
        await asyncio.wait_for(
            client.chat.completions.create(
                model=llm_choice,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
            ),
            timeout=8,
        )
        return True, None
    except Exception as exc:
        return False, str(exc)
