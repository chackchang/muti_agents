"""编排层对外门面。

外部（``agent/api.py``、``evals``）只应该 ``from agent.orchestration import ...``，
不直接依赖内部的模块文件名。这样内部怎么拆文件、怎么重排节点，都不会波及调用方。

用 PEP 562 的模块级 ``__getattr__`` 惰性导入：包被导入时几乎不执行任何东西，
只有真正取用某个名字时才去加载对应模块。否则 ``import agent.orchestration.intent_planner``
会顺带把整条分析链（LangGraph、LangChain、prompt）全部拖进来。
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentDependencies": ("agent.orchestration.agent_runtime", "AgentDependencies"),
    "GENERATION_RETRY_FAILED_MESSAGE": (
        "agent.orchestration.agent_langchain",
        "GENERATION_RETRY_FAILED_MESSAGE",
    ),
    "get_langchain_chat_model": ("agent.orchestration.agent_langchain", "get_langchain_chat_model"),
    "iter_langchain_agent_stream": (
        "agent.orchestration.agent_langchain",
        "iter_langchain_agent_stream",
    ),
    "retry_langchain_agent_after_degenerate": (
        "agent.orchestration.agent_langchain",
        "retry_langchain_agent_after_degenerate",
    ),
    "run_langchain_agent": ("agent.orchestration.agent_langchain", "run_langchain_agent"),
    "run_langgraph_analysis": ("agent.orchestration.agent_langgraph", "run_langgraph_analysis"),
    "stream_langchain_agent": ("agent.orchestration.agent_langchain", "stream_langchain_agent"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value  # 缓存，之后的访问不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
