"""utils 包：爬虫、适配、归一化与导出。"""

__all__ = ["JsonExporter", "Pipeline"]


def __getattr__(name: str):
    """按需导入，避免 ``python -m utils.exporters.json`` 触发重复加载。"""
    if name == "JsonExporter":
        from .exporters.json import JsonExporter

        return JsonExporter
    if name == "Pipeline":
        from .exporters.json import Pipeline

        return Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
