"""导出层。"""

__all__ = ["JsonExporter"]


def __getattr__(name: str):
    """按需导入，避免 ``python -m utils.exporters.json`` 触发重复加载。"""
    if name == "JsonExporter":
        from .json import JsonExporter

        return JsonExporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
