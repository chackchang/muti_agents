"""JSON 导出器与清洗管线的工具入口。

库调用：

    from utils.exporters.json import export, export_json

    document = export("data/text1", "niuke", "out/text1.json")
    payload = export_json("data/text1", "niuke")

命令行：

    python -m utils.exporters.json data/text1 niuke --out out/text1.json

请用 ``python -m`` 运行；直接执行本文件时目录会进入 sys.path，
同名模块会遮蔽标准库 ``json``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..adapters.base import Adapter
from ..norm.base import Document, Normalizer


class Pipeline:
    """清洗管线编排：adapter 读取 markdown，再交给 norm 统一格式。

    只负责“取 → 归一化”，不落盘；落盘由 :class:`JsonExporter` 负责。
    """

    def __init__(
        self,
        adapter: Adapter | None = None,
        normalizer: Normalizer | None = None,
    ):
        self.adapter = adapter or Adapter()
        self.normalizer = normalizer or Normalizer()

    def run(self, path: str, type: str) -> Document:
        """按 ``path + type`` 取内容并归一化为 Document。"""
        content, metadata = self.adapter.read_document(path, type)
        return self.normalizer.normalize(content, metadata)


class JsonExporter:
    """只负责 Document 与 JSON 之间的转换。"""

    @staticmethod
    def dumps(document: Document) -> str:
        return json.dumps(document, ensure_ascii=False, indent=2)

    @classmethod
    def write(cls, document: Document, output_path: str) -> Path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(cls.dumps(document), encoding="utf-8")
        return target


def export(path: str, type: str, output_path: str | None = None) -> Document:
    """工具入口：输入 path、type，返回 Document；给了 output_path 就写 JSON。"""
    document = Pipeline().run(path, type)
    if output_path:
        JsonExporter.write(document, output_path)
    return document


def export_json(path: str, type: str) -> str:
    """工具入口的 JSON 字符串版本，方便 agent 直接使用。"""
    return JsonExporter.dumps(export(path, type))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把文档/网页文本清洗为统一 JSON")
    parser.add_argument("--path", required=True, help="输入文件路径")
    parser.add_argument(
        "--type", required=True, choices=Adapter.supported_types(), help="适配器类型"
    )
    parser.add_argument("--out", help="输出 JSON 路径；省略时打印到 stdout")
    args = parser.parse_args(argv)

    if args.out:
        document = export(args.path, args.type, args.out)
        print(
            f"已写入 {args.out}（id={document['id']}，{len(document['content'])} 字符）",
            file=sys.stderr,
        )
    else:
        print(export_json(args.path, args.type))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
