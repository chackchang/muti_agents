"""Validation policy for persisted document section paths."""

import re
from typing import Any, Dict, List, Tuple

from agent.model.intent_models import IntentPlan


def _normalize_path(value: Any) -> str:
    return re.sub(r"[\s\W_]+", "", str(value or "").lower(), flags=re.UNICODE)


def group_section_catalog(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows or []:
        document_id = str(row.get("document_id") or "").strip()
        path = str(row.get("section_path_text") or "").strip()
        key = (document_id, _normalize_path(path))
        if not document_id or not path or key in seen:
            continue
        seen.add(key)
        grouped.setdefault(document_id, []).append(dict(row))
    for sections in grouped.values():
        sections.sort(key=lambda item: (int(item.get("section_start_line") or 0), str(item.get("section_path_text") or "")))
    return grouped


def validate_catalog_section_steps(
    plan: IntentPlan,
    catalog: Dict[str, List[Dict[str, Any]]],
) -> Tuple[IntentPlan, List[str]]:
    """Canonicalize exact section paths and drop unsafe section steps.

    A selected target document has a real catalog, so an arbitrary model phrase
    must not fall through to the legacy fuzzy section search.
    """
    validated = plan.model_copy(deep=True)
    warnings: List[str] = []
    kept = []
    for step in validated.retrieval_steps:
        if step.tool != "section_search":
            kept.append(step)
            continue
        document_id = str(step.document_id or "").strip()
        if not document_id and len(catalog) == 1:
            document_id = next(iter(catalog))
            step.document_id = document_id
        sections = list(catalog.get(document_id) or [])
        if not document_id and catalog:
            warnings.append("catalog_section_query_missing_document_id")
            continue
        if not document_id or not sections:
            # Broad corpus section queries retain existing fuzzy behavior.
            kept.append(step)
            continue
        requested = str(step.exact_section_path or step.section_query or "").strip()
        normalized = _normalize_path(requested)
        matches = [
            item for item in sections
            if _normalize_path(item.get("section_path_text")) == normalized
        ]
        if len(matches) != 1:
            warnings.append(f"invalid_catalog_section_query:{document_id}")
            continue
        canonical = str(matches[0].get("section_path_text") or "").strip()
        step.exact_section_path = canonical
        step.section_query = canonical
        kept.append(step)
    validated.retrieval_steps = kept
    validated.max_tools = min(validated.max_tools, len(kept))
    if warnings:
        validated.warnings = list(dict.fromkeys([*validated.warnings, *warnings]))
    return validated, warnings
