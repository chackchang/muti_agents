"""Deterministic final-evidence selection across non-comparable retrieval routes."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from agent.model.intent_models import IntentPlan
from agent.http_api.reranker_client import rerank_documents


TEXT_CANDIDATE_LIMIT = 20
TEXT_FINAL_LIMIT = 10
ARTIFACT_CANDIDATE_LIMIT = 12
ARTIFACT_FINAL_LIMIT = 6
EXTERNAL_CANDIDATE_LIMIT = 12
EXTERNAL_FINAL_LIMIT = 6
DEFAULT_MAX_FINAL_EVIDENCE = 22


def _hit_key(hit: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(hit.get("document_id") or "").strip(),
        str(hit.get("chunk_id") or "").strip(),
        str(hit.get("content") or "").strip()[:120],
    )


def _source_group(hit: Dict[str, Any], target_ids: set[str], graph_ids: set[str]) -> str:
    metadata = dict(hit.get("metadata") or {})
    source_type = str(metadata.get("source_type") or "")
    route = str(metadata.get("retrieval_route") or "")
    document_id = str(hit.get("document_id") or "")
    if source_type == "external_academic":
        return "external_academic"
    if source_type == "general_web":
        return "general_web"
    if document_id in graph_ids and document_id not in target_ids:
        return "graph_supplement"
    if route == "section_search" or source_type == "local_section":
        # Section hits are prose evidence and should compete with hybrid/vector
        # hits in the same reranker pool.
        return "target_text" if not target_ids or document_id in target_ids else "graph_supplement"
    if route == "artifact_search" or source_type == "local_artifact":
        return "target_artifact" if not target_ids or document_id in target_ids else "graph_supplement"
    return "target_text" if not target_ids or document_id in target_ids else "graph_supplement"


def _required_groups(plan: IntentPlan) -> set[str]:
    source_to_group = {
        "local_kb": "target_text",
        "local_section": "target_text",
        "local_artifact": "target_artifact",
        "external_academic": "external_academic",
        "general_web": "general_web",
    }
    return {
        source_to_group[source]
        for source in (plan.source_requirements or {}).get("required_source_types") or []
        if source in source_to_group
    }


async def select_final_evidence(
    candidates: List[Dict[str, Any]],
    plan: IntentPlan,
    *,
    question: str = "",
    target_document_ids: List[str],
    graph_expanded_document_ids: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep native order inside each group; do not compare scores across sources."""
    target_ids = {str(item).strip() for item in target_document_ids if str(item).strip()}
    graph_ids = {str(item).strip() for item in graph_expanded_document_ids if str(item).strip()}
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    deduped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for raw_hit in candidates or []:
        if not isinstance(raw_hit, dict):
            continue
        hit = dict(raw_hit)
        metadata = dict(hit.get("metadata") or {})
        group = _source_group(hit, target_ids, graph_ids)
        key = _hit_key(hit)
        existing = deduped.get(key)
        if existing is not None:
            existing_meta = dict(existing.get("metadata") or {})
            routes = list(dict.fromkeys([
                *list(existing_meta.get("retrieval_routes") or []),
                *list(metadata.get("retrieval_routes") or []),
                str(metadata.get("retrieval_route") or ""),
            ]))
            existing_meta["retrieval_routes"] = [route for route in routes if route]
            existing["metadata"] = existing_meta
            continue
        metadata["evidence_group"] = group
        hit["metadata"] = metadata
        deduped[key] = hit
        groups[group].append(hit)

    # Planner decides source types; deterministic runtime policy decides counts.
    rerank_applied = False
    rerank_reason = "not_needed"
    text_candidates = groups.get("target_text", [])[:TEXT_CANDIDATE_LIMIT]
    if text_candidates and question:
        documents = [
            " | ".join(
                value for value in (
                    str(hit.get("document_title") or ""),
                    str((hit.get("metadata") or {}).get("section_path") or (hit.get("metadata") or {}).get("section_title") or ""),
                    str(hit.get("content") or "")[:2000],
                )
                if value
            )
            for hit in text_candidates
        ]
        try:
            reranked = await rerank_documents(question, documents)
        except Exception:
            # Reranking is an optional refinement; retain deterministic policy selection.
            reranked = None
            rerank_reason = "error_fallback_native_order"
        if reranked:
            score_by_index = {index: score for index, score in reranked}
            scores = list(score_by_index.values())
            minimum = min(scores) if scores else 0.0
            maximum = max(scores) if scores else 0.0
            span = maximum - minimum
            for index, hit in enumerate(text_candidates):
                metadata = dict(hit.get("metadata") or {})
                raw_score = score_by_index.get(index, minimum)
                normalized = (raw_score - minimum) / span if span else 1.0
                routes = set(metadata.get("retrieval_routes") or [])
                routes.add(str(metadata.get("retrieval_route") or ""))
                section_bonus = 0.05 if "section_search" in routes or metadata.get("source_type") == "local_section" else 0.0
                multi_route_bonus = 0.02 if len([route for route in routes if route]) > 1 else 0.0
                metadata["rerank_score"] = raw_score
                metadata["selection_score"] = min(1.0, normalized + section_bonus + multi_route_bonus)
                hit["metadata"] = metadata
            text_candidates.sort(key=lambda hit: float((hit.get("metadata") or {}).get("selection_score") or 0.0), reverse=True)
            groups["target_text"] = text_candidates
            rerank_applied = True
            rerank_reason = "siliconflow_reranker"
        elif rerank_reason == "not_needed":
            rerank_reason = "unavailable_fallback_native_order"

    fixed_limits = {
        "target_text": TEXT_FINAL_LIMIT,
        "target_artifact": ARTIFACT_FINAL_LIMIT,
        "external_academic": EXTERNAL_FINAL_LIMIT,
        "general_web": EXTERNAL_FINAL_LIMIT,
        "graph_supplement": 4,
    }
    candidate_limits = {
        "target_artifact": ARTIFACT_CANDIDATE_LIMIT,
        "external_academic": EXTERNAL_CANDIDATE_LIMIT,
        "general_web": EXTERNAL_CANDIDATE_LIMIT,
    }
    for group, limit in candidate_limits.items():
        groups[group] = groups.get(group, [])[:limit]
    external_remaining = EXTERNAL_CANDIDATE_LIMIT
    for group in ("external_academic", "general_web"):
        groups[group] = groups[group][:external_remaining]
        external_remaining = max(0, external_remaining - len(groups[group]))

    # Reserve the best item from each artifact type, then fill by score.
    artifact_candidates = groups.get("target_artifact", [])
    if artifact_candidates:
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for hit in artifact_candidates:
            artifact_type = str((hit.get("metadata") or {}).get("artifact_type") or "unknown")
            by_type[artifact_type].append(hit)
        for items in by_type.values():
            items.sort(key=lambda hit: float(hit.get("score") or 0.0), reverse=True)
        reserved = [items[0] for items in by_type.values() if items]
        reserved_keys = {_hit_key(hit) for hit in reserved}
        remainder = [hit for hit in artifact_candidates if _hit_key(hit) not in reserved_keys]
        remainder.sort(key=lambda hit: float(hit.get("score") or 0.0), reverse=True)
        groups["target_artifact"] = [*reserved, *remainder]

    required_groups = _required_groups(plan)
    max_total = DEFAULT_MAX_FINAL_EVIDENCE
    # Evidence counts are owned by this runtime policy. The planner field is
    # retained only for backward-compatible plan serialization and is not used
    # to change selection behavior.
    target_minimum = 1
    ordered_groups = ["target_text", "target_artifact", "external_academic", "general_web", "graph_supplement"]
    selected: List[Dict[str, Any]] = []
    selected_keys: set[Tuple[str, str, str]] = set()
    selected_counts: Dict[str, int] = defaultdict(int)

    external_taken = 0

    def take(group: str, count: int) -> None:
        nonlocal external_taken
        for hit in groups.get(group, []):
            if len(selected) >= max_total or selected_counts[group] >= count:
                return
            if group in {"external_academic", "general_web"} and external_taken >= EXTERNAL_FINAL_LIMIT:
                return
            key = _hit_key(hit)
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected_counts[group] += 1
            if group in {"external_academic", "general_web"}:
                external_taken += 1
            selected.append(hit)

    for group in ordered_groups:
        if group in required_groups:
            take(group, 1)

    if len(selected) < target_minimum:
        for group in ordered_groups:
            take(group, fixed_limits.get(group, 2))
            if len(selected) >= target_minimum:
                break

    # Fill remaining capacity after required/target coverage.
    for group in ordered_groups:
        take(group, fixed_limits.get(group, 2))
        if len(selected) >= max_total:
            break

    group_order = {group: index for index, group in enumerate(ordered_groups)}
    selected.sort(key=lambda hit: group_order.get(str((hit.get("metadata") or {}).get("evidence_group") or ""), 99))

    for group, count in selected_counts.items():
        rank = 0
        for hit in selected:
            metadata = dict(hit.get("metadata") or {})
            if metadata.get("evidence_group") == group:
                rank += 1
                metadata["evidence_rank_in_group"] = rank
                metadata["evidence_selected"] = True
                hit["metadata"] = metadata

    missing_required = [group for group in required_groups if not selected_counts.get(group)]
    summary = {
        "candidate_count": len(deduped),
        "selected_count": len(selected),
        "target_min_final_evidence": target_minimum,
        "below_target_minimum": len(selected) < target_minimum,
        "max_final_evidence": max_total,
        "rerank_applied": rerank_applied,
        "rerank_reason": rerank_reason,
        "group_limits": fixed_limits,
        "selected_group_counts": dict(selected_counts),
        "candidate_group_counts": {group: len(items) for group, items in groups.items()},
        "missing_required_groups": missing_required,
    }
    return selected, summary
