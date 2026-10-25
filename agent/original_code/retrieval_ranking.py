"""Ranking and fusion policies for multi-route retrieval."""

from typing import Any, Dict, List

from .models import ChunkResult


def merge_ranked_chunk_results(result_sets: List[List[ChunkResult]], limit: int) -> List[ChunkResult]:
    """Fuse route rankings with reciprocal rank fusion (RRF).

    Raw scores from vector, lexical, and translated-language routes are not
    comparable. RRF uses only per-route rank, while retaining the best raw
    score and route metadata for diagnostics and downstream policies.
    """
    rrf_constant = 60.0
    entries: Dict[str, Dict[str, Any]] = {}
    for route_index, results in enumerate(result_sets):
        for rank, result in enumerate(results or [], start=1):
            key = str(result.chunk_id)
            contribution = 1.0 / (rrf_constant + rank)
            entry = entries.get(key)
            if entry is None:
                metadata = dict(result.metadata or {})
                routes = list(dict.fromkeys([
                    *list(metadata.get("retrieval_routes") or []),
                    str(metadata.get("retrieval_route") or ""),
                    f"route_{route_index}",
                ]))
                metadata["retrieval_routes"] = [route for route in routes if route]
                entries[key] = {
                    "result": result.model_copy(update={"metadata": metadata}),
                    "fusion_score": contribution,
                    "route_count": 1,
                }
                continue

            entry["fusion_score"] += contribution
            entry["route_count"] += 1
            previous = entry["result"]
            current = result if result.score > previous.score else previous
            metadata = dict(current.metadata or {})
            routes = list(dict.fromkeys([
                *list((previous.metadata or {}).get("retrieval_routes") or []),
                *list(metadata.get("retrieval_routes") or []),
                str(metadata.get("retrieval_route") or ""),
                *list((result.metadata or {}).get("retrieval_routes") or []),
                str((result.metadata or {}).get("retrieval_route") or ""),
                f"route_{route_index}",
            ]))
            metadata["retrieval_routes"] = [route for route in routes if route]
            entry["result"] = current.model_copy(update={"metadata": metadata})

    ranked: List[ChunkResult] = []
    for entry in entries.values():
        result = entry["result"]
        metadata = dict(result.metadata or {})
        metadata["retrieval_fusion"] = "rrf"
        metadata["retrieval_fusion_score"] = round(float(entry["fusion_score"]), 8)
        metadata["retrieval_route_count"] = int(entry["route_count"])
        ranked.append(result.model_copy(update={"metadata": metadata}))
    ranked.sort(
        key=lambda item: (
            -float((item.metadata or {}).get("retrieval_fusion_score") or 0.0),
            -float(item.score),
        )
    )
    return ranked[: max(1, int(limit or 1))]
