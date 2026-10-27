"""意图规划的数据契约。

这些 pydantic 模型原先定义在 ``agent.orchestration.intent_planner`` 里，但检索层
（``agent.retrieval``）和证据层（``agent.evidence``）也要读它们，于是下层反过来
import 了编排层。搬到模型层后依赖方向统一为「编排 → 检索/证据 → 模型」，
编排层才能作为独立门面被外部引用。
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

IntentType = Literal[
    "direct_answer",
    "local_paper_qa",
    "local_section_qa",
    "local_artifact_qa",
    "multi_paper_compare",
    "external_paper_discovery",
    "web_information",
    "general_technical_explanation",
    "unclear",
]

RetrievalToolName = Literal[
    "none",
    "hybrid_search",
    "vector_search",
    "section_search",
    "artifact_search",
    "openalex_search",
    "web_search",
]

ArtifactType = Literal["table", "figure", "algorithm"]
GraphRelationType = Literal["semantic_similarity", "cites", "method_lineage"]
GraphDirection = Literal["both", "outgoing", "incoming"]
EvidenceGroupName = Literal[
    "target_text",
    "target_section",
    "target_artifact",
    "graph_supplement",
    "external_academic",
    "general_web",
]


class EvidenceGroupBudget(BaseModel):
    """Per-source evidence slots for the final answer, not retrieval result ranks."""

    group: EvidenceGroupName
    target_count: int = Field(default=0, ge=0, le=15)
    max_count: int = Field(default=0, ge=0, le=15)
    priority: int = Field(default=3, ge=1, le=6)
    required: bool = False
    reason: str = ""


class RetrievalStep(BaseModel):
    tool: RetrievalToolName
    query: str = ""
    limit: int = Field(default=10, ge=1, le=50)
    search_type: Optional[str] = None
    section_query: Optional[str] = None
    # Set only after the target document's persisted section catalog validates it.
    exact_section_path: Optional[str] = None
    evidence_group: Optional[EvidenceGroupName] = None
    artifact_types: List[ArtifactType] = Field(default_factory=list)
    document_id: Optional[str] = None
    reason: str = ""


class IntentPlan(BaseModel):
    intent: IntentType
    needs_retrieval: bool
    retrieval_steps: List[RetrievalStep] = Field(default_factory=list)
    max_tools: int = Field(default=5, ge=0, le=5)
    max_final_evidence: int = Field(default=15, ge=1, le=15)
    target_min_final_evidence: int = Field(default=6, ge=1, le=15)
    evidence_groups: List[EvidenceGroupBudget] = Field(default_factory=list)
    allow_external_sources: bool = False
    evidence_policy: str = "answer_with_available_evidence_and_state_uncertainty"
    direct_answer_allowed: bool = False
    rewrite_allowed: bool = True
    use_paper_graph: bool = False
    graph_usage_reason: str = ""
    graph_relation_types: List[GraphRelationType] = Field(default_factory=list)
    graph_direction: GraphDirection = "both"
    graph_neighbor_limit: int = Field(default=6, ge=1, le=12)
    reason: str = ""
    warnings: List[str] = Field(default_factory=list)
    source_requirements: Dict[str, Any] = Field(default_factory=dict)
    answer_policy: Dict[str, Any] = Field(default_factory=dict)


class PlannerCapabilities(BaseModel):
    local_search_enabled: bool = True
    vector_search_enabled: bool = True
    hybrid_search_enabled: bool = True
    section_search_enabled: bool = True
    artifact_search_enabled: bool = True
    openalex_search_enabled: bool = False
    web_search_enabled: bool = False
    direct_answer_enabled: bool = True
    max_tools: int = Field(default=5, ge=0, le=5)

    def available_tools(self) -> List[str]:
        tools: List[str] = []
        if self.hybrid_search_enabled:
            tools.append("hybrid_search")
        if self.vector_search_enabled:
            tools.append("vector_search")
        if self.section_search_enabled:
            tools.append("section_search")
        if self.artifact_search_enabled:
            tools.append("artifact_search")
        if self.openalex_search_enabled:
            tools.append("openalex_search")
        if self.web_search_enabled:
            tools.append("web_search")
        return tools
