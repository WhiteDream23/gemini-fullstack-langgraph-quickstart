from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


import operator


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str
    # RAG 相关状态
    rag_result: str
    rag_sufficient: bool
    rag_confidence: float
    use_local_knowledge: bool
    # 意图澄清和研究简报状态
    need_clarification: bool
    clarification_question: str
    research_brief: str
    supervisor_messages: Annotated[list, add_messages]
    #  保存文件相关
    awaiting_user_decision: bool  # 是否等待用户决定
    save_permission: str  # 用户的保存决定: "save" | "skip" | None
    save_result: str  # 保存结果


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int


class RAGState(TypedDict):
    """RAG 检索状态"""
    rag_result: str
    rag_sufficient: bool
    rag_confidence: float
    evaluation_reason: str


class ClarificationState(TypedDict):
    """意图澄清状态"""
    need_clarification: bool
    question: str
    verification: str
    clarity_score: float


class ResearchBriefState(TypedDict):
    """研究简报状态"""
    research_brief: str
    research_scope: str
    key_questions: list[str]
    expected_sources: list[str]


class Query(TypedDict):
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    search_query: list[Query]


class WebSearchState(TypedDict):
    search_query: str
    id: str


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report
