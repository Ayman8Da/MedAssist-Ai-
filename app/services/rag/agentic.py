from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.schemas.chat import ChatRequest, ChatResponse, TimingBreakdown, TokenUsage
from app.services.llm.base import LLMProvider
from app.services.rag.diagnostics import build_diagnostics
from app.services.rag.generator import (
    generate_answer,
    insufficient_response,
    save_trace,
    token_usage,
)
from app.services.rag.hybrid import HybridRAGService
from app.services.retrieval.hybrid import HybridRetrievalResult, HybridRetriever
from app.services.retrieval.types import RetrievalFilters, RetrievalHit
from app.utils.ids import new_trace_id

logger = logging.getLogger(__name__)

MAX_CORRECTION_ATTEMPTS = 3

URGENT_KEYWORDS = (
    "chest pain",
    "shortness of breath",
    "severe bleeding",
    "unconscious",
    "anaphylaxis",
    "stroke",
    "sepsis",
    "suicidal",
)


class AgenticState(TypedDict, total=False):
    question: str
    request_id: str
    trace_id: str
    started: float
    steps: list[dict[str, Any]]
    sub_queries: list[str]
    retry_reason: str | None
    retrieval_attempts: int
    hits: list[RetrievalHit]
    retrieval: HybridRetrievalResult | None
    context_summary: str
    verification: dict[str, Any]
    safety_warnings: list[str]
    answer: str
    citations: list[Any]
    retrieved_chunks: list[Any]
    outcome: Any
    total_ms: float
    embedding_ms: float
    retrieval_ms: float
    generation_ms: float
    error: str | None


PLANNER_SYSTEM = (
    "You are a clinical query planner for a medical knowledge retrieval system. "
    "Given a clinician's question, produce 1 to 3 focused sub-queries that will "
    "retrieve the most relevant evidence from a medical knowledge base. "
    "Return ONLY a JSON object of the form {\"sub_queries\": [\"...\", \"...\"]}. "
    "No prose, no markdown fences."
)


def _parse_sub_queries(content: str, fallback: str) -> list[str]:
    try:
        parsed = json.loads(content.strip())
        queries = parsed.get("sub_queries") if isinstance(parsed, dict) else None
        if isinstance(queries, list) and queries:
            cleaned = [str(q).strip() for q in queries if str(q).strip()]
            if cleaned:
                return cleaned[:3]
    except (json.JSONDecodeError, AttributeError):
        pass
    # Last resort: try to extract a JSON object from the text.
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(content[start : end + 1])
            queries = parsed.get("sub_queries") if isinstance(parsed, dict) else None
            if isinstance(queries, list) and queries:
                cleaned = [str(q).strip() for q in queries if str(q).strip()]
                if cleaned:
                    return cleaned[:3]
        except (json.JSONDecodeError, AttributeError):
            pass
    return [fallback]


class AgenticRAGService:
    """LangGraph-based agentic RAG: plan -> retrieve -> verify -> (correct) -> generate."""

    def __init__(
        self,
        *,
        settings: Settings,
        retriever: HybridRetriever,
        hybrid_service: HybridRAGService,
        llm: LLMProvider,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.hybrid_service = hybrid_service
        self.llm = llm
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(AgenticState)
        graph.add_node("safety_input", self._safety_input)
        graph.add_node("planner", self._planner)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("verify", self._verify)
        graph.add_node("expand", self._expand)
        graph.add_node("evidence", self._evidence)
        graph.add_node("safety_agent", self._safety_agent)
        graph.add_node("generate", self._generate)

        graph.add_edge(START, "safety_input")
        graph.add_edge("safety_input", "planner")
        graph.add_edge("planner", "retrieve")
        graph.add_edge("retrieve", "verify")
        graph.add_conditional_edges(
            "verify",
            self._should_retry,
            {"retry": "expand", "ok": "evidence"},
        )
        graph.add_edge("expand", "planner")
        graph.add_edge("evidence", "safety_agent")
        graph.add_edge("safety_agent", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    async def _safety_input(self, state: AgenticState) -> AgenticState:
        question = state["question"].lower()
        warnings: list[str] = []
        if any(keyword in question for keyword in URGENT_KEYWORDS):
            warnings.append("urgent_symptoms_mentioned_escalate_to_in_person_assessment")
        state.setdefault("safety_warnings", []).extend(warnings)
        state["steps"].append(
            {
                "step": "safety_input",
                "agent": "safety",
                "status": "completed",
                "warnings": warnings,
            }
        )
        return state

    async def _planner(self, state: AgenticState) -> AgenticState:
        question = state["question"]
        retry_reason = state.get("retry_reason")
        attempts = state.get("retrieval_attempts", 0)
        user_prompt = f"Question: {question}\n"
        if retry_reason:
            user_prompt += (
                f"\nPrevious retrieval was insufficient ({retry_reason}). "
                "Broaden or rephrase the sub-queries to find more evidence.\n"
            )
        if attempts:
            user_prompt += f"Attempt: {attempts + 1} of {MAX_CORRECTION_ATTEMPTS}.\n"

        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.llm.generate(system_prompt=PLANNER_SYSTEM, user_prompt=user_prompt),
                timeout=self.settings.llm_timeout_seconds,
            )
            sub_queries = _parse_sub_queries(result.content, question)
        except Exception as exc:  # noqa: BLE001 - planner must never block retrieval
            logger.warning("Planner failed, falling back to original question: %s", exc)
            sub_queries = [question]
        latency = round((time.perf_counter() - started) * 1000, 2)
        state["sub_queries"] = sub_queries
        state["retry_reason"] = None
        state["steps"].append(
            {
                "step": "planner",
                "agent": "planner",
                "status": "completed",
                "latency_ms": latency,
                "sub_queries": sub_queries,
            }
        )
        return state

    async def _retrieve(self, state: AgenticState) -> AgenticState:
        filters = RetrievalFilters()  # agentic does not accept user filters yet (Phase 3 keeps it simple)
        merged: dict[str, RetrievalHit] = {}
        retrieval_result: HybridRetrievalResult | None = None
        started = time.perf_counter()
        for query in state.get("sub_queries", [state["question"]]):
            try:
                result = await self.retriever.retrieve(
                    # session is injected later; retrieval needs a session. See answer().
                    None,  # placeholder, replaced in answer() via closure
                    query=query,
                    top_k=self.settings.top_k,
                    filters=filters,
                )
            except Exception:  # noqa: BLE001
                continue
            retrieval_result = result
            for hit in result.hits:
                existing = merged.get(hit.chunk_id)
                if existing is None or (hit.score or 0) > (existing.score or 0):
                    merged[hit.chunk_id] = hit
        # Note: retriever.retrieve requires a session; the placeholder above is invalid.
        # The real call happens in answer() through _run_retrieval. This node is a stub
        # that is never reached with a None session because answer() overrides retrieval.
        latency = round((time.perf_counter() - started) * 1000, 2)
        hits = sorted(merged.values(), key=lambda h: -(h.score or 0))
        state["hits"] = hits
        state["retrieval"] = retrieval_result
        state["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1
        state["steps"].append(
            {
                "step": "hybrid_retrieve",
                "agent": "retrieval",
                "status": "completed",
                "latency_ms": latency,
                "retrieved": len(hits),
                "attempt": state["retrieval_attempts"],
                "sub_queries": state.get("sub_queries", []),
            }
        )
        return state

    async def _verify(self, state: AgenticState) -> AgenticState:
        hits = state.get("hits", [])
        attempts = state.get("retrieval_attempts", 0)
        if not hits:
            passed = False
            score = 0.0
            reasons = ["no_relevant_documents"]
        else:
            unique_docs = {h.document_id for h in hits}
            score = min(1.0, len(unique_docs) / 3.0)
            passed = len(unique_docs) >= 1
            reasons = [] if passed else ["insufficient_document_coverage"]
        state["verification"] = {"passed": passed, "score": score, "reasons": reasons}
        state["steps"].append(
            {
                "step": "verify",
                "agent": "verifier",
                "status": "completed",
                "passed": passed,
                "score": round(score, 3),
                "reasons": reasons,
                "attempt": attempts,
            }
        )
        if not passed and attempts < MAX_CORRECTION_ATTEMPTS:
            state["retry_reason"] = reasons[0] if reasons else "verification_failed"
        return state
        hits = state.get("hits", [])
        attempts = state.get("retrieval_attempts", 0)
        if hits:
            return "ok"
        if attempts < MAX_CORRECTION_ATTEMPTS:
            state["retry_reason"] = "no_relevant_documents"
            return "retry"
        return "ok"

    def _should_retry(self, state: AgenticState) -> str:
        verification = state.get("verification", {})
        passed = bool(verification.get("passed", False)) or bool(state.get("hits"))
        attempts = state.get("retrieval_attempts", 0)
        if passed:
            return "ok"
        if attempts < MAX_CORRECTION_ATTEMPTS:
            state["retry_reason"] = (verification.get("reasons") or ["verification_failed"])[0]
            return "retry"
        return "ok"

        state["steps"].append(
            {
                "step": "correction",
                "agent": "verifier",
                "status": "retrying",
                "attempt": state.get("retrieval_attempts", 0),
                "reason": state.get("retry_reason"),
            }
        )
        return state

    async def _expand(self, state: AgenticState) -> AgenticState:
        state["steps"].append(
            {
                "step": "correction",
                "agent": "verifier",
                "status": "retrying",
                "attempt": state.get("retrieval_attempts", 0),
                "reason": state.get("retry_reason"),
            }
        )
        return state

    async def _evidence(self, state: AgenticState) -> AgenticState:
        hits = state.get("hits", [])
        if not hits:
            state["context_summary"] = ""
            state["steps"].append({"step": "clinical_evidence", "agent": "evidence", "status": "completed", "retrieved": 0})
            return state
        state["context_summary"] = f"{len(hits)} chunks from {len({h.document_id for h in hits})} documents"
        state["steps"].append(
            {
                "step": "clinical_evidence",
                "agent": "evidence",
                "status": "completed",
                "retrieved": len(hits),
                "documents": len({h.document_id for h in hits}),
            }
        )
        return state

    async def _safety_agent(self, state: AgenticState) -> AgenticState:
        warnings = list(state.get("safety_warnings", []))
        hits = state.get("hits", [])
        if not hits:
            warnings.append("no_supporting_evidence")
        # Flag if all evidence is older than 5 years.
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).year  # noqa: UP017
        years = []
        for hit in hits:
            published = hit.payload.get("publication_date")
            if isinstance(published, str) and len(published) >= 4:
                try:
                    years.append(int(published[:4]))
                except ValueError:
                    pass
        if years and max(years) < now - 5:
            warnings.append("evidence_older_than_5_years")
        state["safety_warnings"] = warnings
        state["steps"].append(
            {
                "step": "safety_agent",
                "agent": "safety",
                "status": "completed",
                "warnings": warnings,
            }
        )
        return state

    async def _generate(self, state: AgenticState) -> AgenticState:
        hits = state.get("hits", [])
        started = time.perf_counter()
        outcome = await generate_answer(
            settings=self.settings,
            llm=self.llm,
            question=state["question"],
            hits=hits,
        )
        generation_ms = round((time.perf_counter() - started) * 1000, 2)
        state["outcome"] = outcome
        state["answer"] = outcome.content or ""
        state["citations"] = outcome.citations
        state["retrieved_chunks"] = outcome.retrieved_chunks
        state["generation_ms"] = generation_ms
        if outcome.insufficient:
            state["steps"].append(
                {"step": "generate_answer", "agent": "generator", "status": "completed", "latency_ms": generation_ms, "insufficient": True}
            )
        else:
            state["steps"].append(
                {"step": "generate_answer", "agent": "generator", "status": "completed", "latency_ms": generation_ms}
            )
        return state

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    async def answer(
        self, session: AsyncSession, payload: ChatRequest, *, request_id: str
    ) -> ChatResponse:
        started = time.perf_counter()
        trace_id = new_trace_id()
        steps: list[dict[str, Any]] = []

        # We need retrieval to use the session. The graph's _retrieve is a closure
        # over self but has no session. Patch it by calling retrieval directly here
        # is not possible once graph compiled. Instead, we override retrieval by
        # monkey-patching a session-bound method on self before invocation.
        self._session = session
        original_retrieve = self._retrieve

        async def retrieve_with_session(state: AgenticState) -> AgenticState:
            filters = RetrievalFilters()
            merged: dict[str, RetrievalHit] = {}
            retrieval_result = None
            r_started = time.perf_counter()
            for query in state.get("sub_queries", [state["question"]]):
                try:
                    result = await self.retriever.retrieve(
                        session,
                        query=query,
                        top_k=self.settings.top_k,
                        filters=filters,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Retrieval failed for sub-query %r: %s", query, exc)
                    continue
                retrieval_result = result
                for hit in result.hits:
                    existing = merged.get(hit.chunk_id)
                    if existing is None or (hit.score or 0) > (existing.score or 0):
                        merged[hit.chunk_id] = hit
            latency = round((time.perf_counter() - r_started) * 1000, 2)
            hits = sorted(merged.values(), key=lambda h: -(h.score or 0))
            state["hits"] = hits
            state["retrieval"] = retrieval_result
            state["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1
            state["steps"].append(
                {
                    "step": "hybrid_retrieve",
                    "agent": "retrieval",
                    "status": "completed",
                    "latency_ms": latency,
                    "retrieved": len(hits),
                    "attempt": state["retrieval_attempts"],
                    "sub_queries": state.get("sub_queries", []),
                }
            )
            return state

        # Temporarily replace the graph node function. LangGraph compiled graph
        # references the original; instead rebuild graph with the bound retrieve.
        self._retrieve = retrieve_with_session  # type: ignore[assignment]
        graph = self._build_graph()
        self._retrieve = original_retrieve

        initial_state: AgenticState = {
            "question": payload.question,
            "request_id": request_id,
            "trace_id": trace_id,
            "started": started,
            "steps": steps,
            "sub_queries": [],
            "retry_reason": None,
            "retrieval_attempts": 0,
            "hits": [],
            "retrieval": None,
            "context_summary": "",
            "verification": {},
            "safety_warnings": [],
            "answer": "",
            "citations": [],
            "retrieved_chunks": [],
            "outcome": None,
            "total_ms": 0.0,
            "embedding_ms": 0.0,
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "error": None,
        }

        try:
            final_state = await graph.ainvoke(initial_state)
        except Exception as exc:  # noqa: BLE001
            total_ms = round((time.perf_counter() - started) * 1000, 2)
            await save_trace(
                session,
                trace_id=trace_id,
                mode="agentic",
                request_id=request_id,
                question=payload.question,
                steps=steps,
                status="failed",
                latency_ms=total_ms,
                retrieval_attempts=initial_state.get("retrieval_attempts", 0),
                error=f"agentic_failed: {exc}",
            )
            raise

        steps = final_state.get("steps", steps)
        outcome = final_state.get("outcome")
        retrieval = final_state.get("retrieval")
        _ = final_state.get("hits", [])
        safety_warnings = final_state.get("safety_warnings", [])
        total_ms = round((time.perf_counter() - started) * 1000, 2)

        # Build timings from retrieval stages if available.
        embedding_ms = 0.0
        retrieval_ms = 0.0
        if retrieval is not None:
            for stage in retrieval.stages:
                if stage.name == "embed_query":
                    embedding_ms += stage.latency_ms
                else:
                    retrieval_ms += stage.latency_ms
        generation_ms = float(final_state.get("generation_ms", 0.0))

        timings = TimingBreakdown(
            embedding_ms=round(embedding_ms, 2),
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=generation_ms,
            total_ms=total_ms,
        )

        if outcome is not None and getattr(outcome, "insufficient", False):
            response = insufficient_response(
                request_id=request_id,
                system="agentic",
                timings=timings,
                warnings=[*safety_warnings, *getattr(outcome, "warnings", [])],
            )
            if retrieval is not None:
                response.diagnostics = build_diagnostics(
                    retrieval, rrf_k=self.settings.rrf_k, reranker=self.retriever.reranker
                )
            await save_trace(
                session,
                trace_id=trace_id,
                mode="agentic",
                request_id=request_id,
                question=payload.question,
                steps=steps,
                status="completed",
                latency_ms=total_ms,
                retrieval_attempts=final_state.get("retrieval_attempts", 1),
            )
            return response

        warnings = list(getattr(outcome, "warnings", []) if outcome else [])
        warnings = [*safety_warnings, *warnings]
        if retrieval is not None and not retrieval.reranker_used:
            warnings.append("reranker_skipped" if self.retriever.reranker.enabled else "reranker_disabled")

        diagnostics = None
        if retrieval is not None:
            diagnostics = build_diagnostics(
                retrieval, rrf_k=self.settings.rrf_k, reranker=self.retriever.reranker
            )

        await save_trace(
            session,
            trace_id=trace_id,
            mode="agentic",
            request_id=request_id,
            question=payload.question,
            steps=steps,
            status="completed",
            latency_ms=total_ms,
            retrieval_attempts=final_state.get("retrieval_attempts", 1),
            prompt_tokens=getattr(outcome, "prompt_tokens", None) if outcome else None,
            completion_tokens=getattr(outcome, "completion_tokens", None) if outcome else None,
        )

        return ChatResponse(
            request_id=request_id,
            system="agentic",
            answer=final_state.get("answer", ""),
            citations=getattr(outcome, "citations", []) if outcome else [],
            retrieved_chunks=getattr(outcome, "retrieved_chunks", []) if outcome else [],
            insufficient_evidence=False,
            timings=timings,
            usage=token_usage(outcome) if outcome else TokenUsage(),
            retrieval_attempts=final_state.get("retrieval_attempts", 1),
            model=getattr(outcome, "model", None) if outcome else None,
            warnings=warnings,
            diagnostics=diagnostics,
        )
