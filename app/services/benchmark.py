from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import BenchmarkRun
from app.db.session import Database
from app.schemas.chat import ChatRequest
from app.services.llm.base import LLMProvider
from app.services.rag.agentic import AgenticRAGService
from app.services.rag.traditional import TraditionalRAGService

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "sample_documents" / "manifest.json"

FAITHFULNESS_PROMPT = (
    "You are an evaluator. Given a question, an answer, and the retrieved context, "
    "score how faithful the answer is to the context (no unsupported claims). "
    "Return ONLY JSON: {\"score\": number between 0 and 1, \"reason\": \"short\"}."
)
RELEVANCE_PROMPT = (
    "You are an evaluator. Score how well the answer addresses the question. "
    "Return ONLY JSON: {\"score\": number between 0 and 1, \"reason\": \"short\"}."
)
COHERENCE_PROMPT = (
    "You are an evaluator. Score the coherence and clarity of the answer. "
    "Return ONLY JSON: {\"score\": number between 0 and 1, \"reason\": \"short\"}."
)
CONTEXT_PRECISION_PROMPT = (
    "You are an evaluator. Given a question and a list of retrieved chunks, "
    "decide which chunks are relevant. Return ONLY JSON: "
    "{\"relevant_indices\": [1-based indices of relevant chunks], \"reason\": \"short\"}."
)


def _load_questions() -> list[str]:
    if not MANIFEST_PATH.exists():
        return []
    import json as _json

    manifest = _json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return list(manifest.get("benchmark_questions", []))


def _parse_json(content: str) -> dict[str, Any] | None:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


class BenchmarkRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        traditional: TraditionalRAGService,
        agentic: AgenticRAGService,
        llm: LLMProvider,
    ) -> None:
        self.settings = settings
        self.db = db
        self.traditional = traditional
        self.agentic = agentic
        self.llm = llm

    async def start(self, session: AsyncSession) -> str:
        questions = _load_questions()
        run_id = f"bench_{uuid4().hex[:12]}"
        run = BenchmarkRun(
            id=run_id,
            status="pending",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            total_questions=len(questions),
            completed_questions=0,
        )
        session.add(run)
        await session.commit()
        asyncio.create_task(self._execute(run_id))
        return run_id

    async def _execute(self, run_id: str) -> None:
        async with self.db.session_factory() as session:
            run = await session.get(BenchmarkRun, run_id)
            if run is None:
                return
            run.status = "running"
            run.updated_at = datetime.now(UTC)
            await session.commit()

            questions = _load_questions()
            results: list[dict[str, Any]] = []
            try:
                for index, question in enumerate(questions):
                    q_result = await self._run_question(session, question, index)
                    results.append(q_result)
                    run.completed_questions = index + 1
                    run.updated_at = datetime.now(UTC)
                    await session.commit()

                summary = self._summarize(results)
                run.results_json = json.dumps(results)
                run.summary_json = json.dumps(summary)
                run.status = "completed"
                run.updated_at = datetime.now(UTC)
                await session.commit()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Benchmark run %s failed", run_id)
                run.status = "failed"
                run.error = str(exc)[:500]
                run.updated_at = datetime.now(UTC)
                await session.commit()

    async def _run_question(self, session: AsyncSession, question: str, index: int) -> dict[str, Any]:
        modes = {}
        for mode_name, service in (("traditional", self.traditional), ("agentic", self.agentic)):
            try:
                response = await service.answer(
                    session, ChatRequest(question=question), request_id=f"bench-{index}-{mode_name}"
                )
            except Exception as exc:  # noqa: BLE001
                modes[mode_name] = {"error": str(exc)}
                continue
            latency = response.timings.total_ms
            answer = response.answer
            citations = [c.model_dump() for c in response.citations]
            retrieved = [c.model_dump() for c in response.retrieved_chunks]
            context_text = "\n\n".join(c.get("text", "") for c in retrieved) or ""
            metrics = await self._evaluate(question, answer, retrieved, context_text)
            modes[mode_name] = {
                "answer": answer,
                "citations": citations,
                "retrieved_chunks": retrieved,
                "insufficient_evidence": response.insufficient_evidence,
                "timings": response.timings.model_dump(),
                "usage": response.usage.model_dump(),
                "warnings": response.warnings,
                "diagnostics": response.diagnostics.model_dump() if response.diagnostics else None,
                "metrics": metrics,
                "latency_ms": latency,
            }

        return {"index": index, "question": question, "modes": modes}

    async def _evaluate(
        self, question: str, answer: str, retrieved: list[dict[str, Any]], context_text: str
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        # Citation accuracy: fraction of citations marked used.
        # (computed from response elsewhere; here approximate from retrieved count)
        metrics["citation_accuracy"] = None

        # Context precision: ask LLM which retrieved chunks are relevant.
        if retrieved:
            chunks_text = "\n".join(f"[{i + 1}] {c.get('text', '')[:400]}" for i, c in enumerate(retrieved))
            prompt = (
                f"Question: {question}\n\nRetrieved chunks:\n{chunks_text}\n\n"
                "Return ONLY JSON: {\"relevant_indices\": [1-based], \"reason\": \"short\"}."
            )
            try:
                res = await self.llm.generate(system_prompt=CONTEXT_PRECISION_PROMPT, user_prompt=prompt)
                parsed = _parse_json(res.content) or {}
                relevant = parsed.get("relevant_indices", [])
                if isinstance(relevant, list) and retrieved:
                    metrics["context_precision"] = round(len(set(relevant)) / len(retrieved), 4)
                else:
                    metrics["context_precision"] = None
            except Exception:  # noqa: BLE001
                metrics["context_precision"] = None
        else:
            metrics["context_precision"] = None

        # Faithfulness
        metrics["faithfulness"] = await self._judge(FAITHFULNESS_PROMPT, question, answer, context_text)
        # Answer relevance
        metrics["answer_relevance"] = await self._judge(RELEVANCE_PROMPT, question, answer, context_text)
        # Coherence
        metrics["coherence"] = await self._judge(COHERENCE_PROMPT, question, answer, context_text)

        # Context recall requires a gold reference; not available -> null.
        metrics["context_recall"] = None
        return metrics

    async def _judge(self, system_prompt: str, question: str, answer: str, context_text: str) -> float | None:
        prompt = f"Question: {question}\n\nAnswer: {answer}\n\nContext: {context_text[:6000]}"
        try:
            res = await self.llm.generate(system_prompt=system_prompt, user_prompt=prompt)
            parsed = _parse_json(res.content)
            if parsed and isinstance(parsed.get("score"), (int, float)):
                score = float(parsed["score"])
                return round(min(1.0, max(0.0, score)), 4)
        except Exception:  # noqa: BLE001
            pass
        return None

    def _summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {"modes": {}}
        for mode in ("traditional", "agentic"):
            metrics_list = [r["modes"].get(mode, {}).get("metrics", {}) for r in results if mode in r.get("modes", {})]
            latencies = [r["modes"].get(mode, {}).get("latency_ms") for r in results if mode in r.get("modes", {})]
            def avg(metrics: list[dict[str, Any]], key: str) -> float | None:
                vals = [m.get(key) for m in metrics if isinstance(m.get(key), (int, float))]
                return round(sum(vals) / len(vals), 4) if vals else None

            summary["modes"][mode] = {
                "questions": len(metrics_list),
                "avg_faithfulness": avg(metrics_list, "faithfulness"),
                "avg_answer_relevance": avg(metrics_list, "answer_relevance"),
                "avg_context_precision": avg(metrics_list, "context_precision"),
                "avg_coherence": avg(metrics_list, "coherence"),
                "avg_latency_ms": round(sum(x for x in latencies if isinstance(x, (int, float))) / len([x for x in latencies if isinstance(x, (int, float))]), 2) if any(isinstance(x, (int, float)) for x in latencies) else None,
            }
        return summary

    async def get_latest(self, session: AsyncSession) -> BenchmarkRun | None:
        result = await session.execute(
            select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc()).limit(1)
        )
        return result.scalar_one_or_none()

    async def get(self, session: AsyncSession, run_id: str) -> BenchmarkRun | None:
        return await session.get(BenchmarkRun, run_id)

    async def list_runs(self, session: AsyncSession, limit: int = 20) -> list[BenchmarkRun]:
        result = await session.execute(
            select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
