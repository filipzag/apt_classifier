"""FastAPI application: JSON API over ``CaseService`` plus the built React front end.

Run with ``make api`` (development, auto-reload) or ``make run-next`` (builds the web app and
serves everything from one process on port 8100). Every LLM-backed operation is a plain
request that blocks until the model answers; FastAPI runs the synchronous handlers in a thread
pool so the API stays responsive while a review runs.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api import serializers
from apt_classifier.report import EXPORT_KINDS
from apt_classifier.service import (
    CaseNotFound,
    CaseService,
    IntakePayload,
    ServiceUnavailable,
    UploadedBlob,
)

log = logging.getLogger("apt_classifier.api")
WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
MAX_UPLOAD_BYTES = 16 * 1024 * 1024


# ----- request models -----
class DecisionIn(BaseModel):
    observation_id: str
    technique_id: str
    action: str
    replacement: str | None = None
    rationale: str | None = None


class DecisionsIn(BaseModel):
    decisions: list[DecisionIn]


class AnswerIn(BaseModel):
    status: str
    text: str | None = None


class RenameIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class WeightsIn(BaseModel):
    weights: dict[str, float]


class CyberREIn(BaseModel):
    observation_id: str | None = None
    use_prefilter: bool | None = None


class ElasticSearchIn(BaseModel):
    lookback_hours: int | None = None


class ElasticImportIn(BaseModel):
    candidate_ids: list[str]


def describe_error(exc: BaseException) -> str:
    text = str(exc) or type(exc).__name__
    if "402" in text or "Payment Required" in text:
        return "OpenRouter refused the request: the account has no credit left (402)."
    if "401" in text and "openrouter" in text.lower():
        return "OpenRouter rejected the API key (401)."
    if "Connection refused" in text or "ConnectError" in type(exc).__name__:
        return f"A backing service is unreachable: {text[:200]}"
    return f"{type(exc).__name__}: {text[:300]}"


def get_service(request: Request) -> CaseService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(503, "the analysis service is still starting")
    return service


def create_app(service: CaseService | None = None, *, serve_web: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "service", None) is None:
            log.info("loading the knowledge base, embedder and vector store")
            app.state.service = CaseService()
        yield

    app = FastAPI(title="APT Attribution Assistant API", version="2.0", lifespan=lifespan)
    app.state.service = service

    @app.exception_handler(CaseNotFound)
    async def _not_found(_: Request, exc: CaseNotFound) -> JSONResponse:
        return JSONResponse({"detail": f"case {exc.args[0]} was not found"}, status_code=404)

    @app.exception_handler(ValueError)
    async def _bad_request(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(ServiceUnavailable)
    async def _unavailable(_: Request, exc: ServiceUnavailable) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(Exception)
    async def _failure(_: Request, exc: Exception) -> JSONResponse:
        log.exception("request failed")
        return JSONResponse({"detail": describe_error(exc)}, status_code=500)

    # ----- system -----
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/system")
    def system(svc: CaseService = Depends(get_service)) -> dict[str, Any]:
        return svc.system_info()

    @app.get("/api/sample")
    def sample(svc: CaseService = Depends(get_service)) -> dict[str, str]:
        return {"text": svc.sample_text()}

    @app.get("/api/techniques")
    def technique_search(
        q: str = "", limit: int = 20, svc: CaseService = Depends(get_service)
    ) -> list[dict[str, str]]:
        return svc.search_techniques(q, limit=min(max(limit, 1), 50))

    # ----- cases -----
    def _case(svc: CaseService, case_id: str) -> dict[str, Any]:
        state, meta = svc.get(case_id)
        return serializers.case_payload(
            svc.engine.kb, state, meta, svc.engine.attribution_thresholds, svc.summary_for(state)
        )

    @app.get("/api/cases")
    def list_cases(svc: CaseService = Depends(get_service)) -> list[dict[str, Any]]:
        return [
            {
                "case_id": r.case_id,
                "name": r.name,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
                "summary": {
                    "observations": r.observations,
                    "stage": r.stage,
                    "leading_group": r.leading_group,
                    "leading_group_name": (
                        serializers._name(svc.engine.kb, r.leading_group)  # noqa: SLF001
                        if r.leading_group
                        else None
                    ),
                },
            }
            for r in svc.list_cases()
        ]

    @app.post("/api/cases", status_code=201)
    async def create_case(
        name: str = Form(""),
        narrative: str = Form(""),
        phases: str = Form("{}"),
        alerts: str = Form(""),
        iocs: str = Form(""),
        target_geographies: str = Form(""),
        target_industries: str = Form(""),
        motivations: str = Form(""),
        objectives: str = Form(""),
        suspected_origins: str = Form(""),
        files: list[UploadFile] = File(default=[]),
        svc: CaseService = Depends(get_service),
    ) -> dict[str, Any]:
        try:
            phase_map = json.loads(phases or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(400, "phases must be a JSON object") from exc
        blobs = []
        for upload in files:
            data = await upload.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                data = data[:MAX_UPLOAD_BYTES]
            if data:
                blobs.append(UploadedBlob(upload.filename or "upload", data))
        payload = IntakePayload(
            name=name,
            narrative=narrative,
            phases={str(k): str(v) for k, v in dict(phase_map).items()},
            alerts=alerts,
            iocs=iocs,
            files=blobs,
            target_geographies=target_geographies,
            target_industries=target_industries,
            motivations=motivations,
            objectives=objectives,
            suspected_origins=suspected_origins,
        )
        # the analysis is CPU/LLM bound: run it off the event loop
        from starlette.concurrency import run_in_threadpool

        case_id = await run_in_threadpool(svc.create_case, payload)
        return await run_in_threadpool(_case, svc, case_id)

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str, svc: CaseService = Depends(get_service)) -> dict[str, Any]:
        return _case(svc, case_id)

    @app.patch("/api/cases/{case_id}")
    def rename_case(
        case_id: str, body: RenameIn, svc: CaseService = Depends(get_service)
    ) -> dict[str, Any]:
        meta = svc.rename(case_id, body.name)
        return {"case_id": meta.case_id, "name": meta.name, "updated_at": meta.updated_at}

    @app.delete("/api/cases/{case_id}", status_code=204)
    def delete_case(case_id: str, svc: CaseService = Depends(get_service)) -> Response:
        if not svc.delete(case_id):
            raise HTTPException(404, f"case {case_id} was not found")
        return Response(status_code=204)

    @app.post("/api/cases/{case_id}/decisions")
    def apply_decisions(
        case_id: str, body: DecisionsIn, svc: CaseService = Depends(get_service)
    ) -> dict[str, Any]:
        svc.apply_decisions(case_id, [d.model_dump() for d in body.decisions])
        return _case(svc, case_id)

    @app.post("/api/cases/{case_id}/weights")
    def rescore(
        case_id: str, body: WeightsIn, svc: CaseService = Depends(get_service)
    ) -> dict[str, Any]:
        svc.rescore(case_id, body.weights)
        return _case(svc, case_id)

    # ----- interview -----
    @app.get("/api/cases/{case_id}/interview")
    def interview(case_id: str, svc: CaseService = Depends(get_service)) -> dict[str, Any]:
        return svc.interview(case_id)

    @app.post("/api/cases/{case_id}/interview/answer")
    def answer(
        case_id: str, body: AnswerIn, svc: CaseService = Depends(get_service)
    ) -> dict[str, Any]:
        svc.answer(case_id, body.status, body.text)
        return svc.interview(case_id)

    @app.post("/api/cases/{case_id}/interview/restart")
    def restart(case_id: str, svc: CaseService = Depends(get_service)) -> dict[str, Any]:
        svc.restart_interview(case_id)
        return svc.interview(case_id)

    @app.post("/api/cases/{case_id}/elastic/search")
    def elastic_search(
        case_id: str, body: ElasticSearchIn, svc: CaseService = Depends(get_service)
    ) -> dict[str, Any]:
        batch = svc.elastic_search(case_id, body.lookback_hours)
        return {
            "ok": bool(batch.ok),
            "error": getattr(batch, "error", None),
            "esql_query": getattr(batch, "esql_query", None),
            "candidates": [
                {
                    "id": c.id,
                    "timestamp": getattr(c, "timestamp", None),
                    "index": getattr(c, "index", None),
                    "event_id": getattr(c, "event_id", None),
                    "text": c.text[:400],
                }
                for c in (batch.candidates or ())
            ],
        }

    @app.post("/api/cases/{case_id}/elastic/import")
    def elastic_import(
        case_id: str, body: ElasticImportIn, svc: CaseService = Depends(get_service)
    ) -> dict[str, Any]:
        svc.elastic_import(case_id, body.candidate_ids)
        return _case(svc, case_id)

    # ----- LLM steps -----
    @app.post("/api/cases/{case_id}/review")
    def run_review(case_id: str, svc: CaseService = Depends(get_service)) -> dict[str, Any]:
        svc.run_review(case_id)
        return _case(svc, case_id)

    @app.post("/api/cases/{case_id}/zero-shot")
    def run_zero_shot(case_id: str, svc: CaseService = Depends(get_service)) -> dict[str, Any]:
        svc.run_zero_shot(case_id)
        return _case(svc, case_id)

    @app.post("/api/cases/{case_id}/cyberre")
    def run_cyberre(
        case_id: str, body: CyberREIn, svc: CaseService = Depends(get_service)
    ) -> dict[str, Any]:
        svc.run_cyberre(case_id, body.observation_id, use_prefilter=body.use_prefilter)
        return _case(svc, case_id)

    # ----- views -----
    @app.get("/api/cases/{case_id}/dossier/{group_id}")
    def dossier(
        case_id: str, group_id: str, svc: CaseService = Depends(get_service)
    ) -> dict[str, Any]:
        payload = svc.dossier(case_id, group_id.upper())
        payload["group_id"] = group_id.upper()
        payload["name"] = serializers._name(svc.engine.kb, group_id.upper())  # noqa: SLF001
        return payload

    @app.get("/api/cases/{case_id}/canvas")
    def canvas(
        case_id: str,
        top_groups: int = 8,
        max_evidence: int = 12,
        svc: CaseService = Depends(get_service),
    ) -> dict[str, Any]:
        return serializers.graph(
            svc.canvas(case_id, top_groups=top_groups, max_evidence=max_evidence)
        )

    @app.get("/api/cases/{case_id}/diamond")
    def diamond(
        case_id: str,
        top_groups: int = 8,
        max_capabilities: int = 15,
        max_infrastructure: int = 15,
        svc: CaseService = Depends(get_service),
    ) -> dict[str, Any]:
        return serializers.graph(
            svc.diamond(
                case_id,
                top_groups=top_groups,
                max_capabilities=max_capabilities,
                max_infrastructure=max_infrastructure,
            )
        )

    @app.get("/api/cases/{case_id}/export/{kind}")
    def export(case_id: str, kind: str, svc: CaseService = Depends(get_service)) -> Response:
        if kind not in EXPORT_KINDS:
            raise HTTPException(404, f"unknown export {kind!r}")
        content, media_type, filename = svc.export(case_id, kind)
        return Response(
            content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # ----- live threat intelligence -----
    @app.get("/api/cases/{case_id}/intel")
    def case_intel(
        case_id: str,
        days: int | None = None,
        claims: bool | None = None,
        refresh: bool = False,
        svc: CaseService = Depends(get_service),
    ) -> dict[str, Any]:
        snapshot = svc.fetch_intel(days=days, include_claims=claims, refresh=refresh)
        state = svc.attach_intel(case_id, snapshot)
        matches = serializers.intel(state)["matches"]
        scores = {m["item_id"]: m["score"] for m in matches}
        return {
            "snapshot": serializers.snapshot_summary(snapshot),
            "matches": matches,
            "items": serializers.snapshot_items(snapshot, scores),
        }

    # ----- the web app -----
    if serve_web and WEB_DIST.is_dir():
        assets = WEB_DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> Response:
            if path.startswith("api/"):
                raise HTTPException(404)
            candidate = WEB_DIST / path
            if path and candidate.is_file():
                return FileResponse(str(candidate))
            return FileResponse(str(WEB_DIST / "index.html"))

    return app
