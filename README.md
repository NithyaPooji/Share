config.py

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "JHARNA AI"
    SECRET_KEY: str = "jharna-ai-dev-secret-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 h

    # LLM
    MODEL_NEEV: str = "gemini-2.5-flash"

    # ── Primary LLM – custom OpenAI-compatible endpoint ───────────────────────
    # When set, this is tried FIRST; LiteLLM proxy/Google is the fallback.
    LLM_API_BASE_URL: Optional[str] = None   # e.g. http://localhost:7000/v1
    LLM_API_KEY: Optional[str] = None        # API key for the above endpoint
    LLM_MODEL_NAME: Optional[str] = None     # model served at that endpoint (e.g. gpt-4, llama-3)

    # ── LiteLLM proxy / Google (fallback) ─────────────────────────────────────
    # Google direct API (used when USE_LITELLM_PROXY=False)
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_GENAI_USE_VERTEXAI: int = 0  # 0 = AI Studio, 1 = Vertex AI

    # LiteLLM proxy (used when USE_LITELLM_PROXY=True)
    USE_LITELLM_PROXY: bool = True
    LITELLM_PROXY_API_BASE: Optional[str] = None
    LITELLM_PROXY_API_KEY: Optional[str] = None

    # SSL – single flag covering both proxy and direct calls
    SSL_VERIFY: str = "true"

    @property
    def ssl_verify(self) -> bool:
        return self.SSL_VERIFY.lower() != "false"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./jharna_ai.db"

    # Storage directories
    WORKSPACE_DIR: str = "./workspace"
    UPLOAD_DIR: str = "./uploads"

    # Git
    GITHUB_TOKEN: Optional[str] = None
    GITLAB_TOKEN: Optional[str] = None

    # JIRA (placeholder)
    JIRA_URL: Optional[str] = None
    JIRA_EMAIL: Optional[str] = None
    JIRA_API_TOKEN: Optional[str] = None

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


settings = Settings()


test.py


"""
Test generation router.
Streams progress via WebSocket; REST endpoints for CRUD + push.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.database import get_db
from db.models import GeneratedTest, Project, UploadedDoc, User
from models.schemas import (
    GeneratedTestOut,
    ProgressEvent,
    PushTestsRequest,
    TestGenerateRequest,
)
from routers.auth import get_current_user
from routers.projects import _get_project_or_404
from services.git_service import build_diff, commit_tests, open_pull_request, push_branch
from services.test_runner import run_test_file
from services.coverage_runner import run_project_coverage
from utils.llm_client import (
    MODEL_NAME, USE_PROXY, LITELLM_BASE, LITELLM_KEY,
    GOOGLE_API_KEY, SSL_VERIFY,
    PRIMARY_API_BASE, PRIMARY_API_KEY, PRIMARY_MODEL,
    build_test_prompt, trim_code_for_context,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/tests", tags=["tests"])

EXT_LANG = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".java": "Java",
}
DEFAULT_FW = {
    "Python": "pytest", "JavaScript": "Jest",
    "TypeScript": "Jest", "Java": "JUnit",
}


# ── REST: generate tests ──────────────────────────────────────────────────────

@router.post("/generate", response_model=list[GeneratedTestOut])
async def generate_tests(
    body: TestGenerateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = await _get_project_or_404(body.project_id, current_user.id, db)
    if not project.local_path:
        raise HTTPException(400, "Project has no local path")

    # Gather doc context (capped for token control)
    doc_context: Optional[str] = None
    if body.doc_ids:
        docs_result = await db.execute(
            select(UploadedDoc).where(UploadedDoc.id.in_(body.doc_ids))
        )
        texts = [d.parsed_text or "" for d in docs_result.scalars().all()]
        doc_context = "\n\n".join(texts)[:2000]

    tech_stack   = project.tech_stack or {}
    test_fw_list = tech_stack.get("test_frameworks", [])

    results: list[GeneratedTestOut] = []
    for rel_path in body.file_paths:
        full_path = str(Path(project.local_path) / rel_path)
        try:
            test_content, test_file_rel = await _generate_test_for_file(
                full_path, rel_path, project.local_path, tech_stack, test_fw_list, doc_context
            )
        except Exception as exc:
            log.exception("Test generation failed for %s", rel_path)
            raise HTTPException(500, f"Test generation failed for '{rel_path}': {exc}") from exc

        diff = build_diff(project.local_path, test_file_rel, test_content)
        record = GeneratedTest(
            project_id=project.id,
            source_file=rel_path,
            test_file=test_file_rel,
            test_content=test_content,
            diff=diff,
            status="generated",
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        results.append(GeneratedTestOut.model_validate(record))

    return results


@router.post("/coverage-report/{project_id}", response_model=dict)
async def generate_coverage_report(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Run the full test suite with coverage enabled and return a structured report.
    Supports: Python (pytest-cov), JavaScript/TypeScript (Jest/Istanbul), Java (JaCoCo).
    """
    project = await _get_project_or_404(project_id, current_user.id, db)
    if not project.local_path:
        raise HTTPException(400, "Project has no local path – cannot run coverage")

    tech_stack = project.tech_stack or {}

    # Offload blocking subprocess to thread pool (can take minutes)
    report = await asyncio.to_thread(
        run_project_coverage,
        project.local_path,
        tech_stack,
    )
    return report


@router.post("/{test_id}/run", response_model=dict)
async def run_test(
    test_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Execute the generated test file and return pass/fail results."""
    result = await db.execute(select(GeneratedTest).where(GeneratedTest.id == test_id))
    test = result.scalar_one_or_none()
    if not test:
        raise HTTPException(404, "Test not found")

    project = await _get_project_or_404(test.project_id, current_user.id, db)
    if not project.local_path:
        raise HTTPException(400, "Project has no local path – cannot execute tests")

    tech_stack = project.tech_stack or {}

    # run_test_file is blocking (subprocess); offload to thread pool
    run_result = await asyncio.to_thread(
        run_test_file,
        project.local_path,
        test.test_file,
        test.test_content,
        tech_stack,
    )
    return run_result


@router.get("/{project_id}", response_model=list[GeneratedTestOut])
async def list_generated_tests(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await _get_project_or_404(project_id, current_user.id, db)
    result = await db.execute(
        select(GeneratedTest)
        .where(GeneratedTest.project_id == project_id)
        .order_by(GeneratedTest.created_at.desc())
    )
    return [GeneratedTestOut.model_validate(t) for t in result.scalars().all()]


@router.post("/push", response_model=dict)
async def push_tests_as_pr(
    body: PushTestsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tests_result = await db.execute(
        select(GeneratedTest).where(GeneratedTest.id.in_(body.test_ids))
    )
    tests = tests_result.scalars().all()
    if not tests:
        raise HTTPException(404, "No tests found with given IDs")

    project = await _get_project_or_404(tests[0].project_id, current_user.id, db)
    test_files = {t.test_file: t.test_content for t in tests}

    # Step 1 — commit files locally (works for both git and local projects)
    try:
        commit_sha = commit_tests(project.local_path, test_files, body.pr_title)
    except Exception as exc:
        raise HTTPException(500, f"Failed to commit test files: {exc}") from exc

    # Step 2 — push branch (silently skipped if no remote)
    try:
        push_branch(project.local_path)
    except Exception as exc:
        log.warning("push_branch failed (local project?): %s", exc)
        # Don't abort — files are committed locally, just can't push

    # Step 3 — open PR (may raise ValueError with a user-readable message)
    pr_url: str | None = None
    try:
        pr_url = open_pull_request(
            git_url=project.source_path,
            head_branch=project.branch_name,
            base_branch=body.target_branch,
            title=body.pr_title,
            body=body.pr_body or _default_pr_body(tests),
            git_token=current_user.github_token or None,  # per-user token
            local_path=project.local_path,
        )
    except ValueError as exc:
        # Informative failure — tests are already committed locally
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        log.exception("open_pull_request failed")
        raise HTTPException(500, f"PR creation failed: {exc}") from exc

    for t in tests:
        t.pr_url = pr_url
        t.status = "pushed"
    await db.commit()
    return {"pr_url": pr_url, "commit_sha": commit_sha}


# ── WebSocket: streaming progress ─────────────────────────────────────────────

@router.websocket("/ws/generate/{project_id}")
async def ws_generate_tests(websocket: WebSocket, project_id: int):
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        payload = json.loads(data)
        file_paths: list[str] = payload.get("file_paths", [])
        total = len(file_paths)
        for idx, rel_path in enumerate(file_paths):
            await websocket.send_text(ProgressEvent(
                event="progress", step="generating",
                message=f"Generating tests for {rel_path}",
                percent=int((idx / total) * 100),
            ).model_dump_json())
            await asyncio.sleep(0.05)
        await websocket.send_text(ProgressEvent(
            event="done", step="complete",
            message="All tests generated successfully", percent=100,
        ).model_dump_json())
    except WebSocketDisconnect:
        pass


# ── Core generation helper ────────────────────────────────────────────────────

async def _generate_test_for_file(
    full_path: str,
    rel_path: str,
    project_path: str,
    tech_stack: dict,
    test_fw_list: list[dict],
    doc_context: Optional[str],
) -> tuple[str, str]:
    """
    Call LiteLLM directly (no ADK runner overhead) to generate tests.
    Applies token-reduction trimming before sending to the LLM.
    """
    import litellm

    # ── Resolve language + framework ──────────────────────────────────────────
    ext      = Path(rel_path).suffix.lower()
    language = EXT_LANG.get(ext, "Python")
    framework = next(
        (f["name"] for f in test_fw_list
         if language.lower() in f.get("language", "").lower()),
        DEFAULT_FW.get(language, "pytest"),
    )

    # ── Read + trim source (token budget) ────────────────────────────────────
    try:
        source = Path(full_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RuntimeError(f"Cannot read source file: {exc}") from exc

    trimmed_source = trim_code_for_context(source, max_lines=150)

    # ── Build compact prompt ──────────────────────────────────────────────────
    prompt = build_test_prompt(
        language=language,
        framework=framework,
        source_file=rel_path,
        code_snippet=trimmed_source,
        doc_context=doc_context,
    )

    messages = [{"role": "user", "content": prompt}]
    response = None

    # ── 1. Primary: custom OpenAI-compatible endpoint ─────────────────────────
    if PRIMARY_API_BASE:
        primary_kwargs: dict = {
            "model":    PRIMARY_MODEL,
            "messages": messages,
            "api_base": PRIMARY_API_BASE,
            "api_key":  PRIMARY_API_KEY or "none",  # some servers require a non-empty key
        }
        if not SSL_VERIFY:
            primary_kwargs["ssl_verify"] = False
        try:
            log.info("Primary LLM: %s @ %s — %s/%s/%s",
                     PRIMARY_MODEL, PRIMARY_API_BASE, language, framework, rel_path)
            response = await litellm.acompletion(**primary_kwargs)
        except Exception as exc:
            log.warning("Primary LLM failed (%s) — falling back to LiteLLM for %s",
                        exc, rel_path)

    # ── 2. Fallback: LiteLLM proxy or Google direct ───────────────────────────
    if response is None:
        fallback_kwargs: dict = {
            "model":    MODEL_NAME,
            "messages": messages,
        }
        if USE_PROXY:
            if LITELLM_BASE:
                fallback_kwargs["api_base"] = LITELLM_BASE
            if LITELLM_KEY:
                fallback_kwargs["api_key"] = LITELLM_KEY
        else:
            if GOOGLE_API_KEY:
                fallback_kwargs["api_key"] = GOOGLE_API_KEY
        if not SSL_VERIFY:
            fallback_kwargs["ssl_verify"] = False

        log.info("Fallback LLM: %s — %s/%s/%s", MODEL_NAME, language, framework, rel_path)
        response = await litellm.acompletion(**fallback_kwargs)

    raw_content: str = response.choices[0].message.content or ""

    # ── Strip markdown code fences the LLM may add ───────────────────────────
    test_content = _strip_code_fences(raw_content).strip()
    test_file_rel = _infer_test_path(rel_path, language)

    return test_content, test_file_rel


# ── Utilities ─────────────────────────────────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    """Remove ```python / ``` wrappers that LLMs commonly add."""
    text = re.sub(r"^```[a-zA-Z]*\n", "", text.strip())
    text = re.sub(r"\n```$", "", text)
    return text


def _infer_test_path(source_rel: str, language: str) -> str:
    p = Path(source_rel)
    if language == "Python":
        return str(p.parent / f"test_{p.stem}.py")
    if language in ("JavaScript", "TypeScript"):
        return str(p.parent / f"{p.stem}.test{p.suffix}")
    if language == "Java":
        return str(p.parent / f"{p.stem}Test.java")
    return str(p.parent / f"test_{p.stem}{p.suffix}")


def _default_pr_body(tests) -> str:
    files = "\n".join(f"- `{t.test_file}`" for t in tests)
    return (
        "## JHARNA AI – Generated Unit Tests\n\n"
        "This PR was automatically created by JHARNA AI.\n\n"
        f"### Files added:\n{files}\n\n"
        "_Please review the generated tests before merging._"
    )


llmclient

"""
Centralised LLM client configuration.

Call order for test generation:
  1. Primary   – custom OpenAI-compatible endpoint (LLM_API_BASE_URL + LLM_API_KEY)
  2. Fallback  – LiteLLM proxy  (USE_LITELLM_PROXY=True, LITELLM_PROXY_API_BASE/KEY)
               – Google direct  (USE_LITELLM_PROXY=False, GOOGLE_API_KEY)

Set only LLM_API_BASE_URL to enable the primary endpoint; the fallback is
always available as long as LITELLM_PROXY_API_BASE or GOOGLE_API_KEY is set.
"""
import os
from functools import lru_cache
from google.adk.models.lite_llm import LiteLlm

# ── Fallback (LiteLLM proxy / Google) ─────────────────────────────────────────
MODEL_NAME    = os.getenv("MODEL_NEEV", "gemini-2.5-flash")
USE_PROXY     = os.getenv("USE_LITELLM_PROXY", "true").lower() == "true"
LITELLM_BASE  = os.getenv("LITELLM_PROXY_API_BASE")
LITELLM_KEY   = os.getenv("LITELLM_PROXY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SSL_VERIFY    = os.getenv("SSL_VERIFY", "true").lower() != "false"

# ── Primary (custom OpenAI-compatible endpoint) ────────────────────────────────
PRIMARY_API_BASE  = os.getenv("LLM_API_BASE_URL")       # e.g. http://localhost:7000/v1
PRIMARY_API_KEY   = os.getenv("LLM_API_KEY")            # e.g. ghbjn
PRIMARY_MODEL_RAW = os.getenv("LLM_MODEL_NAME")         # model name at that endpoint

# LiteLLM needs a provider prefix when no known host is detected. For custom
# OpenAI-compatible endpoints we prepend "openai/" unless the user already
# included a "/" (e.g. "openai/gpt-4", "anthropic/claude-3").
if PRIMARY_MODEL_RAW and "/" not in PRIMARY_MODEL_RAW:
    PRIMARY_MODEL = f"openai/{PRIMARY_MODEL_RAW}"
else:
    PRIMARY_MODEL = PRIMARY_MODEL_RAW or f"openai/{MODEL_NAME}"

# Tell Google ADK not to use Vertex AI (use AI Studio key instead)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "0"))


@lru_cache(maxsize=1)
def get_llm() -> LiteLlm:
    """
    Return a shared LiteLlm instance (cached for the process lifetime).

    If USE_LITELLM_PROXY=True  → uses proxy base URL + key.
    If USE_LITELLM_PROXY=False → uses GOOGLE_API_KEY directly.
    """
    kwargs: dict = {"model": MODEL_NAME, "ssl_verify": SSL_VERIFY}

    if USE_PROXY:
        if LITELLM_BASE:
            kwargs["api_base"] = LITELLM_BASE
        if LITELLM_KEY:
            kwargs["api_key"] = LITELLM_KEY
    else:
        # Direct Google AI Studio – LiteLlm accepts api_key as the Google key
        if GOOGLE_API_KEY:
            kwargs["api_key"] = GOOGLE_API_KEY

    return LiteLlm(**kwargs)


# ── Token-reduction helpers ────────────────────────────────────────────────────

def trim_code_for_context(code: str, max_lines: int = 150) -> str:
    """
    Trim a source file to the most relevant lines to reduce token usage.
    Keeps the first `max_lines` lines which typically contain signatures +
    class definitions – enough for test generation without full implementations.
    """
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code
    trimmed = lines[:max_lines]
    trimmed.append(f"# ... ({len(lines) - max_lines} more lines truncated for brevity)")
    return "\n".join(trimmed)


def build_test_prompt(
    language: str,
    framework: str,
    source_file: str,
    code_snippet: str,
    existing_tests: str | None = None,
    doc_context: str | None = None,
) -> str:
    """
    Build a compact LLM prompt for test generation.
    Short, directive prompts cost fewer tokens than verbose ones.
    """
    parts = [
        f"Generate {framework} unit tests for the following {language} code.",
        f"File: {source_file}",
        "Requirements:",
        "- Cover all public functions/methods",
        "- Include edge cases and error paths",
        "- Keep tests independent and deterministic",
        "- Return ONLY the test file content, no explanation",
        "",
        f"```{language}",
        code_snippet,
        "```",
    ]

    if existing_tests:
        parts += [
            "",
            "Existing tests (do NOT duplicate, only ADD missing coverage):",
            f"```{language}",
            existing_tests[:500],  # cap context
            "```",
        ]

    if doc_context:
        parts += [
            "",
            "Business requirements context (use for logical assertions):",
            doc_context[:800],  # cap context
        ]

    return "\n".join(parts)


    
    
