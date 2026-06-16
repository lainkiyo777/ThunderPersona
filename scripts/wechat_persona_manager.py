from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from prepare_wechat_persona_dataset import (
    build_rag_docs,
    build_sft_samples,
    clean_messages,
    parse_markdown,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / ".wechat-exports"
PERSONAS_DIR = EXPORTS / "personas"
CONTACT_INDEX = EXPORTS / "contact_index.json"
WECHAT_CLI = ROOT / ".venv-wechat-cli" / "Scripts" / "wechat-cli.exe"
ProgressCallback = Callable[[str, int, str], None]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def contact_display_name(contact: dict[str, Any]) -> str:
    return str(
        contact.get("display_name")
        or contact.get("remark")
        or contact.get("nick_name")
        or contact.get("nickname")
        or contact.get("username")
        or ""
    ).strip()


def safe_slug(label: str, unique: str) -> str:
    ascii_part = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    digest = hashlib.sha1(f"{label}|{unique}".encode("utf-8")).hexdigest()[:8]
    if ascii_part:
        return f"{ascii_part[:42]}-{digest}"
    return f"persona-{digest}"


def normalize_contact(contact: dict[str, Any]) -> dict[str, Any]:
    username = str(contact.get("username") or "").strip()
    nick_name = str(contact.get("nick_name") or contact.get("nickname") or "").strip()
    remark = str(contact.get("remark") or "").strip()
    alias = str(contact.get("alias") or "").strip()
    display_name = remark or nick_name or username
    search_text = " ".join(
        item.lower()
        for item in [display_name, remark, nick_name, alias, username]
        if item
    )
    return {
        "username": username,
        "nick_name": nick_name,
        "remark": remark,
        "alias": alias,
        "display_name": display_name,
        "search_text": search_text,
        "slug": safe_slug(display_name, username or display_name),
    }


def _wechat_env() -> dict[str, str]:
    env = os.environ.copy()
    env["USERPROFILE"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    if os.name == "nt" and "Path" in env and "PATH" in env:
        if len(env.get("Path", "")) >= len(env.get("PATH", "")):
            env.pop("PATH", None)
        else:
            env["Path"] = env.pop("PATH")
    return env


def run_wechat_cli(args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    if not WECHAT_CLI.is_file():
        raise RuntimeError(f"wechat-cli executable not found: {WECHAT_CLI}")
    return subprocess.run(
        [str(WECHAT_CLI), *args],
        cwd=str(ROOT),
        env=_wechat_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout or int(os.environ.get("PERSONA_WECHAT_TIMEOUT", "300")),
        check=False,
    )


def refresh_contact_index(limit: int = 100000) -> dict[str, Any]:
    proc = run_wechat_cli(["contacts", "--format", "json", "--limit", str(limit)])
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "wechat-cli contacts failed").strip())

    raw_contacts = json.loads(proc.stdout or "[]")
    contacts = [normalize_contact(item) for item in raw_contacts if isinstance(item, dict)]
    seen: set[str] = set()
    unique_contacts: list[dict[str, Any]] = []
    for contact in contacts:
        key = contact.get("username") or contact.get("display_name")
        if not key or key in seen:
            continue
        seen.add(key)
        unique_contacts.append(contact)

    payload = {
        "version": 1,
        "updated_at": _now_iso(),
        "source": "wechat-cli contacts",
        "contacts": unique_contacts,
    }
    CONTACT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    CONTACT_INDEX.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_contact_index() -> dict[str, Any]:
    if not CONTACT_INDEX.is_file():
        return {"version": 1, "updated_at": None, "source": None, "contacts": []}
    return json.loads(CONTACT_INDEX.read_text(encoding="utf-8"))


def fuzzy_score(contact: dict[str, Any], query: str) -> int:
    query = query.strip().lower()
    if not query:
        return 1
    text = str(contact.get("search_text") or "").lower()
    display = str(contact.get("display_name") or "").lower()
    if query == display:
        return 1000
    if display.startswith(query):
        return 900
    if query in display:
        return 800
    if query in text:
        return 700

    pos = -1
    gaps = 0
    for char in query:
        next_pos = text.find(char, pos + 1)
        if next_pos < 0:
            return 0
        if pos >= 0:
            gaps += next_pos - pos - 1
        pos = next_pos
    return max(100, 500 - gaps)


def search_contacts(query: str, limit: int = 30) -> dict[str, Any]:
    index = load_contact_index()
    contacts = index.get("contacts") or []
    scored = [
        (fuzzy_score(contact, query), contact)
        for contact in contacts
        if fuzzy_score(contact, query) > 0
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return {
        "updated_at": index.get("updated_at"),
        "total": len(contacts),
        "contacts": [contact for _, contact in scored[:limit]],
    }


def persona_prefix(slug: str) -> Path:
    if slug == "zhang-sir":
        return EXPORTS / slug
    return PERSONAS_DIR / slug


def persona_paths(slug: str) -> dict[str, Path]:
    prefix = persona_prefix(slug)
    return {
        "prefix": prefix,
        "markdown": prefix.with_suffix(".md"),
        "messages": prefix.with_suffix(".messages.jsonl"),
        "rag": prefix.with_suffix(".rag_docs.jsonl"),
        "sft": prefix.with_suffix(".sft.jsonl"),
        "report": prefix.with_suffix(".report.json"),
        "vector": EXPORTS / "vector_index" / slug,
    }


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def list_distilled_personas() -> list[dict[str, Any]]:
    candidates: dict[str, Path] = {}
    for path in EXPORTS.glob("*.messages.jsonl"):
        candidates[path.name.removesuffix(".messages.jsonl")] = path
    if PERSONAS_DIR.is_dir():
        for path in PERSONAS_DIR.glob("*.messages.jsonl"):
            candidates[path.name.removesuffix(".messages.jsonl")] = path

    personas: list[dict[str, Any]] = []
    for slug, messages_path in sorted(candidates.items()):
        paths = persona_paths(slug)
        rag_path = paths["rag"] if paths["rag"].is_file() else messages_path.with_name(f"{slug}.rag_docs.jsonl")
        report_path = paths["report"] if paths["report"].is_file() else messages_path.with_name(f"{slug}.report.json")
        if not rag_path.is_file():
            continue
        target_name = slug
        updated_at = None
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                target_name = str(report.get("target_name") or target_name)
                updated_at = report.get("generated_at")
            except json.JSONDecodeError:
                pass
        personas.append(
            {
                "slug": slug,
                "target_name": target_name,
                "messages_file": str(messages_path),
                "rag_file": str(rag_path),
                "vector_index_dir": str(paths["vector"]),
                "message_count": _count_jsonl(messages_path),
                "doc_count": _count_jsonl(rag_path),
                "updated_at": updated_at or datetime.fromtimestamp(messages_path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return personas


def get_distilled_persona(slug: str) -> dict[str, Any] | None:
    for persona in list_distilled_personas():
        if persona["slug"] == slug:
            return persona
    return None


def distill_contact(
    contact: dict[str, Any],
    export_limit: int = 20000,
    build_vector: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    normalized = normalize_contact(contact)
    display_name = contact_display_name(normalized)
    chat_key = normalized.get("username") or display_name
    if not chat_key:
        raise RuntimeError("Selected contact has no usable name or username.")

    slug = normalized["slug"]
    paths = persona_paths(slug)
    paths["prefix"].parent.mkdir(parents=True, exist_ok=True)

    if progress:
        progress("exporting", 10, f"Exporting chat history for {display_name}")
    export_proc = run_wechat_cli(
        [
            "export",
            chat_key,
            "--format",
            "markdown",
            "--output",
            str(paths["markdown"]),
            "--limit",
            str(max(1, export_limit)),
        ],
        timeout=int(os.environ.get("PERSONA_EXPORT_TIMEOUT", "600")),
    )
    if export_proc.returncode != 0:
        raise RuntimeError((export_proc.stderr or export_proc.stdout or "wechat-cli export failed").strip())

    if progress:
        progress("parsing", 35, "Parsing markdown export")
    raw = parse_markdown(paths["markdown"])
    if progress:
        progress("cleaning", 45, f"Cleaning {len(raw)} raw messages")
    cleaned, skipped = clean_messages(raw)
    if not cleaned:
        raise RuntimeError(f"No usable text messages found for {display_name}.")

    if progress:
        progress("building", 58, f"Building RAG chunks from {len(cleaned)} clean messages")
    rag_docs = build_rag_docs(cleaned, display_name, chunk_size=24)
    sft_samples = build_sft_samples(
        cleaned,
        display_name,
        context_turns=12,
        max_reply_chars=600,
    )

    if progress:
        progress("writing", 72, "Writing persona artifacts")
    write_jsonl(paths["messages"], [message.__dict__ for message in cleaned])
    write_jsonl(paths["rag"], rag_docs)
    write_jsonl(paths["sft"], sft_samples)
    paths["report"].write_text(
        json.dumps(
            {
                "target_name": display_name,
                "slug": slug,
                "source_contact": normalized,
                "raw_messages": len(raw),
                "cleaned_messages": len(cleaned),
                "rag_docs": len(rag_docs),
                "sft_samples": len(sft_samples),
                "skipped": dict(skipped),
                "generated_at": _now_iso(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    vector_ready = False
    if build_vector:
        if progress:
            progress("vector", 82, "Building FAISS vector index")
        vector_proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "wechat_vector_index.py"),
                "build",
                "--rag-file",
                str(paths["rag"]),
                "--index-dir",
                str(paths["vector"]),
            ],
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=int(os.environ.get("PERSONA_VECTOR_TIMEOUT", "900")),
            check=False,
        )
        if vector_proc.returncode != 0:
            raise RuntimeError((vector_proc.stderr or vector_proc.stdout or "Vector index build failed").strip())
        vector_ready = True

    if progress:
        progress("finalizing", 92, "Loading distilled persona")
    persona = get_distilled_persona(slug)
    if not persona:
        raise RuntimeError(f"Distillation finished but persona files were not found for {display_name}.")
    persona["vector_ready"] = vector_ready
    return persona
