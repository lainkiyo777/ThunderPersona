"""Build a self-persona dataset from existing WeChat persona exports.

This script reverses the existing cleaned message role convention:
  - original role=me becomes target/self
  - original role=target becomes context/other

It does not print message contents. Outputs follow the same file contract as the
contact persona pipeline, so the web app can switch to the generated persona.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from prepare_wechat_persona_dataset import (
    Message,
    build_rag_docs,
    build_sft_samples,
    clean_messages,
    parse_markdown,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / ".wechat-exports"
WECHAT_CLI = ROOT / ".venv-wechat-cli" / "Scripts" / "wechat-cli.exe"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} line {line_no}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def discover_inputs(exports_dir: Path, output_prefix: Path) -> list[Path]:
    output_messages = output_prefix.with_suffix(".messages.jsonl").resolve()
    candidates = [
        *exports_dir.glob("*.messages.jsonl"),
        *(exports_dir / "personas").glob("*.messages.jsonl"),
    ]
    unique: dict[Path, Path] = {}
    for path in candidates:
        resolved = path.resolve()
        if resolved == output_messages:
            continue
        if path.name.startswith("self."):
            continue
        unique[resolved] = path
    return sorted(unique.values(), key=lambda item: str(item).lower())


def source_slug(path: Path) -> str:
    name = path.name
    if name.endswith(".messages.jsonl"):
        return name.removesuffix(".messages.jsonl")
    return path.stem


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


def safe_session_slug(chat: str, username: str) -> str:
    label = (chat or username or "session").strip()
    digest = hashlib.sha1(f"{chat}|{username}".encode("utf-8")).hexdigest()[:10]
    return f"session-{digest}"


def load_sessions(limit: int) -> list[dict]:
    proc = run_wechat_cli(["sessions", "--limit", str(limit), "--format", "json"])
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "wechat-cli sessions failed").strip())
    raw = json.loads(proc.stdout or "[]")
    return [item for item in raw if isinstance(item, dict)]


def export_sessions(
    limit: int,
    output_dir: Path,
    messages_per_session: int | None,
    resume: bool,
    timeout: int,
) -> dict:
    sessions = load_sessions(limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    manifest = {
        "version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "session_limit": limit,
        "messages_per_session": messages_per_session,
        "total_sessions": len(sessions),
        "exported": [],
        "skipped_existing": 0,
        "failed": [],
    }

    for index, session in enumerate(sessions, 1):
        chat = str(session.get("chat") or "").strip()
        username = str(session.get("username") or "").strip()
        chat_key = username or chat
        if not chat_key:
            manifest["failed"].append({"index": index, "reason": "missing_chat_key"})
            continue

        slug = safe_session_slug(chat, username)
        output_path = output_dir / f"{slug}.md"
        if resume and output_path.is_file() and output_path.stat().st_size > 0:
            manifest["skipped_existing"] += 1
            manifest["exported"].append({"index": index, "slug": slug, "path": str(output_path), "resumed": True})
            continue

        args = ["export", chat_key, "--format", "markdown", "--output", str(output_path)]
        if messages_per_session:
            args.extend(["--limit", str(messages_per_session)])
        proc = run_wechat_cli(args, timeout=timeout)
        if proc.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0:
            manifest["exported"].append({"index": index, "slug": slug, "path": str(output_path), "resumed": False})
        else:
            reason = (proc.stderr or proc.stdout or "wechat-cli export failed").strip()
            manifest["failed"].append({"index": index, "slug": slug, "reason": reason[:500]})

        if index % 25 == 0:
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "stage": "exporting",
                        "processed": index,
                        "total": len(sessions),
                        "exported": len(manifest["exported"]),
                        "failed": len(manifest["failed"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def invert_messages(messages_in: list[Message], self_name: str, fallback_other: str) -> tuple[list[Message], Counter]:
    messages: list[Message] = []
    skipped: Counter = Counter()

    for item in messages_in:
        role = item.role
        content = item.content.strip()
        timestamp = item.timestamp.strip()
        if not content or not timestamp:
            skipped["missing_required"] += 1
            continue
        if role == "me":
            messages.append(Message(timestamp=timestamp, sender=self_name, role="target", content=content))
        elif role == "target":
            sender = str(item.sender or fallback_other).strip() or fallback_other
            messages.append(Message(timestamp=timestamp, sender=sender, role="me", content=content))
        else:
            skipped["unknown_role"] += 1

    return messages, skipped


def invert_jsonl(path: Path, self_name: str) -> tuple[list[Message], Counter]:
    rows = read_jsonl(path)
    source = source_slug(path)
    messages = [
        Message(
            timestamp=str(row.get("timestamp") or ""),
            sender=str(row.get("sender") or source),
            role=str(row.get("role") or ""),
            content=str(row.get("content") or ""),
        )
        for row in rows
    ]
    return invert_messages(messages, self_name, source)


def invert_markdown(path: Path, self_name: str) -> tuple[list[Message], Counter]:
    return invert_messages(parse_markdown(path), self_name, source_slug(path))


def parse_time(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def sort_messages(messages: Iterable[Message]) -> list[Message]:
    return sorted(
        messages,
        key=lambda msg: (
            parse_time(msg.timestamp) or datetime.min,
            msg.sender,
            msg.role,
            msg.content,
        ),
    )


def build_self_dataset(
    input_paths: list[Path],
    output_prefix: Path,
    self_name: str,
    context_turns: int,
    rag_chunk_size: int,
    max_reply_chars: int,
) -> dict:
    grouped: dict[str, list[Message]] = {}
    skipped: Counter = Counter()
    raw_count = 0

    for path in input_paths:
        if path.suffix.lower() == ".md":
            inverted, source_skipped = invert_markdown(path, self_name)
        else:
            inverted, source_skipped = invert_jsonl(path, self_name)
        raw_count += len(inverted)
        skipped.update(source_skipped)
        cleaned, clean_skipped = clean_messages(inverted)
        skipped.update(clean_skipped)
        if cleaned:
            grouped[source_slug(path)] = sort_messages(cleaned)

    all_messages = sort_messages(message for group in grouped.values() for message in group)
    if not all_messages:
        raise RuntimeError("No usable self messages found in the selected exports.")

    rag_docs: list[dict] = []
    sft_samples: list[dict] = []
    per_source: dict[str, dict] = {}
    doc_offset = 0

    for source, messages in grouped.items():
        target_count = sum(1 for msg in messages if msg.role == "target")
        other_count = sum(1 for msg in messages if msg.role == "me")
        per_source[source] = {
            "messages": len(messages),
            "self_messages": target_count,
            "other_messages": other_count,
        }

        docs = build_rag_docs(messages, f"{self_name}-{source}", chunk_size=rag_chunk_size)
        for index, doc in enumerate(docs):
            doc["id"] = f"wechat-self-{doc_offset + index:06d}"
            doc.setdefault("metadata", {})["chat"] = source
            doc.setdefault("metadata", {})["persona"] = self_name
        doc_offset += len(docs)
        rag_docs.extend(docs)

        sft_samples.extend(
            build_sft_samples(
                messages,
                self_name,
                context_turns=context_turns,
                max_reply_chars=max_reply_chars,
            )
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    messages_path = output_prefix.with_suffix(".messages.jsonl")
    rag_path = output_prefix.with_suffix(".rag_docs.jsonl")
    sft_path = output_prefix.with_suffix(".sft.jsonl")
    report_path = output_prefix.with_suffix(".report.json")

    write_jsonl(messages_path, (asdict(message) for message in all_messages))
    write_jsonl(rag_path, rag_docs)
    write_jsonl(sft_path, sft_samples)

    timestamps = [parsed for msg in all_messages if (parsed := parse_time(msg.timestamp))]
    report = {
        "target_name": self_name,
        "slug": output_prefix.name,
        "source": "self aggregation from existing persona message exports",
        "source_files": [str(path) for path in input_paths],
        "source_count": len(input_paths),
        "raw_inverted_messages": raw_count,
        "cleaned_messages": len(all_messages),
        "target_text_messages": sum(1 for msg in all_messages if msg.role == "target"),
        "me_text_messages": sum(1 for msg in all_messages if msg.role == "me"),
        "rag_docs": len(rag_docs),
        "sft_samples": len(sft_samples),
        "skipped": dict(skipped),
        "per_source": per_source,
        "date_start": min(timestamps).isoformat(timespec="minutes") if timestamps else None,
        "date_end": max(timestamps).isoformat(timespec="minutes") if timestamps else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "outputs": {
            "messages": str(messages_path),
            "rag_docs": str(rag_path),
            "sft": str(sft_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self persona from existing cleaned WeChat exports.")
    parser.add_argument("--input", action="append", type=Path, help="Input *.messages.jsonl. Can be repeated.")
    parser.add_argument("--exports-dir", default=EXPORTS, type=Path)
    parser.add_argument("--output-prefix", default=EXPORTS / "self", type=Path)
    parser.add_argument("--self-name", default="我")
    parser.add_argument("--export-sessions", action="store_true", help="Export recent WeChat sessions before building.")
    parser.add_argument("--session-limit", default=100000, type=int)
    parser.add_argument("--source-dir", default=EXPORTS / "self_sources", type=Path)
    parser.add_argument("--messages-per-session", default=None, type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--export-timeout", default=600, type=int)
    parser.add_argument("--context-turns", default=12, type=int)
    parser.add_argument("--rag-chunk-size", default=24, type=int)
    parser.add_argument("--max-reply-chars", default=600, type=int)
    args = parser.parse_args()

    if args.export_sessions:
        manifest = export_sessions(
            limit=args.session_limit,
            output_dir=args.source_dir,
            messages_per_session=args.messages_per_session,
            resume=not args.no_resume,
            timeout=args.export_timeout,
        )
        input_paths = [Path(item["path"]) for item in manifest.get("exported", [])]
    else:
        input_paths = args.input or discover_inputs(args.exports_dir, args.output_prefix)
    if not input_paths:
        raise SystemExit("No input *.messages.jsonl files found.")

    report = build_self_dataset(
        input_paths=input_paths,
        output_prefix=args.output_prefix,
        self_name=args.self_name,
        context_turns=args.context_turns,
        rag_chunk_size=args.rag_chunk_size,
        max_reply_chars=args.max_reply_chars,
    )
    printable = {
        key: report[key]
        for key in [
            "target_name",
            "source_count",
            "cleaned_messages",
            "target_text_messages",
            "me_text_messages",
            "rag_docs",
            "sft_samples",
            "date_start",
            "date_end",
            "outputs",
        ]
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
