"""Prepare a WeChat export for persona-style RAG/SFT experiments.

Input: markdown produced by `wechat-cli export`.
Outputs:
  - *.messages.jsonl: cleaned text messages, one per line
  - *.rag_docs.jsonl: retrieval chunks for memory/RAG
  - *.sft.jsonl: context -> target reply samples in chat messages format
  - *.report.json: counts and date range
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


MESSAGE_RE = re.compile(
    r"^- \[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] "
    r"(?P<sender>.*?): (?P<content>.*)$"
)

NON_TEXT_PREFIXES = (
    "[voice]",
    "[call]",
    "[location]",
    "[image]",
    "[video]",
    "[file]",
    "[link/file]",
    "[emoji]",
    "[sticker]",
    "[red packet]",
    "[transfer]",
)

NON_TEXT_PREFIXES_ZH = (
    "[语音]",
    "[通话]",
    "[位置]",
    "[图片]",
    "[视频]",
    "[文件]",
    "[链接/文件]",
    "[表情]",
    "[红包]",
    "[转账]",
)

PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ID_RE = re.compile(r"(?<!\d)\d{6}(?:19|20)\d{2}\d{2}\d{2}\d{3}[\dXx](?!\d)")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
COORD_RE = re.compile(r"\b(?:x|y|lat|lng|longitude|latitude)=\"?-?\d+(?:\.\d+)?\"?")
XML_RE = re.compile(r"<[^>]+>")


@dataclass
class Message:
    timestamp: str
    sender: str
    role: str
    content: str


def parse_markdown(path: Path) -> list[Message]:
    messages: list[Message] = []
    current: Message | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = MESSAGE_RE.match(raw_line)
        if match:
            if current is not None:
                messages.append(current)
            sender = match.group("sender").strip()
            role = "me" if sender == "me" else "target"
            current = Message(
                timestamp=match.group("timestamp"),
                sender=sender,
                role=role,
                content=match.group("content").strip(),
            )
            continue

        if current is not None and raw_line.strip():
            current.content = f"{current.content}\n{raw_line.strip()}".strip()

    if current is not None:
        messages.append(current)
    return messages


def classify_skip_reason(text: str) -> str | None:
    stripped = text.strip()
    lower = stripped.lower()
    if not stripped:
        return "empty"
    if any(stripped.startswith(prefix) for prefix in NON_TEXT_PREFIXES_ZH):
        return "non_text"
    if any(lower.startswith(prefix) for prefix in NON_TEXT_PREFIXES):
        return "non_text"
    if stripped.startswith("<") or "<?xml" in stripped or "<msg" in stripped:
        return "xml"
    if "<location" in stripped or "<voicemsg" in stripped:
        return "xml"
    return None


def redact(text: str) -> str:
    text = PHONE_RE.sub("[PHONE]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = ID_RE.sub("[ID]", text)
    text = URL_RE.sub("[URL]", text)
    text = COORD_RE.sub("[COORD]", text)
    text = XML_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_messages(messages: Iterable[Message]) -> tuple[list[Message], Counter]:
    kept: list[Message] = []
    skipped: Counter = Counter()

    for msg in messages:
        reason = classify_skip_reason(msg.content)
        if reason:
            skipped[reason] += 1
            continue

        content = redact(msg.content)
        if not content:
            skipped["empty_after_redact"] += 1
            continue
        if len(content) > 500:
            skipped["too_long"] += 1
            continue

        kept.append(Message(msg.timestamp, msg.sender, msg.role, content))

    return kept, skipped


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_rag_docs(messages: list[Message], target_name: str, chunk_size: int) -> list[dict]:
    docs: list[dict] = []
    for start in range(0, len(messages), chunk_size):
        chunk = messages[start : start + chunk_size]
        if not chunk:
            continue
        text = "\n".join(f"{m.sender}: {m.content}" for m in chunk)
        docs.append(
            {
                "id": f"wechat-{target_name}-{start:06d}",
                "text": text,
                "metadata": {
                    "source": "wechat",
                    "chat": target_name,
                    "start_time": chunk[0].timestamp,
                    "end_time": chunk[-1].timestamp,
                    "message_count": len(chunk),
                },
            }
        )
    return docs


def format_context(messages: list[Message]) -> str:
    return "\n".join(f"{m.sender}: {m.content}" for m in messages)


def build_sft_samples(
    messages: list[Message],
    target_name: str,
    context_turns: int,
    max_reply_chars: int,
) -> list[dict]:
    samples: list[dict] = []
    i = 0
    system = (
        f"You are writing a private style simulation for the chat contact {target_name}. "
        "Reply only as this contact would in this conversation. "
        "Do not claim to be the real person outside this private experiment."
    )

    while i < len(messages):
        msg = messages[i]
        if msg.role != "target":
            i += 1
            continue

        j = i
        replies: list[str] = []
        while j < len(messages) and messages[j].role == "target":
            replies.append(messages[j].content)
            j += 1

        reply = "\n".join(replies).strip()
        context = messages[max(0, i - context_turns) : i]
        if context and reply and len(reply) <= max_reply_chars:
            samples.append(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": (
                                "Given the recent WeChat context, write the next reply "
                                f"from {target_name}.\n\nContext:\n{format_context(context)}"
                            ),
                        },
                        {"role": "assistant", "content": reply},
                    ],
                    "metadata": {
                        "chat": target_name,
                        "reply_start_time": msg.timestamp,
                        "reply_message_count": len(replies),
                    },
                }
            )
        i = j

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--context-turns", default=12, type=int)
    parser.add_argument("--rag-chunk-size", default=24, type=int)
    parser.add_argument("--max-reply-chars", default=600, type=int)
    args = parser.parse_args()

    raw = parse_markdown(args.input)
    cleaned, skipped = clean_messages(raw)
    rag_docs = build_rag_docs(cleaned, args.target_name, args.rag_chunk_size)
    sft_samples = build_sft_samples(
        cleaned,
        args.target_name,
        args.context_turns,
        args.max_reply_chars,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    messages_path = args.output_prefix.with_suffix(".messages.jsonl")
    rag_path = args.output_prefix.with_suffix(".rag_docs.jsonl")
    sft_path = args.output_prefix.with_suffix(".sft.jsonl")
    report_path = args.output_prefix.with_suffix(".report.json")

    write_jsonl(messages_path, (asdict(m) for m in cleaned))
    write_jsonl(rag_path, rag_docs)
    write_jsonl(sft_path, sft_samples)

    timestamps = []
    for msg in cleaned:
        try:
            timestamps.append(datetime.strptime(msg.timestamp, "%Y-%m-%d %H:%M"))
        except ValueError:
            pass

    report = {
        "input": str(args.input),
        "target_name": args.target_name,
        "raw_messages": len(raw),
        "cleaned_text_messages": len(cleaned),
        "target_text_messages": sum(1 for m in cleaned if m.role == "target"),
        "me_text_messages": sum(1 for m in cleaned if m.role == "me"),
        "rag_docs": len(rag_docs),
        "sft_samples": len(sft_samples),
        "skipped": dict(skipped),
        "date_start": min(timestamps).isoformat(timespec="minutes") if timestamps else None,
        "date_end": max(timestamps).isoformat(timespec="minutes") if timestamps else None,
        "outputs": {
            "messages": str(messages_path),
            "rag_docs": str(rag_path),
            "sft": str(sft_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
