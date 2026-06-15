"""Local RAG + persona prompt MVP for a WeChat contact.

This script intentionally has no third-party dependencies. It loads the cleaned
JSONL files produced by prepare_wechat_persona_dataset.py, retrieves relevant
chat-history chunks with BM25, and builds a persona prompt that can be pasted
into a chat model. If OPENAI_API_KEY is configured, --answer can call the
OpenAI Responses API directly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
WORD_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")
EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa70-\U0001faff"
    "]"
)


@dataclass
class RagDoc:
    doc_id: str
    text: str
    metadata: dict
    tokens: Counter
    length: int


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in WORD_RE.finditer(text.lower()):
        term = match.group(0)
        if CJK_RE.fullmatch(term):
            tokens.extend(term)
            if len(term) >= 2:
                tokens.extend(term[i : i + 2] for i in range(len(term) - 1))
            if len(term) >= 3:
                tokens.extend(term[i : i + 3] for i in range(len(term) - 2))
        else:
            tokens.append(term)
    return tokens


def build_docs(rows: Iterable[dict]) -> list[RagDoc]:
    docs: list[RagDoc] = []
    for row in rows:
        tokens = Counter(tokenize(row.get("text", "")))
        docs.append(
            RagDoc(
                doc_id=row.get("id", ""),
                text=row.get("text", ""),
                metadata=row.get("metadata", {}),
                tokens=tokens,
                length=sum(tokens.values()),
            )
        )
    return docs


def bm25_search(docs: list[RagDoc], query: str, top_k: int) -> list[tuple[float, RagDoc]]:
    query_terms = Counter(tokenize(query))
    if not query_terms:
        return [(0.0, doc) for doc in docs[-top_k:]][::-1]

    n_docs = len(docs)
    doc_freq: Counter = Counter()
    for doc in docs:
        for term in doc.tokens:
            doc_freq[term] += 1

    avg_len = sum(doc.length for doc in docs) / max(n_docs, 1)
    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, RagDoc]] = []

    for doc in docs:
        score = 0.0
        for term, qf in query_terms.items():
            tf = doc.tokens.get(term, 0)
            if tf == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * doc.length / max(avg_len, 1))
            score += idf * ((tf * (k1 + 1)) / denom) * min(qf, 3)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def count_bursts(messages: list[dict], target_name: str) -> tuple[int, float]:
    bursts: list[int] = []
    i = 0
    while i < len(messages):
        if messages[i].get("role") != "target":
            i += 1
            continue
        j = i
        while j < len(messages) and messages[j].get("role") == "target":
            j += 1
        bursts.append(j - i)
        i = j
    return len(bursts), statistics.mean(bursts) if bursts else 0.0


def top_short_phrases(contents: list[str], limit: int = 12) -> list[str]:
    counter = Counter()
    for text in contents:
        normalized = re.sub(r"\s+", " ", text).strip()
        if 1 <= len(normalized) <= 18:
            counter[normalized] += 1
    return [phrase for phrase, count in counter.most_common(limit) if count >= 2]


def build_persona(messages: list[dict], target_name: str) -> dict:
    target = [m for m in messages if m.get("role") == "target"]
    contents = [m.get("content", "") for m in target if m.get("content")]
    lengths = [len(text) for text in contents] or [0]
    burst_count, avg_burst = count_bursts(messages, target_name)

    laugh_count = sum(1 for text in contents if "哈" in text or "hhh" in text.lower())
    emoji_count = sum(1 for text in contents if EMOJI_RE.search(text))
    question_count = sum(1 for text in contents if any(mark in text for mark in ("?", "？", "吗", "咋", "怎么", "为啥", "啥")))
    exclaim_count = sum(1 for text in contents if "!" in text or "！" in text)
    short_count = sum(1 for length in lengths if length <= 8)

    style_rules = []
    if statistics.mean(lengths) <= 12:
        style_rules.append("回复偏短，优先用一两句微信口语，不写长段解释。")
    else:
        style_rules.append("可以写完整句，但仍保持微信聊天的自然节奏。")
    if short_count / max(len(lengths), 1) >= 0.45:
        style_rules.append("大量使用短反应句，必要时连续发两三条短句。")
    if laugh_count / max(len(contents), 1) >= 0.12:
        style_rules.append("常用哈哈类笑声缓冲语气，但不要每句都加。")
    if question_count / max(len(contents), 1) >= 0.12:
        style_rules.append("会用反问、追问和口语化疑问推动对话。")
    if emoji_count / max(len(contents), 1) < 0.05:
        style_rules.append("emoji 使用较少，除非上下文强烈需要。")
    if exclaim_count / max(len(contents), 1) < 0.08:
        style_rules.append("语气通常不靠大量感叹号堆叠。")

    return {
        "target_name": target_name,
        "target_text_messages": len(contents),
        "avg_chars": round(statistics.mean(lengths), 2),
        "median_chars": statistics.median(lengths),
        "short_reply_rate": pct(short_count / max(len(lengths), 1)),
        "laugh_rate": pct(laugh_count / max(len(contents), 1)),
        "question_rate": pct(question_count / max(len(contents), 1)),
        "emoji_rate": pct(emoji_count / max(len(contents), 1)),
        "burst_count": burst_count,
        "avg_burst_messages": round(avg_burst, 2),
        "common_short_phrases": top_short_phrases(contents),
        "style_rules": style_rules,
    }


def format_sources(scored_docs: list[tuple[float, RagDoc]]) -> str:
    if not scored_docs:
        return "未检索到相关历史片段。"
    blocks: list[str] = []
    for idx, (score, doc) in enumerate(scored_docs, 1):
        meta = doc.metadata
        header = (
            f"[片段 {idx}] score={score:.2f}; "
            f"time={meta.get('start_time', '?')} -> {meta.get('end_time', '?')}"
        )
        blocks.append(f"{header}\n{doc.text}")
    return "\n\n".join(blocks)


def build_prompt(query: str, persona: dict, scored_docs: list[tuple[float, RagDoc]]) -> str:
    phrases = persona.get("common_short_phrases") or []
    phrase_line = "、".join(phrases[:10]) if phrases else "无稳定高频短句。"
    rules = "\n".join(f"- {rule}" for rule in persona.get("style_rules", []))
    sources = format_sources(scored_docs)

    return f"""你是一个私人的写作辅助模型，用来做聊天风格模拟练习。

重要边界：
- 你不是本人，不要声称自己是真实的「{persona['target_name']}」。
- 不要输出身份证号、手机号、住址、账号、定位、银行卡、密码等隐私信息。
- 不要编造历史事实；历史片段里没有的事实，用不确定语气或自然带过。
- 不要帮助冒充本人去欺骗第三方。

任务：
根据“当前上下文/用户消息”，写出一个像「{persona['target_name']}」会发的微信回复。
只输出回复正文，不要解释，不要加标签。

风格画像：
- 文本消息数：{persona['target_text_messages']}
- 平均字数：{persona['avg_chars']}；中位数字数：{persona['median_chars']}
- 短回复比例：{persona['short_reply_rate']}
- 哈哈/笑声比例：{persona['laugh_rate']}
- 疑问/追问比例：{persona['question_rate']}
- emoji 比例：{persona['emoji_rate']}
- 平均连续回复条数：{persona['avg_burst_messages']}
- 高频短句/口头禅候选：{phrase_line}

风格规则：
{rules}

检索到的历史片段：
{sources}

当前上下文/用户消息：
{query}

请输出「{persona['target_name']}」风格的一条或多条微信回复。"""


def provider_info(provider: str = "auto", model: str | None = None) -> dict:
    selected = provider
    api_key_available = False
    resolved_model = model

    if provider == "auto":
        if os.environ.get("DEEPSEEK_API_KEY"):
            selected = "deepseek"
        elif os.environ.get("OPENAI_API_KEY"):
            selected = "openai"
        else:
            selected = "none"

    if selected == "deepseek":
        api_key_available = bool(os.environ.get("DEEPSEEK_API_KEY"))
        resolved_model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    elif selected == "openai":
        api_key_available = bool(os.environ.get("OPENAI_API_KEY"))
        resolved_model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    else:
        resolved_model = model

    return {
        "provider": selected,
        "api_key_available": api_key_available,
        "model": resolved_model,
    }


def call_deepseek(prompt: str, model: str | None = None) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    body = {
        "model": model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.7,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API error {exc.code}: {detail}") from exc

    choices = data.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "").strip()


def call_openai(prompt: str, model: str | None = None) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    body = {
        "model": model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        "input": prompt,
        "temperature": 0.7,
        "max_output_tokens": 400,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {detail}") from exc

    if data.get("output_text"):
        return data["output_text"].strip()

    parts: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def call_model(prompt: str, provider: str = "auto", model: str | None = None) -> str:
    info = provider_info(provider, model)
    selected = info["provider"]
    if selected == "deepseek":
        return call_deepseek(prompt, info["model"])
    if selected == "openai":
        return call_openai(prompt, info["model"])
    raise RuntimeError("No model API key is configured. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a RAG-backed persona prompt from a WeChat export."
    )
    parser.add_argument("--messages-file", required=True, type=Path)
    parser.add_argument("--rag-file", required=True, type=Path)
    parser.add_argument("--target-name", help="Display name for the simulated contact. Defaults to the first target sender.")
    parser.add_argument("--query", help="Current message/context. Omit for interactive mode.")
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--save-persona", type=Path)
    parser.add_argument("--save-prompt", type=Path)
    parser.add_argument("--quiet", action="store_true", help="Do not print the generated prompt/answer.")
    parser.add_argument("--answer", action="store_true", help="Call the configured model provider.")
    parser.add_argument("--provider", choices=["auto", "deepseek", "openai"], default="auto")
    parser.add_argument("--model", default=None)
    return parser


def run_once(args: argparse.Namespace, query: str, messages: list[dict], docs: list[RagDoc]) -> str:
    persona = build_persona(messages, args.target_name)
    scored_docs = bm25_search(docs, query, args.top_k)
    prompt = build_prompt(query, persona, scored_docs)

    if args.save_persona:
        args.save_persona.parent.mkdir(parents=True, exist_ok=True)
        args.save_persona.write_text(
            json.dumps(persona, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.save_prompt:
        args.save_prompt.parent.mkdir(parents=True, exist_ok=True)
        args.save_prompt.write_text(prompt, encoding="utf-8")

    if args.answer:
        return call_model(prompt, args.provider, args.model)
    return prompt


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    messages = read_jsonl(args.messages_file)
    if not args.target_name:
        args.target_name = next(
            (m.get("sender") for m in messages if m.get("role") == "target" and m.get("sender")),
            "target",
        )
    docs = build_docs(read_jsonl(args.rag_file))

    if args.query:
        result = run_once(args, args.query, messages, docs)
        if not args.quiet:
            print(result)
        return

    print("输入当前消息/上下文，回车生成 prompt。输入 /exit 退出。", file=sys.stderr)
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return
        if query in {"/exit", "/quit"}:
            return
        if not query:
            continue
        result = run_once(args, query, messages, docs)
        if not args.quiet:
            print(result)
            print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
