"""Local RAG + persona prompt workbench for a WeChat contact.

This module loads cleaned JSONL files produced by
prepare_wechat_persona_dataset.py, retrieves relevant chat-history chunks, and
builds a persona prompt for private writing-style simulation.
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
LAUGH_MARKERS = ("哈哈", "hhh", "笑死", "笑", "草")
QUESTION_MARKERS = ("?", "？", "吗", "嘛", "呢", "怎么", "为什么", "为啥", "啥")
EXCLAIM_MARKERS = ("!", "！")


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


def ratio_label(value: float, low: float, high: float, low_label: str, mid_label: str, high_label: str) -> str:
    if value < low:
        return low_label
    if value >= high:
        return high_label
    return mid_label


def confidence_summary(message_count: int) -> dict:
    if message_count >= 3000:
        return {
            "level": "high",
            "label": "High confidence",
            "summary": "样本量充足，可以稳定刻画短句、节奏、口癖和常见互动方式。",
        }
    if message_count >= 600:
        return {
            "level": "medium",
            "label": "Medium confidence",
            "summary": "样本量可用，适合做聊天风格模拟；少见场景仍需要依赖检索片段。",
        }
    return {
        "level": "low",
        "label": "Low confidence",
        "summary": "样本量偏少，只能做轻量风格参考，不应该强行补全没有证据的关系细节。",
    }


def build_persona_dna(
    target_name: str,
    message_count: int,
    avg_chars: float,
    median_chars: float,
    short_rate: float,
    laugh_rate: float,
    question_rate: float,
    emoji_rate: float,
    exclaim_rate: float,
    avg_burst: float,
    phrases: list[str],
) -> dict:
    rhythm_label = ratio_label(short_rate, 0.35, 0.70, "完整句偏多", "短句和完整句混合", "高频短句")
    affect_label = ratio_label(laugh_rate, 0.04, 0.14, "情绪标记少", "偶尔用笑声缓冲", "笑声存在感强")
    question_label = ratio_label(question_rate, 0.05, 0.14, "少追问", "会自然追问", "追问和反问明显")
    emoji_label = ratio_label(emoji_rate, 0.02, 0.08, "很少用 emoji", "偶尔用 emoji", "emoji 较明显")

    rhythm_summary = (
        f"平均 {avg_chars:.2f} 字，中位数 {median_chars:g} 字；"
        f"短回复占 {pct(short_rate)}，平均连续回复 {avg_burst:.2f} 条。"
    )
    voice_summary = (
        f"{affect_label}，{question_label}，{emoji_label}；"
        f"感叹号占比 {pct(exclaim_rate)}。"
    )

    return {
        "target_name": target_name,
        "confidence": confidence_summary(message_count),
        "rhythm": {
            "label": rhythm_label,
            "summary": rhythm_summary,
            "signals": [
                f"短回复率 {pct(short_rate)}",
                f"平均字数 {avg_chars:.2f}",
                f"连续回复 {avg_burst:.2f} 条",
            ],
        },
        "voice": {
            "label": affect_label,
            "summary": voice_summary,
            "signals": [
                f"笑声率 {pct(laugh_rate)}",
                f"追问率 {pct(question_rate)}",
                f"emoji 率 {pct(emoji_rate)}",
            ],
        },
        "interaction": {
            "label": question_label,
            "summary": "互动方式由追问率、短句率和检索片段共同决定；更像聊天反应，而不是观点输出。",
            "signals": [
                "情绪消息优先短安抚或轻打趣",
                "具体问题优先检索相似片段",
                "缺少证据时少发挥",
            ],
        },
        "phrase_bank": phrases[:10],
        "response_protocol": [
            "先判断当前用户消息的情绪和关系语境，再决定是安抚、打趣、追问还是直接回答。",
            "优先输出短微信回复；除非历史片段支持长回复，否则不要写成解释文或分析文。",
            "遇到历史里没有证据的事实或关系细节，用含糊、自然的说法带过，不要编故事。",
            "保留口语节奏，但避免把高频口癖堆满每一句，防止变成夸张模仿。",
        ],
        "anti_patterns": [
            "不要写成客服式完整句、公众号式鸡汤或 AI 助手式分析。",
            "不要连续列出多条候选回复；除非用户明确要求，只给最终可发送文本。",
            "不要为了像而强行塞满哈哈、语气词或口头禅。",
            "不要补全聊天记录里没有的身份、地点、动机、关系细节。",
        ],
        "honest_boundaries": [
            "这是基于聊天记录的风格模拟，不是本人，也不代表本人真实想法。",
            "只模拟聊天表达，不推断隐私、身份、位置、财务、账号等敏感信息。",
            "检索不到相近场景时，回答只能参考整体风格，准确度会下降。",
            "不用于冒充本人联系第三方或制造误导性对话截图。",
        ],
    }


def build_persona(messages: list[dict], target_name: str) -> dict:
    target = [m for m in messages if m.get("role") == "target"]
    contents = [m.get("content", "") for m in target if m.get("content")]
    lengths = [len(text) for text in contents] or [0]
    burst_count, avg_burst = count_bursts(messages, target_name)

    laugh_count = sum(1 for text in contents if any(mark in text.lower() for mark in LAUGH_MARKERS))
    emoji_count = sum(1 for text in contents if EMOJI_RE.search(text))
    question_count = sum(1 for text in contents if any(mark in text for mark in QUESTION_MARKERS))
    exclaim_count = sum(1 for text in contents if any(mark in text for mark in EXCLAIM_MARKERS))
    short_count = sum(1 for length in lengths if length <= 8)

    short_rate = short_count / max(len(lengths), 1)
    laugh_rate = laugh_count / max(len(contents), 1)
    question_rate = question_count / max(len(contents), 1)
    emoji_rate = emoji_count / max(len(contents), 1)
    exclaim_rate = exclaim_count / max(len(contents), 1)
    phrases = top_short_phrases(contents)

    style_rules = []
    if statistics.mean(lengths) <= 12:
        style_rules.append("回复偏短，优先用一两句微信口语，不写长段解释。")
    else:
        style_rules.append("可以写完整句，但仍保持微信聊天的自然节奏。")
    if short_rate >= 0.45:
        style_rules.append("大量使用短反应句，必要时连续发两三条短句。")
    if laugh_rate >= 0.12:
        style_rules.append("常用哈哈类笑声缓冲语气，但不要每句都加。")
    if question_rate >= 0.12:
        style_rules.append("会用反问、追问和口语化疑问推动对话。")
    if emoji_rate < 0.05:
        style_rules.append("emoji 使用较少，除非上下文强烈需要。")
    if exclaim_rate < 0.08:
        style_rules.append("语气通常不靠大量感叹号堆叠。")

    avg_chars = round(statistics.mean(lengths), 2)
    median_chars = statistics.median(lengths)
    persona_dna = build_persona_dna(
        target_name=target_name,
        message_count=len(contents),
        avg_chars=avg_chars,
        median_chars=median_chars,
        short_rate=short_rate,
        laugh_rate=laugh_rate,
        question_rate=question_rate,
        emoji_rate=emoji_rate,
        exclaim_rate=exclaim_rate,
        avg_burst=avg_burst,
        phrases=phrases,
    )

    return {
        "target_name": target_name,
        "target_text_messages": len(contents),
        "avg_chars": avg_chars,
        "median_chars": median_chars,
        "short_reply_rate": pct(short_rate),
        "laugh_rate": pct(laugh_rate),
        "question_rate": pct(question_rate),
        "emoji_rate": pct(emoji_rate),
        "exclaim_rate": pct(exclaim_rate),
        "rates": {
            "short": round(short_rate, 4),
            "laugh": round(laugh_rate, 4),
            "question": round(question_rate, 4),
            "emoji": round(emoji_rate, 4),
            "exclaim": round(exclaim_rate, 4),
        },
        "burst_count": burst_count,
        "avg_burst_messages": round(avg_burst, 2),
        "common_short_phrases": phrases,
        "style_rules": style_rules,
        "persona_dna": persona_dna,
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


def format_bullets(items: list[str]) -> str:
    if not items:
        return "- 无稳定信号"
    return "\n".join(f"- {item}" for item in items)


def format_persona_dna(persona: dict) -> str:
    dna = persona.get("persona_dna") or {}
    rhythm = dna.get("rhythm", {})
    voice = dna.get("voice", {})
    interaction = dna.get("interaction", {})
    confidence = dna.get("confidence", {})
    phrases = dna.get("phrase_bank") or persona.get("common_short_phrases") or []
    anti_patterns = dna.get("anti_patterns") or []
    phrase_line = "、".join(phrases[:10]) if phrases else "无稳定高频短句"
    anti_pattern_line = "；".join(anti_patterns[:4]) if anti_patterns else "无明确反模式"

    return f"""置信度：{confidence.get('label', 'Unknown')} - {confidence.get('summary', '')}
节奏：{rhythm.get('label', '')}。{rhythm.get('summary', '')}
语气：{voice.get('label', '')}。{voice.get('summary', '')}
互动：{interaction.get('label', '')}。{interaction.get('summary', '')}
高频短句/口头禅候选：{phrase_line}
反模式：{anti_pattern_line}"""


def build_prompt(query: str, persona: dict, scored_docs: list[tuple[float, RagDoc]]) -> str:
    dna = persona.get("persona_dna") or {}
    rules = format_bullets(persona.get("style_rules", []))
    protocol = format_bullets(dna.get("response_protocol", []))
    anti_patterns = format_bullets(dna.get("anti_patterns", []))
    boundaries = format_bullets(dna.get("honest_boundaries", []))
    sources = format_sources(scored_docs)

    return f"""你是一个私人写作辅助模型，用来做微信聊天风格模拟练习。

任务：
根据“当前上下文/用户消息”，写出一条像「{persona['target_name']}」会发的微信回复。
只输出回复正文，不要解释，不要加标签，不要写分析过程。

重要边界：
{boundaries}

Persona DNA：
{format_persona_dna(persona)}

风格画像：
- 文本消息数：{persona['target_text_messages']}
- 平均字数：{persona['avg_chars']}；中位数字数：{persona['median_chars']}
- 短回复比例：{persona['short_reply_rate']}
- 哈哈/笑声比例：{persona['laugh_rate']}
- 疑问/追问比例：{persona['question_rate']}
- emoji 比例：{persona['emoji_rate']}
- 平均连续回复条数：{persona['avg_burst_messages']}

回复协议：
{protocol}

反模式（不要这样回）：
{anti_patterns}

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
        "max_tokens": int(os.environ.get("PERSONA_MAX_TOKENS", "160")),
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
        "max_output_tokens": int(os.environ.get("PERSONA_MAX_TOKENS", "160")),
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
