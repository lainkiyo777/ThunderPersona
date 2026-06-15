from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
STATIC = Path(__file__).resolve().parent / "static"
EXPORTS = ROOT / ".wechat-exports"
MAX_HISTORY_ITEMS = 12
MAX_HISTORY_CHARS = 800
sys.path.insert(0, str(SCRIPTS))

from wechat_persona_rag_mvp import (  # noqa: E402
    RagDoc,
    bm25_search,
    build_docs,
    build_persona,
    build_prompt,
    call_model,
    provider_info,
    read_jsonl,
)
from wechat_vector_index import FaissVectorIndex  # noqa: E402


class PersonaRuntime:
    def __init__(
        self,
        messages_file: Path,
        rag_file: Path,
        target_name: str | None,
        provider: str,
        model: str | None,
        vector_index_dir: Path | None,
    ):
        self.messages_file = messages_file
        self.rag_file = rag_file
        self.target_name = target_name
        self.provider = provider
        self.model = model
        self.vector_index_dir = vector_index_dir
        self.vector_index: FaissVectorIndex | None = None
        self.local_lora: LocalLoraRuntime | None = None
        self.reload()

    def reload(self) -> None:
        self.messages = read_jsonl(self.messages_file)
        if not self.target_name:
            self.target_name = next(
                (
                    m.get("sender")
                    for m in self.messages
                    if m.get("role") == "target" and m.get("sender")
                ),
                "target",
            )
        self.docs = build_docs(read_jsonl(self.rag_file))
        self.persona = build_persona(self.messages, self.target_name)
        self.vector_index = None
        if self.vector_index_dir and (self.vector_index_dir / "manifest.json").is_file():
            self.vector_index = FaissVectorIndex(self.vector_index_dir, device="cpu")

    def vector_search(self, query: str, top_k: int) -> list[tuple[float, RagDoc]]:
        if self.vector_index is None:
            return []
        hits = self.vector_index.search(query, top_k)
        scored: list[tuple[float, RagDoc]] = []
        for hit in hits:
            doc = RagDoc(
                doc_id=hit.doc.get("id", ""),
                text=hit.doc.get("text", ""),
                metadata=hit.doc.get("metadata", {}),
                tokens={},
                length=0,
            )
            scored.append((hit.score, doc))
        return scored

    def hybrid_search(self, query: str, top_k: int) -> list[tuple[float, RagDoc]]:
        bm25 = bm25_search(self.docs, query, top_k * 2)
        vector = self.vector_search(query, top_k * 2)
        combined: dict[str, dict] = {}

        max_bm25 = max((score for score, _ in bm25), default=1.0) or 1.0
        max_vector = max((score for score, _ in vector), default=1.0) or 1.0

        for score, doc in bm25:
            item = combined.setdefault(doc.doc_id, {"doc": doc, "score": 0.0})
            item["score"] += 0.45 * (score / max_bm25)
        for score, doc in vector:
            item = combined.setdefault(doc.doc_id, {"doc": doc, "score": 0.0})
            item["score"] += 0.55 * (score / max_vector)

        ranked = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
        return [(float(item["score"]), item["doc"]) for item in ranked[:top_k]]

    def search(self, query: str, top_k: int, retriever: str) -> list[tuple[float, RagDoc]]:
        if retriever == "vector":
            scored = self.vector_search(query, top_k)
            return scored or bm25_search(self.docs, query, top_k)
        if retriever == "hybrid":
            scored = self.hybrid_search(query, top_k)
            return scored or bm25_search(self.docs, query, top_k)
        return bm25_search(self.docs, query, top_k)

    def build(self, query: str, top_k: int, retriever: str) -> tuple[str, list[tuple[float, object]]]:
        scored_docs = self.search(query, top_k, retriever)
        prompt = build_prompt(query, self.persona, scored_docs)
        return prompt, scored_docs

    def generate_local_lora(self, prompt: str) -> str:
        if self.local_lora is None:
            self.local_lora = LocalLoraRuntime(
                base_model=ROOT / ".hf-models" / "Qwen2.5-1.5B-Instruct",
                adapter=EXPORTS / "lora" / "output" / "qwen2.5-1.5b-persona-lora",
            )
        return self.local_lora.generate(prompt)


@dataclass
class LocalLoraRuntime:
    base_model: Path
    adapter: Path

    def __post_init__(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Local LoRA generation.")
        if not self.adapter.is_dir():
            raise RuntimeError(f"LoRA adapter not found: {self.adapter}")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.adapter), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            str(self.base_model),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, str(self.adapter))
        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "只输出目标联系人风格的微信回复正文。最多两行，每行尽量短。"
                    "不要解释，不要复述用户消息，不要连续列出候选回复。"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=True,
                temperature=0.55,
                top_p=0.85,
                repetition_penalty=1.18,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return clean_local_lora_answer(raw)


def clean_local_lora_answer(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return ""

    stop_markers = ["用户：", "用户:", "assistant", "Assistant", "回复：", "回复:"]
    for marker in stop_markers:
        marker_index = cleaned.find(marker)
        if marker_index > 0:
            cleaned = cleaned[:marker_index].strip()

    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip(" \t-—*·0123456789.、：:")
        if not line:
            continue
        if line in {"分析", "解释", "候选回复"}:
            continue
        lines.append(line)
        if len(lines) >= 2:
            break

    if not lines:
        first_line = cleaned.splitlines()[0].strip()
        lines = [first_line] if first_line else []

    answer = "\n".join(lines).strip()
    if len(answer) > 48:
        answer = answer[:48].rstrip("，。！？!?、；;,. ")
    return answer


RUNTIME: PersonaRuntime


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, body: bytes, content_type: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def clean_history(raw_history: object) -> list[dict[str, str]]:
    if not isinstance(raw_history, list):
        return []

    cleaned: list[dict[str, str]] = []
    for item in raw_history[-MAX_HISTORY_ITEMS:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if len(content) > MAX_HISTORY_CHARS:
            content = content[-MAX_HISTORY_CHARS:]
        cleaned.append({"role": role, "content": content})
    return cleaned


def build_session_query(query: str, history: list[dict[str, str]]) -> str:
    if not history:
        return query

    lines = ["Recent browser-session context, oldest to newest:"]
    for item in history:
        lines.append(f"{item['role']}: {item['content']}")
    lines.extend(["", "Current user message:", query])
    return "\n".join(lines)


def source_payload(scored_docs: list[tuple[float, object]]) -> list[dict]:
    sources = []
    for score, doc in scored_docs:
        metadata = getattr(doc, "metadata", {})
        sources.append(
            {
                "score": round(score, 4),
                "id": getattr(doc, "doc_id", ""),
                "text": getattr(doc, "text", ""),
                "metadata": metadata,
            }
        )
    return sources


def save_run(query: str, prompt: str, answer: str | None, sources: list[dict]) -> dict:
    run_dir = EXPORTS / "chat_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prompt_path = run_dir / f"{stamp}.prompt.md"
    meta_path = run_dir / f"{stamp}.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "query": query,
                "answer": answer,
                "sources": [
                    {
                        "score": item["score"],
                        "id": item["id"],
                        "metadata": item["metadata"],
                    }
                    for item in sources
                ],
                "prompt_path": str(prompt_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"prompt_path": str(prompt_path), "meta_path": str(meta_path)}


class Handler(BaseHTTPRequestHandler):
    server_version = "PersonaRagMVP/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            path = "/index.html"
        if path == "/api/persona":
            json_response(
                self,
                {
                    "persona": RUNTIME.persona,
                    "messages_file": str(RUNTIME.messages_file),
                    "rag_file": str(RUNTIME.rag_file),
                    "doc_count": len(RUNTIME.docs),
                    "message_count": len(RUNTIME.messages),
                    "vector_index_available": RUNTIME.vector_index is not None,
                    "vector_index_dir": str(RUNTIME.vector_index_dir) if RUNTIME.vector_index_dir else None,
                    "local_lora_available": (
                        EXPORTS / "lora" / "output" / "qwen2.5-1.5b-persona-lora" / "adapter_model.safetensors"
                    ).is_file(),
                    **provider_info(RUNTIME.provider, RUNTIME.model),
                },
            )
            return

        file_path = (STATIC / path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(STATIC.resolve())) or not file_path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        text_response(self, file_path.read_bytes(), content_type)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/chat":
                payload = read_body(self)
                query = str(payload.get("query", "")).strip()
                if not query:
                    json_response(self, {"error": "Query is empty."}, status=400)
                    return

                top_k = max(1, min(int(payload.get("top_k", 5)), 12))
                retriever = str(payload.get("retriever", "hybrid")).lower()
                if retriever not in {"bm25", "vector", "hybrid"}:
                    retriever = "hybrid"
                want_answer = bool(payload.get("answer"))
                answer_backend = str(payload.get("answer_backend", "api")).lower()
                history = clean_history(payload.get("history", []))
                session_query = build_session_query(query, history)
                prompt, scored_docs = RUNTIME.build(session_query, top_k, retriever)
                sources = source_payload(scored_docs)
                answer = None
                if want_answer:
                    if answer_backend == "local_lora":
                        answer = RUNTIME.generate_local_lora(prompt)
                    else:
                        answer = call_model(prompt, RUNTIME.provider, RUNTIME.model)

                saved = None
                if bool(payload.get("save")):
                    saved = save_run(query, prompt, answer, sources)

                json_response(
                    self,
                    {
                        "query": query,
                        "prompt": prompt,
                        "answer": answer,
                        "answer_backend": answer_backend,
                        "history_used": len(history),
                        "sources": sources,
                        "retriever": retriever,
                        "saved": saved,
                        "persona": RUNTIME.persona,
                    },
                )
                return

            if parsed.path == "/api/reload":
                RUNTIME.reload()
                json_response(self, {"ok": True, "persona": RUNTIME.persona})
                return

            if parsed.path == "/api/shutdown":
                json_response(self, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            self.send_error(404)
        except Exception as exc:  # noqa: BLE001
            json_response(self, {"error": str(exc)}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--messages-file", default=str(EXPORTS / "persona.messages.jsonl"), type=Path)
    parser.add_argument("--rag-file", default=str(EXPORTS / "persona.rag_docs.jsonl"), type=Path)
    parser.add_argument("--vector-index-dir", default=str(EXPORTS / "vector_index" / "persona"), type=Path)
    parser.add_argument("--target-name", default=None)
    parser.add_argument("--provider", default="auto", choices=["auto", "deepseek", "openai"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    global RUNTIME
    RUNTIME = PersonaRuntime(
        args.messages_file,
        args.rag_file,
        args.target_name,
        args.provider,
        args.model,
        args.vector_index_dir,
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Persona RAG app running at http://{args.host}:{args.port}")
    info = provider_info(RUNTIME.provider, RUNTIME.model)
    print(
        f"Target: {RUNTIME.persona['target_name']} | docs={len(RUNTIME.docs)} "
        f"messages={len(RUNTIME.messages)} | vector={RUNTIME.vector_index is not None} | "
        f"provider={info['provider']} model={info['model']}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
