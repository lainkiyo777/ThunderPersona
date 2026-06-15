from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


@dataclass
class VectorHit:
    score: float
    doc: dict


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype("float32", copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


class FaissVectorIndex:
    def __init__(self, index_dir: Path, device: str | None = None):
        self.index_dir = index_dir
        self.manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        self.docs = read_jsonl(index_dir / "docs.jsonl")
        cwd = os.getcwd()
        try:
            os.chdir(index_dir)
            self.index = faiss.read_index("index.faiss")
        finally:
            os.chdir(cwd)
        self.model = SentenceTransformer(self.manifest["embedding_model"], device=device)

    @classmethod
    def build(
        cls,
        rag_file: Path,
        index_dir: Path,
        embedding_model: str = DEFAULT_MODEL,
        batch_size: int = 32,
    ) -> dict:
        index_dir.mkdir(parents=True, exist_ok=True)
        docs = read_jsonl(rag_file)
        texts = [row["text"] for row in docs]
        model = SentenceTransformer(embedding_model)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings = normalize(np.asarray(embeddings, dtype="float32"))
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        cwd = os.getcwd()
        try:
            os.chdir(index_dir)
            faiss.write_index(index, "index.faiss")
        finally:
            os.chdir(cwd)
        write_jsonl(index_dir / "docs.jsonl", docs)
        manifest = {
            "backend": "faiss",
            "embedding_model": embedding_model,
            "source_file": str(rag_file),
            "doc_count": len(docs),
            "dimensions": int(embeddings.shape[1]),
        }
        (index_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        query_embedding = normalize(np.asarray(query_embedding, dtype="float32"))
        scores, indices = self.index.search(query_embedding, top_k)
        hits: list[VectorHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            hits.append(VectorHit(score=float(score), doc=self.docs[int(idx)]))
        return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/search a FAISS embedding index for WeChat RAG docs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--rag-file", required=True, type=Path)
    build.add_argument("--index-dir", required=True, type=Path)
    build.add_argument("--embedding-model", default=DEFAULT_MODEL)
    build.add_argument("--batch-size", default=32, type=int)

    search = subparsers.add_parser("search")
    search.add_argument("--index-dir", required=True, type=Path)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", default=5, type=int)

    args = parser.parse_args()
    if args.command == "build":
        print(
            json.dumps(
                FaissVectorIndex.build(
                    args.rag_file,
                    args.index_dir,
                    args.embedding_model,
                    args.batch_size,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    index = FaissVectorIndex(args.index_dir)
    rows = []
    for hit in index.search(args.query, args.top_k):
        row = dict(hit.doc)
        row["score"] = round(hit.score, 4)
        rows.append(row)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
