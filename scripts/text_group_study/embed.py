"""Embed every unique text (corpus + test sets) once with the project's ATTACK-BERT embedder.

Reads and writes in STUDY_DIR, or in the folder given as the first argument."""

import hashlib
import json
import sys
import time

import numpy as np
import torch
from harness import STUDY_DIR

from apt_classifier.config import Settings
from apt_classifier.embeddings import SentenceTransformerEmbedder

S = sys.argv[1] if len(sys.argv) > 1 else str(STUDY_DIR)
with open(f"{S}/corpus.json") as fh:
    corpus = json.load(fh)
with open(f"{S}/test_sets.json") as fh:
    tests = json.load(fh)
texts = sorted(
    {r["text"] for r in corpus} | {t for rows in tests.values() for r in rows for t in r["texts"]}
)
free = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0
device = "cpu"  # the shared GPU is near full (Ollama + API); CPU keeps other sessions safe
print(len(texts), "unique texts; device", device, f"(free {free:.1f} GB)")
settings = Settings()
emb = SentenceTransformerEmbedder.from_settings(settings, device=device, batch_size=64)
t = time.time()
vectors = np.asarray(emb.embed(texts), dtype=np.float32)
print(
    "embedded",
    vectors.shape,
    f"in {time.time() - t:.0f} s; model",
    settings.embedding_model,
    settings.embedding_model_revision,
)
digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
np.savez(f"{S}/emb.npz", vectors=vectors, texts=np.asarray(texts, dtype=object), digest=digest)
print("norms", float(np.linalg.norm(vectors, axis=1).mean()))
