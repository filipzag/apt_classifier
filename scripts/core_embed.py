"""Embedding ranker beside the lean core (development, descriptive; the frozen core is not changed).

Each labelled store post is embedded once per (pin, level) on the GPU with Qwen3-Embedding-0.6B
(statements masked exactly as the core reads them, ``data/core/rows_<pin>_<level>.jsonl.gz``),
chunked to ``CHUNK_TOKENS`` and averaged. A report is embedded as a query (instruction prefix),
chunked and averaged. Per report, only posts the core may read count: dated strictly before the
report, own post and copies hidden, optionally the own publisher hidden too (the same rule as
``core_race.py`` for its sets and ``core_older.py`` for vendor).

Scores per ATT&CK group with at least one readable post:
- ``tfidf``: the core's cosine (``TextRanker``, no prior), recomputed here;
- ``emb_centroid``: cosine to the mean embedding of the group's readable posts;
- ``emb_max``: the best cosine over the group's readable posts;
- ``combined``: z(tfidf) + z(emb_centroid) over the groups with readable posts.

    scripts/core_watch.sh .venv/bin/python scripts/core_embed.py embed --pin before_2025 --level behavior
    scripts/core_watch.sh .venv/bin/python scripts/core_embed.py rank --set store_train --level behavior [--hide-publisher]
    .venv/bin/python scripts/core_embed.py score data/core/embed_store_train_behavior.json PRODUCT.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vestige_core.util import DATA_DIR, require_memory  # noqa: E402

MODEL = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_TOKENS = 512
INSTRUCTION = (
    "Instruct: Given a threat intelligence report describing an intrusion, retrieve earlier "
    "reports about the same threat actor\nQuery: "
)
ARMS = ("tfidf", "emb_centroid", "emb_max", "combined")
CORE = DATA_DIR / "core"
BENCH = DATA_DIR / "benchmarks"
OLDER_FILES = {
    "vendor": [BENCH / "vendor" / "reports.jsonl"],
}


# --------------------------------------------------------------------------- the model
class Embedder:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise SystemExit("no CUDA GPU: this runs on the GPU only")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        self.model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

    def chunks(self, statements: list[str], prefix: str = "") -> list[str]:
        """Statements packed in order into chunks of at most CHUNK_TOKENS tokens."""
        budget = CHUNK_TOKENS - (len(self.tok(prefix)["input_ids"]) if prefix else 0) - 2
        out, cur, used = [], [], 0
        for s in statements:
            n = len(self.tok(s, add_special_tokens=False)["input_ids"])
            if cur and used + n > budget:
                out.append(" ".join(cur))
                cur, used = [], 0
            cur.append(s)
            used += n
        if cur:
            out.append(" ".join(cur))
        return [prefix + c for c in out]

    def encode(self, texts: list[str], batch: int = 16) -> np.ndarray:
        torch = self.torch
        vecs = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch):
                enc = self.tok(
                    texts[i : i + batch],
                    padding=True,
                    truncation=True,
                    max_length=CHUNK_TOKENS,
                    return_tensors="pt",
                ).to("cuda")
                hidden = self.model(**enc).last_hidden_state
                last = hidden[:, -1]  # left padding: the last token pools the sequence
                vecs.append(torch.nn.functional.normalize(last.float(), dim=-1).cpu().numpy())
        return np.concatenate(vecs) if vecs else np.zeros((0, 1024), np.float32)

    def document(self, statements: list[str], prefix: str = "") -> np.ndarray:
        v = self.encode(self.chunks(statements, prefix)).mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-12)


def embed_path(pin: str, level: str) -> Path:
    return CORE / f"emb_{pin}_{level}.npz"


# --------------------------------------------------------------------------- embed the store
def embed(args: argparse.Namespace) -> int:
    require_memory()
    rows_path = CORE / f"rows_{args.pin}_{args.level}.jsonl.gz"
    by_unit: dict[tuple[str, str], list[str]] = defaultdict(list)
    with gzip.open(rows_path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        for line in handle:
            key, actor, text = json.loads(line)
            by_unit[(key, actor)].append(text)
    units = sorted(by_unit)
    emb = Embedder()
    start = time.time()
    vecs = np.zeros((len(units), emb.model.config.hidden_size), np.float32)
    for i, u in enumerate(units):
        vecs[i] = emb.document(by_unit[u])
        if i % 100 == 0:
            require_memory()
            print(f"{i}/{len(units)} {time.time() - start:.0f}s", flush=True)
    np.savez_compressed(
        embed_path(args.pin, args.level),
        keys=np.asarray([u[0] for u in units]),
        actors=np.asarray([u[1] for u in units]),
        vecs=vecs,
        fingerprint=np.asarray(header.get("fingerprint") or ""),
        model=np.asarray(MODEL),
    )
    print(json.dumps({"units": len(units), "seconds": round(time.time() - start)}))
    return 0


# --------------------------------------------------------------------------- rank reports
def _host(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _tfidf_cosine(ranker, text: str, w: np.ndarray) -> np.ndarray:
    """``TextRanker.rank`` up to the cosine (a copy: the frozen ranker returns only softmax)."""
    from scipy import sparse

    from vestige_core.rank import _l2

    n = len(ranker.actors)
    if not (text or "").strip() or not w.any():
        return np.zeros(n)
    diag = sparse.diags(w)
    df = np.asarray((sparse.csr_matrix(w) @ ranker.unit_df).todense()).ravel()
    keep_terms = (df >= int(ranker.params["min_df"])).astype(np.float32)
    label = sparse.csr_matrix(
        (np.ones(len(ranker.unit_actor), np.float32), (ranker.unit_actor, np.arange(len(w)))),
        shape=(n, len(w)),
    )
    documents = (label @ diag @ ranker.unit_counts @ sparse.diags(keep_terms)).tocsr()
    documents.eliminate_zeros()
    doc_df = np.bincount(documents.indices, minlength=documents.shape[1])
    idf = (np.log((1.0 + n) / (1.0 + doc_df)) + 1.0).astype(np.float32)
    documents.data = np.log1p(documents.data)
    documents = _l2(documents @ sparse.diags(idf))
    row = ranker.vectorizer.transform([text]).tocsr()
    if row.nnz == 0:
        return np.zeros(n)
    row.data = np.log1p(row.data)
    row = _l2(row @ sparse.diags(idf))
    return np.asarray((row @ documents.T.tocsc()).todense(), dtype=np.float64)[0]


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def _load_reports(set_name: str, kb, level: str, extra):
    from vestige_core.sets import load_reports, load_set

    urls: dict[str, str] = {}
    if set_name in OLDER_FILES:
        reports = []
        for path in OLDER_FILES[set_name]:
            reports += load_reports(path, kb, level, extra_names=extra)
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    urls[str(row["report_id"])] = str(row.get("url") or "")
        return reports, urls
    return load_set(BENCH / set_name, kb, level, extra_names=extra), urls


def rank(args: argparse.Namespace) -> int:
    require_memory()
    from apt_classifier.attack.kb import AttackKnowledgeBase
    from apt_classifier.config import Settings
    from apt_classifier.knowledge_store.members import StoreMembers, members_path
    from apt_classifier.knowledge_store.pins import pin_for
    from apt_classifier.knowledge_store.sources import PUBLISHERS
    from vestige_core.corpus import load_rows
    from vestige_core.rank import TextRanker

    settings = Settings()
    kb = AttackKnowledgeBase.load(settings.kb_path)
    with gzip.open(members_path(settings), "rt", encoding="utf-8") as handle:
        names = tuple(json.load(handle).get("masking_names") or ())
    extra = names if args.level == "behavior" else ()
    reports, urls = _load_reports(args.set, kb, args.level, extra)
    db = sqlite3.connect(f"file:{DATA_DIR / 'knowledge_store' / 'store.sqlite'}?mode=ro", uri=True)
    post_host = {k: _host(u or "") for k, u in db.execute("select key, url from posts")}
    groups = tuple(sorted(kb.groups))
    gindex = {g: i for i, g in enumerate(groups)}
    emb = Embedder()
    out, cache = [], {}
    start = time.time()
    for report in sorted(reports, key=lambda r: pin_for(r.as_of)):
        pin = pin_for(report.as_of)
        if pin not in cache:
            cache.clear()
            require_memory()
            members = StoreMembers.load(members_path(settings, pin), kb)
            rows = load_rows(members, kb, pin, args.level, extra)
            ranker = TextRanker(rows, members.posts, groups, None)
            data = np.load(embed_path(pin, args.level))
            if str(data["fingerprint"]) != members.fingerprint:
                raise SystemExit(f"embeddings of {pin} were built from other store members")
            ekeys, eactors, evecs = list(data["keys"]), list(data["actors"]), data["vecs"]
            edates = np.asarray(
                [str(members.posts.get(k, {}).get("available_from") or "9999") for k in ekeys]
            )
            eg = np.asarray([gindex.get(a, -1) for a in eactors])
            cache[pin] = (members, ranker, ekeys, eg, evecs, edates)
        members, ranker, ekeys, eg, evecs, edates = cache[pin]
        hidden = set(members.hidden_for(report.store_keys, report.raw_statements))
        if args.hide_publisher:
            if args.set in OLDER_FILES:
                host = _host(urls.get(report.report_id, ""))
                hidden |= {k for k in members.posts if host and post_host.get(k) == host}
            else:
                publisher = PUBLISHERS.get(report.source) or report.source
                hidden |= {
                    k
                    for k, p in members.posts.items()
                    if (PUBLISHERS.get(str(p["source"])) or str(p["source"])) == publisher
                }
        text = " ".join(report.statements)
        cos_t = _tfidf_cosine(ranker, text, ranker.weights(report.as_of, hidden))
        q = emb.document(list(report.statements), INSTRUCTION)
        readable = (
            (edates < report.as_of) & (eg >= 0) & np.asarray([k not in hidden for k in ekeys])
        )
        sims = evecs[readable] @ q
        gs = eg[readable]
        has = np.zeros(len(groups), bool)
        has[gs] = True
        cen = np.zeros((len(groups), evecs.shape[1]), np.float32)
        np.add.at(cen, gs, evecs[readable])
        cen[has] /= np.linalg.norm(cen[has], axis=1, keepdims=True) + 1e-12
        cos_c = cen @ q
        cos_m = np.full(len(groups), -1.0)
        np.maximum.at(cos_m, gs, sims)
        idx = np.flatnonzero(has)
        scores = {
            "tfidf": cos_t[idx],
            "emb_centroid": cos_c[idx],
            "emb_max": cos_m[idx],
            "combined": _z(cos_t[idx]) + _z(cos_c[idx]),
        }
        truth = report.group_id
        row = {
            "report_id": report.report_id,
            "group_id": truth,
            "accept": sorted(report.accept),
            "published": report.published,
            "pin": pin,
            "readable_groups": int(len(idx)),
        }
        for arm, s in scores.items():
            order = idx[np.lexsort((idx, -s))]
            names_ = [groups[i] for i in order]
            row[arm] = {
                "first": names_[0] if names_ else None,
                "truth_rank": (names_.index(truth) + 1) if truth in names_ else None,
                "top5": names_[:5],
            }
        out.append(row)
        if len(out) % 25 == 0:
            require_memory()
            print(f"{len(out)}/{len(reports)} {time.time() - start:.0f}s", flush=True)
    name = f"embed_{args.set}_{args.level}{'_pubhidden' if args.hide_publisher else ''}"
    path = CORE / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "set": args.set,
                "level": args.level,
                "hide_publisher": args.hide_publisher,
                "model": MODEL,
                "seconds": round(time.time() - start),
                "rows": out,
            },
            indent=1,
        )
    )
    known = [r for r in out if r["group_id"]]
    print(
        json.dumps(
            {
                "file": str(path),
                "known": len(known),
                **{a: sum(_right(r[a]["first"], r) for r in known) for a in ARMS},
            }
        )
    )
    return 0


def _right(actor, r) -> bool:
    return actor is not None and (actor == r["group_id"] or actor in r["accept"])


# --------------------------------------------------------------------------- score
def score(args: argparse.Namespace) -> int:
    from vestige_core.evaluate import paired

    data = json.loads(Path(args.file).read_text())
    rows = [r for r in data["rows"] if r["group_id"]]
    product = None
    if args.product:
        p = json.loads(Path(args.product).read_text())
        product = {str(r["report_id"]): r.get("leading") for r in (p.get("known") or [])}
        rows = [r for r in rows if r["report_id"] in product]
    out = {"file": args.file, "known": len(rows)}
    for arm in ARMS:
        ranks = [r[arm]["truth_rank"] for r in rows]
        out[arm] = {
            "first": sum(_right(r[arm]["first"], r) for r in rows),
            "top5": sum(1 for k in ranks if k is not None and k <= 5),
            "vs_tfidf": paired(
                rows,
                lambda r, a=arm: _right(r[a]["first"], r),
                lambda r: _right(r["tfidf"]["first"], r),
            ),
        }
        if product is not None:
            out[arm]["vs_product"] = paired(
                rows,
                lambda r, a=arm: _right(r[a]["first"], r),
                lambda r: _right(product[r["report_id"]], r),
            )
    if product is not None:
        out["product_first"] = sum(_right(product[r["report_id"]], r) for r in rows)
    print(json.dumps(out, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--pin", choices=["before_2025", "before_2026_03"], required=True)
    e.add_argument("--level", choices=["strict", "behavior"], required=True)
    r = sub.add_parser("rank")
    r.add_argument("--set", choices=["store_train", "fresh", "fresh2", *OLDER_FILES], required=True)
    r.add_argument("--level", choices=["strict", "behavior"], required=True)
    r.add_argument("--hide-publisher", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("file")
    s.add_argument("product", nargs="?")
    args = parser.parse_args()
    return {"embed": embed, "rank": rank, "score": score}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
