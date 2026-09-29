import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from apt_classifier.attack.download import download_attack_bundle
from apt_classifier.attack.kb import AttackKnowledgeBase
from apt_classifier.attack.parse import parse_attack_bundle
from apt_classifier.config import Settings, get_settings
from apt_classifier.embeddings import Embedder, SentenceTransformerEmbedder
from apt_classifier.vectorstore import ChromaVectorStore, VectorStoreCounts


@dataclass(frozen=True, slots=True)
class IngestionResult:
    bundle_path: Path
    knowledge_base_path: Path
    knowledge_base: AttackKnowledgeBase
    vector_counts: VectorStoreCounts


def ingest(
    settings: Settings | None = None,
    *,
    force_download: bool = False,
    reset: bool = False,
    batch_size: int = 128,
    embedder: Embedder | None = None,
    vector_store: ChromaVectorStore | None = None,
) -> IngestionResult:
    active_settings = settings or get_settings()
    active_settings.ensure_directories()
    bundle_path = download_attack_bundle(active_settings, force=force_download)
    knowledge_base = parse_attack_bundle(bundle_path, version=active_settings.attack_version)
    knowledge_base_path = knowledge_base.save(active_settings.kb_path)
    if vector_store is None:
        active_embedder = embedder or SentenceTransformerEmbedder.from_settings(active_settings)
        vector_store = ChromaVectorStore(
            settings=active_settings,
            embedder=active_embedder,
            batch_size=batch_size,
        )
    vector_counts = vector_store.upsert(knowledge_base, reset=reset)
    return IngestionResult(
        bundle_path=bundle_path,
        knowledge_base_path=knowledge_base_path,
        knowledge_base=knowledge_base,
        vector_counts=vector_counts,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m apt_classifier.ingest")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = ingest(
        force_download=arguments.force_download,
        reset=arguments.reset,
        batch_size=arguments.batch_size,
    )
    knowledge_base = result.knowledge_base
    print(
        f"ATT&CK {knowledge_base.version}: "
        f"techniques={len(knowledge_base.techniques)} "
        f"groups={len(knowledge_base.groups)} "
        f"campaigns={len(knowledge_base.campaigns)} "
        f"software={len(knowledge_base.software)} "
        f"sentences={result.vector_counts.technique_sentences} "
        f"group_documents={result.vector_counts.group_documents} "
        f"actor_behaviors={result.vector_counts.actor_behavior_documents}"
    )
    return 0


run_ingestion = ingest


if __name__ == "__main__":
    raise SystemExit(main())
