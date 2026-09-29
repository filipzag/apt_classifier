from apt_classifier.attack.kb import AttackKnowledgeBase, Group, Technique
from apt_classifier.vectorstore import group_documents, technique_documents


def test_vector_documents_are_sentence_granular_and_deterministic():
    kb = AttackKnowledgeBase(
        techniques={
            "T1": Technique(
                id="T1",
                name="Technique One",
                stix_id="attack-pattern--1",
                description="First behavior. Second behavior.",
                tactics=["execution"],
                platforms=["Windows"],
                sentences={
                    "name": ["Technique One"],
                    "description": ["First behavior.", "Second behavior."],
                    "detection": [],
                },
            )
        },
        groups={
            "G1": Group(
                id="G1",
                name="Group One",
                stix_id="intrusion-set--1",
                technique_ids=["T1"],
                technique_behaviors={
                    "T1": ["Group One loaded its payload through a signed binary proxy."]
                },
            )
        },
    )
    techniques = technique_documents(kb)
    groups = group_documents(kb)
    assert [document.id for document in techniques] == [
        "T1:name:0000",
        "T1:description:0000",
        "T1:description:0001",
        "T1:procedure:0000",
    ]
    procedure = techniques[-1]
    assert procedure.metadata["field"] == "procedure"
    assert "Group One" not in procedure.document
    assert "the adversary" in procedure.document
    assert {document.metadata["document_type"] for document in groups} == {
        "actor_behavior",
        "description",
        "technique_profile",
    }
    behavior = next(
        document for document in groups if document.metadata["document_type"] == "actor_behavior"
    )
    assert behavior.id == "G1:actor-behavior:T1:0000"
    assert behavior.metadata["technique_id"] == "T1"


def _kb_with_procedures(procedures: dict[str, list[str]]) -> AttackKnowledgeBase:
    return AttackKnowledgeBase(
        techniques={
            "T1": Technique(
                id="T1",
                name="Technique One",
                stix_id="attack-pattern--1",
                description="First behavior.",
                sentences={"name": ["Technique One"], "description": ["First behavior."]},
            )
        },
        groups={
            f"G{index}": Group(
                id=f"G{index}",
                name=name,
                stix_id=f"intrusion-set--{index}",
                aliases=["Fancy Wolf"] if index == 1 else [],
                technique_ids=["T1"],
                technique_behaviors={"T1": sentences},
            )
            for index, (name, sentences) in enumerate(procedures.items(), start=1)
        },
    )


def test_procedure_documents_sanitize_aliases_and_deduplicate():
    kb = _kb_with_procedures(
        {
            "Group One": [
                "Fancy Wolf's implant used PowerShell to download tools.(Citation: X)",
                "Fancy Wolf's implant used PowerShell to download tools.",
            ],
            "Group Two": ["FANCY WOLF's implant used powershell to download tools."],
        }
    )
    procedures = [doc for doc in technique_documents(kb) if doc.metadata["field"] == "procedure"]
    assert len(procedures) == 1
    assert procedures[0].document == "the adversary's implant used PowerShell to download tools."
    assert "Fancy Wolf" not in procedures[0].document
    assert procedures[0].metadata["technique_id"] == "T1"


def test_procedure_documents_skip_unknown_techniques():
    kb = _kb_with_procedures({"Group One": ["Used a shell."]})
    kb.groups["G1"].technique_behaviors["T9999"] = ["Unknown technique text."]
    documents = technique_documents(kb)
    assert all(doc.metadata["technique_id"] == "T1" for doc in documents)


class _FakeCollection:
    def __init__(self, metadata: dict, count: int) -> None:
        self.metadata = metadata
        self._count = count

    def count(self) -> int:
        return self._count


class _FakeClient:
    def __init__(self, collection: _FakeCollection) -> None:
        self.collection = collection

    def get_or_create_collection(self, **kwargs):
        return self.collection

    def heartbeat(self) -> int:
        return 1


def _store(collection: _FakeCollection):
    from types import SimpleNamespace

    from apt_classifier.config import Settings
    from apt_classifier.vectorstore import ChromaVectorStore

    settings = Settings(_env_file=None)
    embedder = SimpleNamespace(embed=lambda texts: [[0.0, 1.0] for _ in texts])
    return ChromaVectorStore(settings=settings, embedder=embedder, client=_FakeClient(collection))


def test_stale_technique_index_schema_fails_before_retrieval():
    import pytest

    store = _store(_FakeCollection({"kind": "technique_sentences", "schema_version": 2}, count=10))
    with pytest.raises(RuntimeError, match="incompatible ATT&CK technique vector index"):
        store.query_techniques(["whoami"])
    assert not store._technique_index_is_compatible(store.technique_collection)


def test_compatible_or_empty_technique_index_is_accepted():
    from apt_classifier.vectorstore import TECHNIQUE_FIELDS, TECHNIQUE_SCHEMA_VERSION

    empty = _store(_FakeCollection({}, count=0))
    assert empty._technique_index_is_compatible(empty.technique_collection)
    settings = empty.settings
    current = _store(
        _FakeCollection(
            {
                "kind": "technique_sentences",
                "schema_version": TECHNIQUE_SCHEMA_VERSION,
                "attack_version": settings.attack_version,
                "embedding_model": settings.embedding_model,
                "embedding_revision": settings.embedding_model_revision,
                "fields": ",".join(TECHNIQUE_FIELDS),
            },
            count=5,
        )
    )
    assert current._technique_index_is_compatible(current.technique_collection)


class _RecordingCollection(_FakeCollection):
    def __init__(self, metadata: dict, count: int) -> None:
        super().__init__(metadata, count)
        self.upserts: list[dict] = []

    def upsert(self, **kwargs) -> None:
        self.upserts.append(kwargs)
        self._count += len(kwargs["ids"])

    def modify(self, *, metadata: dict) -> None:
        self.metadata = metadata


class _RecordingClient:
    def __init__(self, existing: dict[str, _RecordingCollection]) -> None:
        self.collections = existing
        self.deleted: list[str] = []

    def get_or_create_collection(self, *, name, metadata, **kwargs):
        if name not in self.collections:
            self.collections[name] = _RecordingCollection(dict(metadata), 0)
        return self.collections[name]

    def delete_collection(self, *, name) -> None:
        self.deleted.append(name)
        self.collections.pop(name, None)


def test_upsert_rebuilds_incompatible_technique_index_and_records_schema():
    from types import SimpleNamespace

    from apt_classifier.config import Settings
    from apt_classifier.vectorstore import (
        TECHNIQUE_COLLECTION,
        TECHNIQUE_SCHEMA_VERSION,
        ChromaVectorStore,
    )

    kb = _kb_with_procedures({"Group One": ["Fancy Wolf ran whoami."]})
    kb.version = "19.2"
    stale = _RecordingCollection({"kind": "technique_sentences", "schema_version": 2}, count=3)
    client = _RecordingClient({TECHNIQUE_COLLECTION: stale})
    store = ChromaVectorStore(
        settings=Settings(_env_file=None),
        embedder=SimpleNamespace(embed=lambda texts: [[0.0, 1.0] for _ in texts]),
        client=client,
    )
    counts = store.upsert(kb)
    assert TECHNIQUE_COLLECTION in client.deleted
    rebuilt = client.collections[TECHNIQUE_COLLECTION]
    assert rebuilt is not stale
    assert rebuilt.metadata["schema_version"] == TECHNIQUE_SCHEMA_VERSION
    assert rebuilt.metadata["attack_kb_version"] == "19.2"
    assert rebuilt.metadata["procedure_documents"] == 1
    assert counts.technique_sentences == 3
    assert store._technique_index_is_compatible(rebuilt)


def test_synthetic_sentences_become_an_indexed_field(tmp_path):
    from apt_classifier.attack.kb import AttackKnowledgeBase, Technique
    from apt_classifier.vectorstore import (
        TECHNIQUE_FIELDS,
        load_synthetic_sentences,
        technique_documents,
    )

    assert "synthetic" in TECHNIQUE_FIELDS
    assert load_synthetic_sentences(tmp_path / "missing.json") == {}
    path = tmp_path / "synthetic_sentences.json"
    path.write_text(
        '{"metadata": {"model": "x"}, "sentences": {"T1059.001": ["PowerShell decoded a stager.", " "], "T9999": ["orphan"]}}',
        encoding="utf-8",
    )
    synthetic = load_synthetic_sentences(path)
    assert synthetic == {"T1059.001": ["PowerShell decoded a stager."], "T9999": ["orphan"]}
    kb = AttackKnowledgeBase(
        techniques={
            "T1059.001": Technique(
                id="T1059.001", name="PowerShell", description="Adversaries may abuse PowerShell."
            )
        }
    )
    documents = technique_documents(kb, synthetic)
    fields = {document.metadata["field"] for document in documents}
    assert "synthetic" in fields
    synthetic_docs = [d for d in documents if d.metadata["field"] == "synthetic"]
    assert len(synthetic_docs) == 1 and synthetic_docs[0].id == "T1059.001:synthetic:0000"
    assert all(d.metadata["field"] != "synthetic" for d in technique_documents(kb))
