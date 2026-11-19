"""In-memory stand-ins for the ChromaDB client and collections (no server, no network).

They mimic what the code under test relies on: ``get_collection`` raising when a name is
missing, ``create_collection(get_or_create=False)`` refusing an existing name, embeddings coming
back from ``get`` as a float64 array of the JSON-decoded float32 values (as the HTTP client
returns them), brute-force cosine ``query`` (optionally hiding some ids, as an approximate index
can), simple ``where`` filters, and a call log so tests can assert that nothing was written.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np

WRITE_CALLS = frozenset(
    {
        "create_collection",
        "get_or_create_collection",
        "delete_collection",
        "add",
        "upsert",
        "update",
        "delete",
        "modify",
    }
)
DEFAULT_HNSW = {"space": "cosine", "ef_construction": 100, "ef_search": 100, "max_neighbors": 16}


class NotFoundError(Exception):
    pass


def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    for key, condition in where.items():
        value = metadata.get(key)
        if isinstance(condition, dict):
            if "$in" in condition and value not in condition["$in"]:
                return False
            if "$eq" in condition and value != condition["$eq"]:
                return False
        elif value != condition:
            return False
    return True


class FakeCollection:
    def __init__(
        self,
        client: FakeClient,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self.name = name
        self.id = str(uuid.uuid4())
        self.metadata = dict(metadata) if metadata else None
        hnsw = dict(DEFAULT_HNSW)
        requested = dict((configuration or {}).get("hnsw") or {})
        if client.ignore_build_settings:
            # a server that keeps its defaults for the settings fixed at creation
            requested.pop("max_neighbors", None)
            requested.pop("ef_construction", None)
        hnsw.update(requested)
        self.configuration_json = {"hnsw": hnsw, "spann": None, "embedding_function": None}
        self._records: dict[str, tuple[np.ndarray, str | None, dict[str, Any] | None]] = {}
        self.hidden_ids: set[str] = set()  # never returned by query (an ANN recall loss)
        self.fail_queries = False
        self.query_rows: list[int] = []  # queries x n_results per query call

    def _log(self, method: str, **details: Any) -> None:
        self._client.calls.append((method, self.name, details))

    def count(self) -> int:
        self._log("count")
        return len(self._records)

    def add(self, ids, embeddings=None, documents=None, metadatas=None) -> None:
        self._log("add", n=len(ids))
        vectors = np.asarray(embeddings, dtype=np.float32)
        for position, item_id in enumerate(ids):
            if item_id in self._records:
                raise ValueError(f"duplicate id {item_id}")
            vector = vectors[position].copy()
            if self._client.normalise_vectors:
                vector = (vector / np.linalg.norm(vector)).astype(np.float32)
            self._records[str(item_id)] = (
                vector,
                documents[position] if documents is not None else None,
                dict(metadatas[position]) if metadatas is not None else None,
            )

    def upsert(self, **kwargs: Any) -> None:
        self._log("upsert")
        raise AssertionError("upsert must not be called")

    def update(self, **kwargs: Any) -> None:
        self._log("update")
        raise AssertionError("update must not be called")

    def delete(self, **kwargs: Any) -> None:
        self._log("delete")
        raise AssertionError("delete must not be called")

    def modify(self, **kwargs: Any) -> None:
        self._log("modify")
        raise AssertionError("modify must not be called")

    # test helpers (not part of the Chroma API)
    def tamper(self, item_id: str, *, embedding=None, document=None, metadata=None) -> None:
        vector, old_document, old_metadata = self._records[item_id]
        self._records[item_id] = (
            np.asarray(embedding, dtype=np.float32) if embedding is not None else vector,
            document if document is not None else old_document,
            metadata if metadata is not None else old_metadata,
        )

    def drop(self, item_id: str) -> None:
        del self._records[item_id]

    def _payload(self, item_ids: list[str], include: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {"ids": list(item_ids)}
        if "embeddings" in include:
            # the HTTP client decodes JSON floats and wraps them in np.array (float64)
            payload["embeddings"] = np.array(
                [json.loads(json.dumps(self._records[i][0].tolist())) for i in item_ids]
            )
        if "documents" in include:
            payload["documents"] = [self._records[i][1] for i in item_ids]
        if "metadatas" in include:
            payload["metadatas"] = [json.loads(json.dumps(self._records[i][2])) for i in item_ids]
        return payload

    def get(
        self, ids=None, where=None, limit=None, offset=None, include=("metadatas", "documents")
    ):
        self._log("get", n=len(ids) if ids is not None else limit, include=tuple(include))
        if ids is not None:
            selected = [str(item) for item in ids if str(item) in self._records]
        else:
            selected = [
                item_id
                for item_id, (_, _, metadata) in self._records.items()
                if _matches(metadata or {}, where)
            ]
            start = int(offset or 0)
            selected = selected[start : start + limit if limit is not None else None]
        return self._payload(selected, list(include))

    def query(self, query_embeddings, n_results=10, where=None, include=("distances",)):
        self._log("query", n=len(query_embeddings), n_results=n_results)
        if self.fail_queries:
            raise ConnectionError("chroma went away")
        self.query_rows.append(len(query_embeddings) * int(n_results))
        candidates = [
            item_id
            for item_id, (_, _, metadata) in self._records.items()
            if item_id not in self.hidden_ids and _matches(metadata or {}, where)
        ]
        if candidates:
            matrix = np.asarray([self._records[i][0] for i in candidates], dtype=np.float64)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        result: dict[str, Any] = {"ids": [], "distances": [], "documents": [], "metadatas": []}
        for query in np.asarray(query_embeddings, dtype=np.float64):
            if not candidates:
                order: list[int] = []
                distances: list[float] = []
            else:
                similarity = matrix @ (query / np.linalg.norm(query))
                order = list(np.argsort(-similarity, kind="stable")[: int(n_results)])
                distances = [float(1.0 - similarity[i]) for i in order]
            chosen = [candidates[i] for i in order]
            result["ids"].append(chosen)
            result["distances"].append(distances)
            result["documents"].append([self._records[i][1] for i in chosen])
            result["metadatas"].append([self._records[i][2] for i in chosen])
        return {key: value for key, value in result.items() if key == "ids" or key in include}


class FakeClient:
    def __init__(self, *, ignore_build_settings: bool = False, normalise_vectors: bool = False):
        self.collections: dict[str, FakeCollection] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.ignore_build_settings = ignore_build_settings
        self.normalise_vectors = normalise_vectors

    def methods(self) -> list[str]:
        return [method for method, _, _ in self.calls]

    def writes(self) -> list[tuple[str, str, dict[str, Any]]]:
        return [call for call in self.calls if call[0] in WRITE_CALLS]

    def get_version(self) -> str:
        return "1.0.0"

    def list_collections(self, limit=None, offset=None) -> list[FakeCollection]:
        self.calls.append(("list_collections", "", {}))
        return list(self.collections.values())

    def get_collection(self, name: str, embedding_function: Any = None, data_loader=None):
        self.calls.append(("get_collection", name, {}))
        if name not in self.collections:
            raise NotFoundError(f"Collection [{name}] does not exist")
        return self.collections[name]

    def create_collection(
        self,
        name: str,
        schema=None,
        configuration=None,
        metadata=None,
        embedding_function=None,
        data_loader=None,
        get_or_create: bool = False,
    ):
        self.calls.append(("create_collection", name, {"get_or_create": get_or_create}))
        if name in self.collections:
            if get_or_create:
                return self.collections[name]
            raise ValueError(f"Collection [{name}] already exists")
        collection = FakeCollection(self, name, metadata=metadata, configuration=configuration)
        self.collections[name] = collection
        return collection

    def get_or_create_collection(self, name: str, configuration=None, metadata=None, **kwargs):
        self.calls.append(("get_or_create_collection", name, {}))
        if name not in self.collections:
            self.collections[name] = FakeCollection(
                self, name, metadata=metadata, configuration=configuration
            )
        return self.collections[name]

    def delete_collection(self, name: str) -> None:
        self.calls.append(("delete_collection", name, {}))
        raise AssertionError("delete_collection must not be called")

    # test helper: a collection made outside the code under test (no call logged)
    def seed(self, name: str, *, metadata=None, configuration=None) -> FakeCollection:
        collection = FakeCollection(self, name, metadata=metadata, configuration=configuration)
        self.collections[name] = collection
        return collection
