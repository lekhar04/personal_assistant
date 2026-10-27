"""ChromaDB wrapper: one collection for long-term memories, one for document chunks.

The two collections are deliberately separate so memory retrieval can never return
document text and vice versa.
"""
from typing import Optional

from . import config


class RetrievalError(Exception):
    """Raised when the vector database fails (storage or search)."""


class VectorStore:
    def __init__(self, embedding_function=None, path=None):
        try:
            import chromadb
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path or config.CHROMA_DIR))
            ef = embedding_function or DefaultEmbeddingFunction()  # local MiniLM (ONNX), no API key
            meta = {"hnsw:space": "cosine"}
            self.memories = self._client.get_or_create_collection(
                "user_memories", embedding_function=ef, metadata=meta
            )
            self.documents = self._client.get_or_create_collection(
                "document_chunks", embedding_function=ef, metadata=meta
            )
        except Exception as e:
            raise RetrievalError(f"Could not initialise the vector database: {e}") from e

    # ---------- generic helpers ----------
    @staticmethod
    def _query(collection, query: str, k: int, max_distance: float, where: Optional[dict] = None):
        count = collection.count()
        if count == 0 or not query.strip():
            return []
        try:
            res = collection.query(
                query_texts=[query], n_results=min(k, count), where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise RetrievalError(f"Vector search failed: {e}") from e
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if dist <= max_distance:  # drop weak matches instead of feeding noise to the LLM
                hits.append({"text": doc, "metadata": meta or {}, "distance": dist})
        return hits

    # ---------- memories ----------
    def add_memory(self, memory_id: int, text: str, category: str) -> None:
        try:
            self.memories.add(
                ids=[f"mem-{memory_id}"], documents=[text],
                metadatas=[{"memory_id": memory_id, "category": category}],
            )
        except Exception as e:
            raise RetrievalError(f"Could not store memory embedding: {e}") from e

    def search_memories(self, query: str, k: int = None, max_distance: float = None) -> list[dict]:
        return self._query(
            self.memories, query, k or config.MEMORY_TOP_K,
            config.MEMORY_MAX_DISTANCE if max_distance is None else max_distance,
        )

    def find_duplicate_memory(self, text: str) -> bool:
        return bool(self._query(self.memories, text, 1, config.MEMORY_DEDUP_DISTANCE))

    def delete_memory(self, memory_id: int) -> None:
        try:
            self.memories.delete(ids=[f"mem-{memory_id}"])
        except Exception as e:
            raise RetrievalError(f"Could not delete memory embedding: {e}") from e

    # ---------- documents ----------
    def add_chunks(self, document_id: int, filename: str, chunks: list[tuple[str, int]],
                   conversation_id: int, scope: str) -> None:
        if not chunks:
            return
        try:
            batch = 100
            for i in range(0, len(chunks), batch):
                part = chunks[i : i + batch]
                self.documents.add(
                    ids=[f"doc{document_id}-{i + j}" for j in range(len(part))],
                    documents=[c[0] for c in part],
                    metadatas=[
                        {"document_id": document_id, "filename": filename, "page": c[1],
                         "conversation_id": conversation_id, "scope": scope}
                        for c in part
                    ],
                )
        except Exception as e:
            self.delete_document(document_id, ignore_errors=True)
            raise RetrievalError(f"Could not index document '{filename}': {e}") from e

    def search_documents(self, query: str, conversation_id: Optional[int],
                         k: int = None, max_distance: float = None) -> list[dict]:
        """Search chunks of documents visible to this conversation (its own + global)."""
        conds = [{"scope": "global"}]
        if conversation_id is not None:
            conds.append({"$and": [{"scope": "conversation"}, {"conversation_id": conversation_id}]})
        where = conds[0] if len(conds) == 1 else {"$or": conds}
        return self._query(
            self.documents, query, k or config.DOC_TOP_K,
            config.DOC_MAX_DISTANCE if max_distance is None else max_distance, where,
        )

    def delete_document(self, document_id: int, ignore_errors: bool = False) -> None:
        try:
            self.documents.delete(where={"document_id": document_id})
        except Exception as e:
            if not ignore_errors:
                raise RetrievalError(f"Could not delete document embeddings: {e}") from e
