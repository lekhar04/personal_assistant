"""Orchestrates one chat turn: retrieve -> build prompt -> generate -> persist -> extract memories."""
import json
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

from . import config, db, documents, prompts
from .llm import LLMClient, LLMError
from .vector_store import RetrievalError, VectorStore

_FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|i'll|i'd|my|me|mine|myself|we|our|call)\b", re.I)


@dataclass
class TurnContext:
    conversation_id: int
    user_text: str
    history: list[dict]
    system_prompt: str
    memories: list[str] = field(default_factory=list)
    doc_sources: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_memory_json(raw: str) -> list[dict]:
    """Tolerantly parse the extractor output into [{'content', 'category'}]."""
    raw = raw.strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    allowed = {"profile", "interest", "goal", "preference", "other"}
    out = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and isinstance(item.get("content"), str) and item["content"].strip():
            cat = str(item.get("category", "other")).lower()
            out.append({"content": item["content"].strip()[:300], "category": cat if cat in allowed else "other"})
    return out[:5]


class ChatService:
    def __init__(self, store: VectorStore, llm: LLMClient):
        self.store = store
        self.llm = llm

    # ---------------- conversations ----------------
    def delete_conversation(self, conversation_id: int) -> None:
        for doc in db.list_conversation_only_documents(conversation_id):
            self.store.delete_document(doc["id"])
        db.delete_conversation(conversation_id)

    # ---------------- documents ----------------
    def ingest_document(self, conversation_id: int, filename: str, data: bytes, make_global: bool) -> int:
        pages = documents.extract_pages(filename, data)
        chunks = documents.chunk_pages(pages)
        scope = "global" if make_global else "conversation"
        doc_id = db.add_document(conversation_id, filename, scope, len(chunks))
        try:
            self.store.add_chunks(doc_id, filename, chunks, conversation_id, scope)
        except RetrievalError:
            db.delete_document(doc_id)
            raise
        return len(chunks)

    def delete_document(self, document_id: int) -> None:
        self.store.delete_document(document_id)
        db.delete_document(document_id)

    # ---------------- memories ----------------
    def delete_memory(self, memory_id: int) -> None:
        self.store.delete_memory(memory_id)
        db.delete_memory(memory_id)

    def clear_memories(self) -> None:
        for m in db.list_memories():
            self.delete_memory(m["id"])

    def extract_and_store_memories(self, conversation_id: int, user_text: str, last_assistant: str = "") -> list[str]:
        """Ask the LLM which facts are worth keeping; store only new, non-duplicate ones."""
        if len(user_text) < 8 or not _FIRST_PERSON.search(user_text):
            return []  # cheap gate: no first-person statement -> nothing personal to remember
        context = f"Assistant's previous message (for context only): {last_assistant[:500]}\n\n" if last_assistant else ""
        raw = self.llm.complete(
            prompts.MEMORY_EXTRACTION_PROMPT,
            [{"role": "user", "content": f"{context}User's latest message:\n{user_text}"}],
            max_tokens=400, temperature=0.0,
        )
        saved = []
        for item in parse_memory_json(raw):
            if self.store.find_duplicate_memory(item["content"]):
                continue
            mem_id = db.add_memory(item["content"], item["category"], conversation_id)
            try:
                self.store.add_memory(mem_id, item["content"], item["category"])
            except RetrievalError:
                db.delete_memory(mem_id)
                raise
            saved.append(item["content"])
        return saved

    # ---------------- chat turn ----------------
    def prepare_turn(self, conversation_id: int, user_text: str) -> TurnContext:
        history = db.get_recent_messages(conversation_id, config.MAX_HISTORY_MESSAGES)
        warnings: list[str] = []

        # Short follow-ups ("and page 3?") embed poorly; add the previous user message for retrieval only.
        search_query = user_text
        if len(user_text.split()) < 6:
            prev = [m["content"] for m in history if m["role"] == "user"]
            if prev:
                search_query = f"{prev[-1]}\n{user_text}"

        memories: list[str] = []
        try:
            memories = [h["text"] for h in self.store.search_memories(search_query)]
        except RetrievalError as e:
            warnings.append(f"Long-term memory lookup failed, continuing without it. ({e})")

        doc_hits: list[dict] = []
        try:
            doc_hits = self.store.search_documents(search_query, conversation_id)
        except RetrievalError as e:
            warnings.append(f"Document search failed, continuing without it. ({e})")

        if doc_hits:
            doc_block = "\n\n".join(
                f"[{i}] (source: {h['metadata'].get('filename')}, p. {h['metadata'].get('page')})\n{h['text']}"
                for i, h in enumerate(doc_hits, 1)
            )
        elif db.list_documents(conversation_id):
            doc_block = prompts.NO_DOCUMENTS_NO_MATCH
        else:
            doc_block = prompts.NO_DOCUMENTS_NONE_UPLOADED

        system = prompts.SYSTEM_PROMPT.format(
            memories="\n".join(f"- {m}" for m in memories) if memories else prompts.NO_MEMORIES,
            documents=doc_block,
        )
        sources = [
            {"filename": h["metadata"].get("filename"), "page": h["metadata"].get("page")} for h in doc_hits
        ]
        return TurnContext(conversation_id, user_text, history, system, memories, sources, warnings)

    def stream_reply(self, ctx: TurnContext) -> Iterator[str]:
        messages = ctx.history + [{"role": "user", "content": ctx.user_text}]
        yield from self.llm.stream(ctx.system_prompt, messages)

    def finish_turn(self, ctx: TurnContext, reply: str) -> list[str]:
        """Persist both messages, auto-title new chats, then extract long-term memories.

        Returns newly saved memories. Memory failures never lose the chat: they're returned via ctx.warnings.
        """
        cid = ctx.conversation_id
        meta = {"memories": ctx.memories, "documents": ctx.doc_sources} if (ctx.memories or ctx.doc_sources) else None
        db.add_message(cid, "user", ctx.user_text)
        db.add_message(cid, "assistant", reply, meta)

        conv = db.get_conversation(cid)
        if conv and conv["title"] == config.DEFAULT_CHAT_TITLE:
            title = ctx.user_text.strip().replace("\n", " ")
            db.rename_conversation(cid, title[:40] + ("…" if len(title) > 40 else ""))

        last_assistant = next((m["content"] for m in reversed(ctx.history) if m["role"] == "assistant"), "")
        try:
            return self.extract_and_store_memories(cid, ctx.user_text, last_assistant)
        except (LLMError, RetrievalError, db.DatabaseError) as e:
            ctx.warnings.append(f"Could not update long-term memory this turn: {e}")
            return []
