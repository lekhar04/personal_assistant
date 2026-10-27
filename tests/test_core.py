import hashlib
import re

import pytest

from src import config, db, documents
from src.chat_service import ChatService, parse_memory_json
from src.llm import normalize_messages
from src.vector_store import VectorStore


class FakeEF:
    """Deterministic bag-of-words hash embedding so tests need no model download."""

    def __init__(self):
        pass

    def __call__(self, input):
        out = []
        for text in input:
            v = [0.0] * 128
            for w in re.findall(r"[a-z]+", text.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 128] += 1.0
            out.append(v)
        return out

    @staticmethod
    def name():
        return "fake-ef"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return FakeEF()

    def embed_query(self, input):
        return self(input)

    def is_legacy(self):
        return False


class FakeLLM:
    def __init__(self):
        self.memory_json = "[]"
        self.last_system = None

    def complete(self, system, messages, **kw):
        return self.memory_json

    def stream(self, system, messages, **kw):
        self.last_system = system
        yield "Hello "
        yield "there"


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "MEMORY_MAX_DISTANCE", 0.9)
    monkeypatch.setattr(config, "DOC_MAX_DISTANCE", 0.9)
    db.init_db()
    llm = FakeLLM()
    svc = ChatService(VectorStore(embedding_function=FakeEF(), path=tmp_path / "chroma"), llm)
    svc.fake_llm = llm
    return svc


def run_turn(svc, cid, text):
    ctx = svc.prepare_turn(cid, text)
    reply = "".join(svc.stream_reply(ctx))
    return ctx, svc.finish_turn(ctx, reply)


def test_conversations_are_isolated(service):
    a, b = db.create_conversation(), db.create_conversation()
    run_turn(service, a, "message in chat A")
    run_turn(service, b, "message in chat B")
    assert [m["content"] for m in db.get_messages(a) if m["role"] == "user"] == ["message in chat A"]
    assert [m["content"] for m in db.get_messages(b) if m["role"] == "user"] == ["message in chat B"]
    assert db.get_conversation(a)["title"].startswith("message in chat A")


def test_delete_conversation_keeps_memories(service):
    cid = db.create_conversation()
    service.fake_llm.memory_json = '[{"content": "The user likes chess.", "category": "interest"}]'
    _, saved = run_turn(service, cid, "I really like chess")
    assert saved == ["The user likes chess."]
    service.delete_conversation(cid)
    assert db.get_conversation(cid) is None and db.get_messages(cid) == []
    assert len(db.list_memories()) == 1


def test_memory_across_conversations_and_dedup(service):
    c1 = db.create_conversation()
    service.fake_llm.memory_json = '[{"content": "The user wants to learn Rust.", "category": "goal"}]'
    run_turn(service, c1, "I want to learn Rust")
    run_turn(service, c1, "I want to learn Rust")  # duplicate must not be stored twice
    assert len(db.list_memories()) == 1
    c2 = db.create_conversation()
    ctx = service.prepare_turn(c2, "What should I learn Rust")
    assert any("Rust" in m for m in ctx.memories)
    assert "The user wants to learn Rust." in ctx.system_prompt


def test_no_memory_or_documents_message(service):
    cid = db.create_conversation()
    ctx = service.prepare_turn(cid, "Hello")
    assert ctx.memories == [] and "No relevant long-term memories" in ctx.system_prompt
    assert "No documents have been uploaded" in ctx.system_prompt


def test_document_rag_scoping(service):
    c1, c2 = db.create_conversation(), db.create_conversation()
    service.ingest_document(c1, "notes.txt", b"The warranty period for the Zephyr blender is two years.", False)
    service.ingest_document(c1, "shared.txt", b"Quarterly revenue grew by twelve percent in Q3.", True)
    ctx1 = service.prepare_turn(c1, "warranty period Zephyr blender")
    assert any(s["filename"] == "notes.txt" for s in ctx1.doc_sources)
    ctx2 = service.prepare_turn(c2, "warranty period Zephyr blender")
    assert not any(s["filename"] == "notes.txt" for s in ctx2.doc_sources)
    ctx3 = service.prepare_turn(c2, "quarterly revenue Q3 grew")
    assert any(s["filename"] == "shared.txt" for s in ctx3.doc_sources)
    service.delete_conversation(c1)
    assert [d["filename"] for d in db.list_documents(c2)] == ["shared.txt"]


def test_parse_memory_json():
    assert parse_memory_json("[]") == []
    assert parse_memory_json("garbage") == []
    out = parse_memory_json('```json\n[{"content":"The user is Sam.","category":"profile"},{"x":1}]\n```')
    assert out == [{"content": "The user is Sam.", "category": "profile"}]


def test_chunking_and_extraction():
    text = ". ".join(f"Sentence number {i} is here" for i in range(200)) + "."
    chunks = documents.chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 3 and all(len(c) <= 400 for c in chunks)
    with pytest.raises(documents.DocumentError):
        documents.extract_pages("a.exe", b"x")
    with pytest.raises(documents.DocumentError):
        documents.extract_pages("a.txt", b"")


def test_normalize_messages():
    msgs = [{"role": "assistant", "content": "x"}, {"role": "user", "content": "a"},
            {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}]
    out = normalize_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant"] and "a" in out[0]["content"] and "b" in out[0]["content"]
