# Personal AI Assistant (Streamlit + Gemini or Claude + ChromaDB)

Choose the LLM in `.env`: set `GEMINI_API_KEY` (with `LLM_PROVIDER=gemini`) or `ANTHROPIC_API_KEY`
(with `LLM_PROVIDER=anthropic`). Embeddings always run locally, so no embedding key is needed.

A chatbot with multiple independent conversations, long-term user memory and
document question answering (RAG).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your GEMINI_API_KEY (or ANTHROPIC_API_KEY) in .env
streamlit run app.py
```

The first run downloads a small local embedding model (MiniLM via ChromaDB, no API key needed).
Run the tests with `python -m pytest tests -q` (they use a fake LLM and fake embeddings, so no network).

## Features

- **Multiple chats**: new, switch, continue, delete. Titles are generated from the first message.
- **Short-term memory**: the last `MAX_HISTORY_MESSAGES` messages of the *current* conversation are sent to the LLM.
- **Long-term memory**: after each turn an LLM extractor decides whether the user stated a durable fact
  (name, interests, goals, preferences). Only those are saved, deduplicated by embedding similarity,
  and retrieved by semantic search in any later chat. View or delete them in the sidebar.
- **Document RAG**: upload PDF/TXT/MD files, chunked and embedded; answers cite `(source: file, p. N)`.
  Documents are per-conversation by default, or shared across all chats with a checkbox.
- **Grounding**: weak matches are dropped by a distance threshold, and the prompt tells the model to say so
  when nothing relevant was found rather than guess.
- **Transparency**: each answer has a "Context used" expander showing which memories and document pages were used.
- **Error handling**: API (auth, rate limit, connection), database, upload (size, type, encrypted/scanned PDFs)
  and retrieval failures show friendly messages. Retrieval/memory failures degrade gracefully instead of blocking chat.

## Architecture

| Concern | Storage | Module |
|---|---|---|
| Conversation history | SQLite `messages` (FK to one conversation) | `src/db.py` |
| Long-term user memory | SQLite `memories` + Chroma `user_memories` | `src/vector_store.py`, `src/chat_service.py` |
| Document knowledge | SQLite `documents` + Chroma `document_chunks` | `src/documents.py`, `src/vector_store.py` |

```
app.py                Streamlit UI
src/config.py         settings (env overridable)
src/db.py             SQLite layer
src/vector_store.py   ChromaDB (memories and document chunks in separate collections)
src/llm.py            Gemini + Anthropic clients, streaming, error mapping
src/prompts.py        system prompt and memory-extraction prompt
src/chat_service.py   per-turn orchestration
```

Each turn: retrieve memories -> retrieve document chunks -> build a system prompt with the three sources in
separate tagged blocks (history goes as normal chat turns) -> stream the answer -> save messages -> extract memories.

## Design notes

- Deleting a conversation removes its messages and conversation-scoped documents, but **keeps long-term memories**
  and global documents. Use the sidebar to forget memories.
- Memory extraction is skipped for messages with no first-person statement, to save an LLM call.
- Tune retrieval strictness with `MEMORY_MAX_DISTANCE` / `DOC_MAX_DISTANCE` (cosine distance, lower = stricter).
- Scanned (image-only) PDFs are not supported; add OCR if needed.
