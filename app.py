"""Streamlit UI for the personal AI assistant with multi-chat, long-term memory and document RAG."""
import streamlit as st

from src import config, db
from src.chat_service import ChatService
from src.db import DatabaseError
from src.documents import DocumentError
from src.llm import LLMClient, LLMError
from src.vector_store import RetrievalError, VectorStore

st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner="Loading databases and embedding model…")
def get_service() -> ChatService:
    db.init_db()
    return ChatService(VectorStore(), LLMClient())


try:
    service = get_service()
except (DatabaseError, RetrievalError) as e:
    st.error(f"Startup failed: {e}")
    st.stop()

ss = st.session_state
ss.setdefault("conversation_id", None)  # None = a new chat that is created on the first message
ss.setdefault("uploader_key", 0)
ss.setdefault("notices", [])


def ensure_conversation() -> int:
    if ss.conversation_id is None:
        ss.conversation_id = db.create_conversation()
    return ss.conversation_id


def safe(fn, *args, **kwargs):
    """Run a service call, surfacing known failures as UI errors. Returns (ok, result)."""
    try:
        return True, fn(*args, **kwargs)
    except (DatabaseError, RetrievalError, DocumentError, LLMError) as e:
        st.error(str(e))
        return False, None


def render_context_used(meta: dict):
    mems, docs = meta.get("memories") or [], meta.get("documents") or []
    if not (mems or docs):
        return
    with st.expander("Context used", expanded=False):
        if mems:
            st.markdown("**Long-term memories**")
            for m in mems:
                st.markdown(f"- {m}")
        if docs:
            st.markdown("**Document sources**")
            for s in dict.fromkeys((d["filename"], d["page"]) for d in docs):
                st.markdown(f"- {s[0]}, p. {s[1]}")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🤖 Assistant")
    if not config.llm_configured():
        st.warning(config.missing_key_message())

    if st.button("➕ New chat", use_container_width=True):
        ss.conversation_id = None
        st.rerun()

    st.subheader("Conversations")
    ok, convs = safe(db.list_conversations)
    for conv in convs or []:
        c1, c2 = st.columns([5, 1])
        active = conv["id"] == ss.conversation_id
        if c1.button(conv["title"], key=f"sel{conv['id']}", use_container_width=True,
                     type="primary" if active else "secondary"):
            ss.conversation_id = conv["id"]
            st.rerun()
        if c2.button("🗑", key=f"del{conv['id']}", help="Delete this conversation"):
            if safe(service.delete_conversation, conv["id"])[0]:
                if active:
                    ss.conversation_id = None
                st.rerun()
    if ok and not convs:
        st.caption("No conversations yet.")

    st.divider()
    st.subheader("📄 Documents")
    files = st.file_uploader(
        "Upload PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True,
        key=f"uploader_{ss.uploader_key}",
    )
    make_global = st.checkbox("Use in all conversations", value=False,
                              help="Otherwise the document is only used in the current conversation.")
    if files and st.button("Add to knowledge base", use_container_width=True):
        cid = ensure_conversation()
        added = 0
        with st.spinner("Indexing…"):
            for f in files:
                ok, n = safe(service.ingest_document, cid, f.name, f.getvalue(), make_global)
                if ok:
                    added += 1
                    ss.notices.append(f"Indexed {f.name} ({n} chunks).")
        if added:
            ss.uploader_key += 1
            st.rerun()

    ok, docs = safe(db.list_documents, ss.conversation_id)
    for d in docs or []:
        c1, c2 = st.columns([5, 1])
        c1.caption(f"{'🌐' if d['scope'] == 'global' else '💬'} {d['filename']} · {d['chunk_count']} chunks")
        if c2.button("✕", key=f"deldoc{d['id']}", help="Remove document"):
            if safe(service.delete_document, d["id"])[0]:
                st.rerun()

    st.divider()
    ok, mems = safe(db.list_memories)
    with st.expander(f"🧠 Long-term memory ({len(mems or [])})"):
        if not mems:
            st.caption("Nothing remembered yet. Tell me about yourself!")
        for m in mems or []:
            c1, c2 = st.columns([5, 1])
            c1.caption(f"{m['content']}  \n`{m['category']}`")
            if c2.button("✕", key=f"delmem{m['id']}", help="Forget this"):
                if safe(service.delete_memory, m["id"])[0]:
                    st.rerun()
        if mems and st.button("Forget everything", use_container_width=True):
            if safe(service.clear_memories)[0]:
                st.rerun()

# ------------------------------------------------------------------ main chat
st.header("Chat")
for n in ss.notices:
    st.toast(n)
ss.notices = []

if ss.conversation_id is not None:
    ok, history = safe(db.get_messages, ss.conversation_id)
    for msg in history or []:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                render_context_used(msg["meta"])
else:
    st.info("Start a new conversation below. I'll remember important things you tell me across chats.")

if prompt := st.chat_input("Message your assistant…"):
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        cid = ensure_conversation()
        ctx = service.prepare_turn(cid, prompt)
        with st.chat_message("assistant"):
            reply = st.write_stream(service.stream_reply(ctx))
        new_memories = service.finish_turn(ctx, reply)
    except (LLMError, DatabaseError) as e:
        st.error(f"{e} Your message was not saved - please try again.")
    else:
        ss.notices += [f"⚠️ {w}" for w in ctx.warnings]
        ss.notices += [f"🧠 Remembered: {m}" for m in new_memories]
        st.rerun()
