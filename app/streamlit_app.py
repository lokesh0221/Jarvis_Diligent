from __future__ import annotations

import streamlit as st

from src.chat_service import ChatService
from src.config import settings


st.set_page_config(page_title="Jarvis Assistant", page_icon="🤖", layout="wide")


@st.cache_resource
def get_chat_service() -> ChatService:
    return ChatService()


service = get_chat_service()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about your enterprise knowledge base."}
    ]

st.title("Jarvis Assistant")

with st.sidebar:
    st.subheader("Settings")
    model = st.text_input("LLM model", value=settings.llm_model)
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=settings.top_k)
    namespace = st.text_input("Namespace", value=settings.pinecone_namespace)
    system_prompt = st.text_area(
        "System prompt",
        value="You are a concise enterprise copilot.",
        height=120,
    )
    st.markdown("Use scripts/ingest.py to ingest documents into Pinecone before chatting.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    service.store_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, matches = service.ask(
                prompt,
                top_k=int(top_k),
                namespace=namespace,
                system_prompt=system_prompt,
            )
        st.markdown(answer)

        if matches:
            with st.expander("Sources"):
                for match in matches:
                    metadata = match.get("metadata", {})
                    source = metadata.get("source", "unknown")
                    text = metadata.get("text", "")
                    st.markdown(f"**{source}**")
                    if text:
                        st.markdown(text)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    service.store_message("assistant", answer)
