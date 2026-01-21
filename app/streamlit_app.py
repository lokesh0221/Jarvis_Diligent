from __future__ import annotations

import streamlit as st

from src.chat_service import ChatService
from src.config import settings
from src.voice import VoiceAssistant


st.set_page_config(page_title="Jarvis Assistant", page_icon="🤖", layout="wide")


@st.cache_resource
def get_chat_service() -> ChatService:
    return ChatService()


@st.cache_resource
def get_voice_assistant(model_name: str) -> VoiceAssistant:
    return VoiceAssistant(stt_model=model_name)


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
    enable_voice = st.checkbox("Enable voice input/output", value=False)
    stt_model = st.selectbox("Speech-to-text model", ["tiny", "base", "small"], index=0)
    system_prompt = st.text_area(
        "System prompt",
        value="You are a concise enterprise copilot.",
        height=120,
    )
    st.markdown("Use scripts/ingest.py to ingest documents into Pinecone before chatting.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

audio_prompt = ""
voice = None
if enable_voice:
    voice = get_voice_assistant(stt_model)
    st.subheader("Voice Input")
    audio_bytes = st.audio_input("Record your question")
    if audio_bytes:
        audio_prompt = voice.transcribe_audio_bytes(audio_bytes.getvalue())
        if audio_prompt:
            st.info(f"Transcribed: {audio_prompt}")

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

        if enable_voice and voice:
            voice_bytes = voice.text_to_speech(answer)
            if voice_bytes:
                st.audio(voice_bytes, format="audio/wav")

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

if enable_voice and audio_prompt:
    st.session_state.messages.append({"role": "user", "content": audio_prompt})
    service.store_message("user", audio_prompt)
    with st.chat_message("user"):
        st.markdown(audio_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, matches = service.ask(
                audio_prompt,
                top_k=int(top_k),
                namespace=namespace,
                system_prompt=system_prompt,
            )
        st.markdown(answer)

        if enable_voice and voice:
            voice_bytes = voice.text_to_speech(answer)
            if voice_bytes:
                st.audio(voice_bytes, format="audio/wav")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    service.store_message("assistant", answer)
