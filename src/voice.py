from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import whisper
import pyttsx3


class VoiceAssistant:
    def __init__(self, stt_model: str = "tiny") -> None:
        self.model = whisper.load_model(stt_model)
        self.tts = pyttsx3.init()

    def transcribe_audio_bytes(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)
        try:
            result = self.model.transcribe(str(tmp_path))
            return str(result.get("text", "")).strip()
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def text_to_speech(self, text: str) -> Optional[bytes]:
        if not text.strip():
            return None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = Path(tmp.name)
        try:
            self.tts.save_to_file(text, str(tmp_path))
            self.tts.runAndWait()
            return tmp_path.read_bytes()
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
