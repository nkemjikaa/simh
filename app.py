import streamlit as st
import numpy as np
import json
import tempfile
import sounddevice as sd
import soundfile as sf
import os

from dejavu import Dejavu
from dejavu.logic.recognizer.file_recognizer import FileRecognizer
from dejavu.config.config_reader import read

from process_audio import get_embedding
from cosine_similarity import get_top_matches

# === CONFIG ===
DURATION = 10  # seconds
SAMPLE_RATE = 48000
CHANNELS = 1
EMBEDDING_PATH = "models/song_embeddings.npy"
METADATA_PATH = "models/metadata.json"

# === LOAD EMBEDDINGS & METADATA ===
embeddings = np.load(EMBEDDING_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# === STREAMLIT UI ===
st.set_page_config(page_title="SIMH – Stuck In My Head")
st.title("🎵 SIMH: Stuck In My Head")
st.markdown("Hum, whistle, beatbox, or sing – let's find that song.")

# === RECORD BUTTON ===
if st.button("🎙️ Record 10s Clip"):
    st.info("Recording...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, recording, SAMPLE_RATE)
        st.audio(tmp.name)

        try:
            config = read("dejavu.cnf.MySQL.yaml")
            djv = Dejavu(config)
            result = djv.recognize(FileRecognizer, tmp.name)

            if result and result.get("results"):
                top = result["results"][0]

                # Clean byte-string style names
                raw = top.get("song_name", "")
                if isinstance(raw, bytes):
                    raw_name = raw.decode("utf-8")
                else:
                    raw_name = raw.strip("b'\"")
                artist = raw_name.split(" - ")[0] if " - " in raw_name else "Unknown"
                song_title = raw_name.split(" - ")[1] if " - " in raw_name else raw_name

                st.success(f"**{song_title}** by **{artist}**")
                st.caption(f"Confidence: {round(top.get('input_confidence', 0) * 100, 2)}% | Offset: {round(top.get('offset_seconds', 0), 2)}s")

                st.write("🔎 Raw match result:", result)
            else:
                st.warning("❌ No match found.")
        except Exception as e:
            st.error(f"Failed to identify song: {e}")
