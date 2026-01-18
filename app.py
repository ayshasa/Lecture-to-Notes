# final.py — AI Lecture Voice-to-Notes Generator (Unified Edition)

import streamlit as st
import whisper
import tempfile
import os
import json
import subprocess
from datetime import datetime
from typing import List
import google.generativeai as genai

from sentence_transformers import SentenceTransformer, util

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="AI Lecture Voice-to-Notes",
    page_icon="🎙️",
    layout="wide"
)

DATA_DIR = "lectures"
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------- API --------------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# -------------------- MODELS (CACHED) --------------------
@st.cache_resource
def load_models():
    return (
        whisper.load_model("base"),
        SentenceTransformer("all-MiniLM-L6-v2")
    )

whisper_model, embedder = load_models()

# -------------------- SIDEBAR --------------------
st.sidebar.title("📚 Lecture Manager")

lectures = sorted(os.listdir(DATA_DIR))
selected = st.sidebar.selectbox("Saved Lectures", ["New Lecture"] + lectures)

# Delete lecture
if selected != "New Lecture":
    if st.sidebar.button("🗑️ Delete selected lecture"):
        os.remove(os.path.join(DATA_DIR, selected))
        st.sidebar.success("Lecture deleted")
        st.experimental_rerun()

# Output language
st.sidebar.subheader("🌍 Output Language")
languages = ["English", "Malayalam", "Hindi", "Tamil", "Kannada", "Other"]
lang_choice = st.sidebar.selectbox("Select language", languages)
custom_lang = ""
if lang_choice == "Other":
    custom_lang = st.sidebar.text_input("Enter preferred language")

final_lang = custom_lang if custom_lang else lang_choice

# Search
search_query = st.sidebar.text_input("🔍 Semantic search")

# Options
eli5 = st.sidebar.checkbox("🧒 Explain Like I'm 5")
trim_audio = st.sidebar.checkbox("🎧 Remove silence before transcription", value=True)

# -------------------- UTILS --------------------
def save_lecture(data):
    fname = f"{data['title']}_{data['timestamp']}.json"
    with open(os.path.join(DATA_DIR, fname), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_lecture(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)

def preprocess_audio(path):
    out = path + "_clean.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-af", "silenceremove=1:0:-50dB", out],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return out if os.path.exists(out) else path
    except:
        return path

def semantic_search(text, query):
    t_emb = embedder.encode(text, convert_to_tensor=True)
    q_emb = embedder.encode(query, convert_to_tensor=True)
    return util.cos_sim(q_emb, t_emb).item()

def chunk_segments(segments, max_words=120):
    chunks = []
    current = ""
    start, end = None, None

    for seg in segments:
        words = seg["text"].split()
        if start is None:
            start = seg["start"]

        if len(current.split()) + len(words) <= max_words:
            current += " " + seg["text"]
            end = seg["end"]
        else:
            chunks.append((start, end, current.strip()))
            current = seg["text"]
            start, end = seg["start"], seg["end"]

    if current:
        chunks.append((start, end, current.strip()))

    return chunks

def chapter_detection(chunks):
    texts = [c[2] for c in chunks]
    embeds = embedder.encode(texts)

    chapters = [[chunks[0]]]

    for i in range(1, len(chunks)):
        sim = util.cos_sim(embeds[i-1], embeds[i])
        if sim < 0.65:
            chapters.append([chunks[i]])
        else:
            chapters[-1].append(chunks[i])

    return chapters

def generate_ai(text):
    mode = "Explain everything in very simple words for a child." if eli5 else "Explain clearly for exam preparation."

    prompt = f"""
{mode}

Output language: {final_lang}

From the lecture below, generate:
• Summary
• Definitions
• Important formulas (if any)
• Important exam points
• 5 quiz questions with answers
• 5 flashcards

Lecture:
{text}
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    return model.generate_content(prompt).text

# -------------------- MAIN UI --------------------
st.title("🎙️ AI Lecture Voice-to-Notes Generator")

# ---------- VIEW SAVED LECTURE ----------
if selected != "New Lecture":
    data = load_lecture(selected)
    content = data["content"]

    if search_query:
        score = semantic_search(content, search_query)
        if score < 0.3:
            st.warning("⚠️ Low relevance match for your search query.")

    st.markdown(content)
    st.stop()

# ---------- UPLOAD NEW ----------
uploaded = st.file_uploader("Upload lecture audio/video", ["mp3", "wav", "mp4"])

if uploaded and st.button("🚀 Generate Notes"):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name

    if trim_audio:
        with st.spinner("🔊 Preprocessing audio..."):
            path = preprocess_audio(path)

    with st.spinner("📝 Transcribing with Whisper..."):
        result = whisper_model.transcribe(path)

    segments = result["segments"]
    detected_lang = result.get("language", "unknown")

    st.success(f"Transcription completed! Detected language: {detected_lang}")

    chunks = chunk_segments(segments)
    chapters = chapter_detection(chunks)

    # Sidebar chapter list
    st.sidebar.subheader("📑 Chapters")
    for i in range(len(chapters)):
        st.sidebar.markdown(f"- Chapter {i+1}")

    # Build transcript
    transcript = ""
    for i, ch in enumerate(chapters, 1):
        transcript += f"\n## 📌 Chapter {i}\n"
        for s, e, t in ch:
            transcript += f"[{s:.2f}-{e:.2f}] {t}\n"

    # AI generation
    with st.spinner("🤖 Generating AI study materials..."):
        ai_notes = generate_ai(transcript)

    final_text = transcript + "\n\n---\n\n" + ai_notes

    st.markdown(final_text)

    # Save
    save_lecture({
        "title": uploaded.name.replace(" ", "_"),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "language": detected_lang,
        "content": final_text
    })

    st.success("✅ Lecture saved successfully!")

    # Cleanup
    try:
        os.remove(path)
    except:
        pass
