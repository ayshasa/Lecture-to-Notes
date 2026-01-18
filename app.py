import streamlit as st
import tempfile
import os
import whisper
import google.generativeai as genai

# ---------------- CONFIG ----------------

st.set_page_config(page_title="AI Lecture to Notes", layout="wide")

st.title("🎤 AI Lecture Voice-to-Notes")

# Load Whisper model
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

# Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------- FUNCTIONS ----------------

def transcribe_audio(uploaded_file):
    # Save uploaded file to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Transcribe using Whisper
    result = whisper_model.transcribe(temp_path)
    os.remove(temp_path)

    return result["text"]


def generate_notes(transcript):
    prompt = f"""
    Convert the following lecture transcript into well-structured study notes.
    Use headings, bullet points, and make it easy to study.

    Transcript:
    {transcript}
    """

    response = gemini_model.generate_content(prompt)
    return response.text


# ---------------- UI ----------------

uploaded = st.file_uploader(
    "Upload lecture audio/video",
    type=["mp3", "wav", "mp4", "m4a", "mpeg"]
)

if uploaded and st.button("🚀 Generate Notes"):

    with st.spinner("📝 Transcribing audio..."):
        transcript = transcribe_audio(uploaded)

    st.success("✅ Transcription completed!")

    with st.expander("📜 Show Transcript"):
        st.write(transcript)

    with st.spinner("🤖 Generating notes with AI..."):
        notes = generate_notes(transcript)

    st.success("✅ Notes generated!")

    st.subheader("📚 Your Notes")
    st.markdown(notes)

    # Download button
    st.download_button(
        "⬇️ Download Notes",
        notes,
        file_name="lecture_notes.txt"
    )
