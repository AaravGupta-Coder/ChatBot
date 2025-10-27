import os
import tempfile
import uuid
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from audio_utils import load_audio, save_audio, apply_mood, apply_tempo, apply_pitch, apply_reverb
import config
import requests

st.set_page_config(page_title="AI Music Remix & Mood Generator", layout="wide")

st.title("🎧 AI Music Remix & Mood Generator")
st.markdown("Upload a track, choose a mood, tweak tempo & pitch, and press **Remix**.")

col1, col2 = st.columns([2,1])

with col1:
    uploaded = st.file_uploader("Upload audio (wav / mp3)", type=["wav","mp3","m4a","flac"], accept_multiple_files=False)
    mood = st.selectbox("Mood preset", options=["neutral","happy","sad","energetic","chill"], index=0)
    tempo = st.slider("Tempo multiplier", 0.5, 1.6, 1.0, 0.01)
    pitch = st.slider("Pitch shift (semitones)", -12, 12, 0, 1)
    use_remote = st.checkbox("Use remote AI resynthesis (requires REMOTE_AI_URL or config API key)")
    remix_btn = st.button("Remix 🎛️")

with col2:
    st.markdown("**Preview / Controls**")
    out_audio_placeholder = st.empty()
    download_placeholder = st.empty()


def call_remote_ai(input_path: str, mood: str, tempo_mult: float, pitch_semitones: int) -> str:
    REMOTE_AI_URL = os.environ.get('REMOTE_AI_URL', '')
    REMOTE_AI_KEY = os.environ.get('REMOTE_AI_KEY', '') or config.AI_API_KEY
    if not REMOTE_AI_URL:
        raise RuntimeError("REMOTE_AI_URL not configured")
    headers = {}
    if REMOTE_AI_KEY:
        headers['Authorization'] = f'Bearer {REMOTE_AI_KEY}'
    files = {'audio': open(input_path, 'rb')}
    data = {'mood': mood, 'tempoMultiplier': str(tempo_mult), 'pitch': str(pitch_semitones)}
    resp = requests.post(REMOTE_AI_URL, headers=headers, files=files, data=data, timeout=300)
    resp.raise_for_status()
    out_path = os.path.join(tempfile.gettempdir(), f"remix_remote_{uuid.uuid4().hex}.wav")
    with open(out_path, 'wb') as f:
        f.write(resp.content)
    return out_path


def visualize_waveform(y, sr, title='Waveform'):
    fig, ax = plt.subplots(figsize=(8,2))
    times = np.arange(len(y)) / float(sr)
    ax.plot(times, y, linewidth=0.5)
    ax.set_xlim(0, times[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    st.pyplot(fig)


def visualize_spectrogram(y, sr, title='Spectrogram'):
    fig, ax = plt.subplots(figsize=(8,3))
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title(title)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    st.pyplot(fig)


if uploaded:
    tmp_in = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}.{uploaded.name.split('.')[-1]}")
    with open(tmp_in, 'wb') as f:
        f.write(uploaded.read())

    y, sr = load_audio(tmp_in, sr=44100)
    st.header("Original")
    visualize_waveform(y, sr, title='Original Waveform')
    visualize_spectrogram(y, sr, title='Original Spectrogram')

    if remix_btn:
        st.info("Processing remix... this may take a while for remote AI")
        try:
            if use_remote:
                out_path = call_remote_ai(tmp_in, mood, tempo, pitch)
            else:
                y2 = y.copy()
                if abs(tempo - 1.0) > 1e-4:
                    y2 = apply_tempo(y2, tempo)
                if pitch != 0:
                    y2 = apply_pitch(y2, sr, pitch)
                y2 = apply_mood(y2, sr, mood)
                out_path = os.path.join(tempfile.gettempdir(), f"remix_local_{uuid.uuid4().hex}.wav")
                save_audio(out_path, y2, sr)

            out_audio_placeholder.audio(out_path)
            download_placeholder.markdown(f"[Download remix]({out_path})")
            st.success("Remix ready")
        except Exception as e:
            st.error(f"Remix failed: {e}")

else:
    st.info("Upload an audio file to get started")
