# AI Music Remix & Mood Generator (Streamlit)
This project provides a Streamlit UI where students can upload tracks, apply mood presets, tweak tempo/pitch, and optionally call a remote AI service to fully resynthesize the track.


## Quick start:

1. Install system dependency `ffmpeg` (required by pydub/librosa for some formats).
2. Create and activate a virtualenv:
   - ``python -m venv venv``
   - ``source venv/bin/activate``
     <br>
   For Windows:
   <br> ``venv\\Scripts\\activate``
4. ``pip install -r requirements.txt``
5. (Optional) Add API key in config.py or set REMOTE_AI_KEY and REMOTE_AI_URL environment variables.
6. Run Streamlit UI: ``streamlit run app_streamlit.py``
7. Run tests: ``pytest -q``
