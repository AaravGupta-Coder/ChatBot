import os
import tempfile
import numpy as np
import soundfile as sf

from audio_utils import load_audio, save_audio, apply_tempo, apply_pitch, apply_reverb, apply_mood


def make_sine(filename, dur=2.0, sr=22050, freq=440.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * freq * t)
    sf.write(filename, y.astype('float32'), sr)
    return filename


def test_tempo_changes():
    fd = tempfile.gettempdir()
    in_file = os.path.join(fd, 'test_sine.wav')
    make_sine(in_file, dur=2.0)
    y, sr = load_audio(in_file, sr=22050)
    y_fast = apply_tempo(y, 1.5)
    assert len(y_fast) < len(y) * 1.0


def test_pitch_and_reverb():
    fd = tempfile.gettempdir()
    in_file = os.path.join(fd, 'test_sine2.wav')
    make_sine(in_file, dur=2.0)
    y, sr = load_audio(in_file, sr=22050)
    y_p = apply_pitch(y, sr, 3)
    y_r = apply_reverb(y_p, sr, wet=0.3)
    assert len(y_r) == len(y) or abs(len(y_r) - len(y)) < 10


def test_mood_pipeline_runs():
    fd = tempfile.gettempdir()
    in_file = os.path.join(fd, 'test_sine3.wav')
    make_sine(in_file, dur=2.5)
    y, sr = load_audio(in_file, sr=22050)
    out = apply_mood(y, sr, 'happy')
    assert out is not None
