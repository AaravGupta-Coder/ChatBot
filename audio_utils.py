import numpy as np
import librosa
import soundfile as sf
from scipy.signal import fftconvolve, butter, filtfilt


def load_audio(path, sr=44100, mono=True):
    y, sr = librosa.load(path, sr=sr, mono=mono)
    return y.astype(np.float32), sr

def save_audio(path, y, sr):
    sf.write(path, y.astype(np.float32), sr)

def apply_tempo(y, rate):
    if abs(rate - 1.0) < 1e-6:
        return y
    return librosa.effects.time_stretch(y, rate)

def apply_pitch(y, sr, n_steps):
    if n_steps == 0:
        return y
    return librosa.effects.pitch_shift(y, sr, n_steps)

def make_impulse(sr, duration=0.6, decay=3.0):
    length = int(sr * duration)
    ir = np.random.randn(length) * np.exp(-np.linspace(0, decay, length))
    ir = ir / (np.max(np.abs(ir)) + 1e-9)
    return ir

def apply_reverb(y, sr, wet=0.2, ir_dur=0.6):
    if wet <= 0:
        return y
    ir = make_impulse(sr, duration=ir_dur)
    conv = fftconvolve(y, ir, mode='full')[: len(y)]
    out = (1 - wet) * y + wet * conv
    out = out / (np.max(np.abs(out)) + 1e-9)
    return out

def _butter_filter(y, sr, cutoff, btype='low', order=4):
    nyq = sr / 2.0
    if cutoff <= 0 or cutoff >= nyq:
        return y
    b, a = butter(order, float(cutoff) / nyq, btype=btype)
    try:
        return filtfilt(b, a, y)
    except Exception:
        return y

def lowpass(y, sr, cutoff=8000):
    return _butter_filter(y, sr, cutoff, btype='low')

def highpass(y, sr, cutoff=120):
    return _butter_filter(y, sr, cutoff, btype='high')

def apply_mood(y, sr, mood):
    if mood == 'happy':
        y = apply_tempo(y, 1.07)
        y = apply_pitch(y, sr, 1)
        y = apply_reverb(y, sr, wet=0.18, ir_dur=0.5)
        y = highpass(y, sr, cutoff=120)
    elif mood == 'sad':
        y = apply_tempo(y, 0.92)
        y = apply_pitch(y, sr, -1)
        y = apply_reverb(y, sr, wet=0.45, ir_dur=1.0)
        y = lowpass(y, sr, cutoff=6000)
    elif mood == 'energetic':
        y = apply_tempo(y, 1.25)
        y = np.tanh(y * 1.2)
        y = apply_reverb(y, sr, wet=0.12, ir_dur=0.25)
    elif mood == 'chill':
        y = apply_tempo(y, 0.95)
        y = apply_reverb(y, sr, wet=0.3, ir_dur=0.8)
    return y
