import librosa
import numpy as np
import pywt
import math
from scipy.signal import resample_poly

def preprocess_audio_clip(clip, original_sr, target_sr=16000):
    if original_sr != target_sr:
        g = math.gcd(int(original_sr), int(target_sr))
        clip = resample_poly(clip, target_sr // g, original_sr // g).astype('float32')
    clip = clip - np.mean(clip)
    max_abs = np.max(np.abs(clip)) or 1.0
    clip = (clip / max_abs).astype('float32')
    return clip, target_sr

def make_freq_grid(fs, fmin=50.0, fmax=None, voices_per_octave=24):
    if fmax is None:
        fmax = min(fs / 2.0, 8000.0)
    n_oct = np.log2(fmax / fmin)
    n_voices = int(np.ceil(n_oct * voices_per_octave))
    freqs = fmax / (2 ** (np.arange(n_voices) / voices_per_octave))
    return freqs[::-1].astype('float32')

def calculate_cwt_power(x, fs, freqs, wavelet='cmor1.5-1.0'):
    central_freq = pywt.central_frequency(wavelet)
    scales = (central_freq * fs / freqs).astype('float32')
    coefs, _ = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / fs)
    power = (np.abs(coefs) ** 2).astype('float32')
    t = np.arange(len(x), dtype='float32') / fs
    return power, t

try:
    y, sr = librosa.load('250806_010_gain_80_cleaned_2.wav', sr=None, mono=True)
    print("Successfully loaded audio file.")

    # Take a small clip from the audio
    start_sample = 100000
    end_sample = 200000
    clip = y[start_sample:end_sample]

    processed_clip, fs = preprocess_audio_clip(clip, sr)
    freqs = make_freq_grid(fs)
    power, t = calculate_cwt_power(processed_clip, fs, freqs)
    print("Successfully calculated CWT.")

except Exception as e:
    print(f"Error calculating CWT: {e}")