import os
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa
import librosa.display
import soundfile as sf
import pywt
from scipy.signal import resample_poly

# =========================
# Config
# =========================
AUDIO_FILE = '250825_002_clean (08_25).wav'
OUTPUT_ROOT = 'robot_movements_dataset'

# Split sensitivity
TOP_DB = 30
FRAME_LENGTH = 2048
HOP_LENGTH = 512

# Merge control
MAX_GAP_S = 0.03
MAX_MERGED_DUR_S = 2.5

# Which intervals to export
USE_MERGED_FILTERED = True  # True = use merged + duration-filtered; False = use RAW intervals (no merge, no dur filter)

# Duration filtering
AUTO_DURATION = True
MIN_DUR_S = 0.05
MAX_DUR_S = 1.50
PCTL_LOW = 10
PCTL_HIGH = 90
PCTL_EXPAND = 0.10

# Output options
SAVE_PER_CLIP = False
SAVE_SCALOGRAMS = False
EXPORT_SINGLE_FILE = True
PAD_SILENCE_S = 0.05          # silence inserted between clips in the concatenated file

# Scalogram settings
TARGET_SR = 16000
FMIN = 50.0
FMAX = None
VOICES_PER_OCT = 24
WAVELET = 'cmor1.5-1.0'

# =========================
# Audio helpers
# =========================
def preprocess_audio_clip(clip, original_sr, target_sr=TARGET_SR):
    if original_sr != target_sr:
        g = math.gcd(int(original_sr), int(target_sr))
        clip = resample_poly(clip, target_sr // g, original_sr // g).astype('float32')
    clip = clip - np.mean(clip)
    max_abs = np.max(np.abs(clip)) or 1.0
    clip = (clip / max_abs).astype('float32')
    return clip, target_sr

def make_freq_grid(fs, fmin=FMIN, fmax=None, voices_per_octave=VOICES_PER_OCT):
    if fmax is None:
        fmax = min(fs / 2.0, 8000.0)
    n_oct = np.log2(fmax / fmin)
    n_voices = int(np.ceil(n_oct * voices_per_octave))
    freqs = fmax / (2 ** (np.arange(n_voices) / voices_per_octave))
    return freqs[::-1].astype('float32')

def calculate_cwt_power(x, fs, freqs, wavelet=WAVELET):
    central_freq = pywt.central_frequency(wavelet)
    scales = (central_freq * fs / freqs).astype('float32')
    coefs, _ = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / fs)
    power = (np.abs(coefs) ** 2).astype('float32')
    t = np.arange(len(x), dtype='float32') / fs
    return power, t

def save_scalogram_plot(power, freqs, t, file_path, title="Wavelet Scalogram"):
    power_db = 10.0 * np.log10(power + 1e-12)
    vmin = np.percentile(power_db, 1)
    vmax = np.percentile(power_db, 99)
    fig, ax = plt.subplots(figsize=(10, 4))
    extent = [t[0], t[-1], freqs[0], freqs[-1]]
    im = ax.imshow(power_db, extent=extent, origin="lower", aspect="auto",
                   cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Power (dB)")
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close(fig)

# =========================
# Interval helpers
# =========================
def merge_intervals(intervals, sr, max_gap_s=MAX_GAP_S, max_merged_dur_s=MAX_MERGED_DUR_S):
    intervals = np.asarray(intervals, dtype=int)
    if intervals.size == 0:
        return []
    intervals = intervals[np.argsort(intervals[:, 0])]
    merged = [intervals[0].tolist()]
    max_gap = int(max_gap_s * sr)
    max_len = int(max_merged_dur_s * sr)

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        gap = start - last_end
        if gap <= max_gap:
            candidate_end = max(last_end, end)
            if candidate_end - last_start <= max_len:
                merged[-1][1] = candidate_end
            else:
                merged.append([start, end])
        else:
            merged.append([start, end])

    return [tuple(x) for x in merged]

def describe_durations(intervals, sr, label):
    intervals = np.asarray(intervals, dtype=int)
    if intervals.size == 0:
        print(f"{label}: none")
        return None
    durs = np.array(((intervals[:, 1] - intervals[:, 0]) / sr), dtype=float)
    print(
        f"{label} dur stats (s): "
        f"min={durs.min():.3f}, p10={np.percentile(durs,10):.3f}, "
        f"median={np.median(durs):.3f}, p90={np.percentile(durs,90):.3f}, "
        f"max={durs.max():.3f}, n={len(durs)}"
    )
    return durs

# =========================
# Main
# =========================
def process_and_save_movements(audio_path, y, sr, base_output_dir=OUTPUT_ROOT, top_db=TOP_DB):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    audio_clips_dir = os.path.join(run_dir, "audio_clips")
    scalograms_dir = os.path.join(run_dir, "scalograms")
    os.makedirs(audio_clips_dir, exist_ok=True)
    os.makedirs(scalograms_dir, exist_ok=True)
    print(f"Created output directory: {run_dir}")

    print(f"Detecting movements with top_db={top_db}, frame_length={FRAME_LENGTH}, hop_length={HOP_LENGTH}...")
    raw = librosa.effects.split(y, top_db=top_db, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    print(f"Found {len(raw)} raw sound events.")

    # Merge and describe
    merged = merge_intervals(raw, sr, max_gap_s=MAX_GAP_S, max_merged_dur_s=MAX_MERGED_DUR_S)
    print(f"Merged to {len(merged)} intervals with max_gap_s={MAX_GAP_S:.3f} and max_merged_dur_s={MAX_MERGED_DUR_S:.2f}")
    describe_durations(raw, sr, "Raw")
    durs_merged = describe_durations(merged, sr, "Merged")

    # Choose duration window (for merged)
    use_manual = (durs_merged is None) or (len(merged) < 20)
    if AUTO_DURATION and not use_manual:
        lo = np.percentile(durs_merged, PCTL_LOW)
        hi = np.percentile(durs_merged, PCTL_HIGH)
        span = max(hi - lo, 1e-3)
        min_dur = max(0.05, lo - PCTL_EXPAND * span)
        max_dur = hi + PCTL_EXPAND * span
        print(f"AUTO duration window: [{min_dur:.3f}, {max_dur:.3f}] s")
    else:
        min_dur, max_dur = MIN_DUR_S, MAX_DUR_S
        print(f"Manual duration window: [{min_dur:.3f}, {max_dur:.3f}] s")

    # Build the interval set to export
    if USE_MERGED_FILTERED:
        filtered = []
        for s0, s1 in merged:
            dur = (s1 - s0) / sr
            if min_dur <= dur <= max_dur:
                filtered.append((s0, s1))
        intervals_to_export = filtered
        print(f"Kept {len(filtered)} merged intervals after duration filtering.")
    else:
        # Use raw intervals directly
        intervals_to_export = [tuple(x) for x in np.asarray(raw, dtype=int)]
        print(f"Using RAW intervals directly: {len(intervals_to_export)}")

    # Diagnostic plot
    print("Saving diagnostic plot of detected movements...")
    plt.figure(figsize=(20, 5))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    for s0, s1 in merged:
        plt.axvspan(s0 / sr, s1 / sr, color='orange', alpha=0.15)
    for s0, s1 in intervals_to_export:
        plt.axvspan(s0 / sr, s1 / sr, color='green', alpha=0.35)
    plt.title(f"Detected Movements (merged={len(merged)}, export={len(intervals_to_export)})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    diag_path = os.path.join(run_dir, "diagnostic_intervals.png")
    plt.savefig(diag_path, dpi=200)
    plt.close()

    if len(intervals_to_export) == 0:
        print("No movements to save. If you want everything, set USE_MERGED_FILTERED=False.")
        return

    # -----------------------------------
    # Option A: save a single concatenated file
    # -----------------------------------
    if EXPORT_SINGLE_FILE:
        print("Building single concatenated WAV...")
        pad = np.zeros(int(PAD_SILENCE_S * sr), dtype=y.dtype) if PAD_SILENCE_S > 0 else None

        chunks = []
        cue_rows = []
        concat_cursor = 0.0
        for i, (s0, s1) in enumerate(intervals_to_export, start=1):
            clip = y[s0:s1]
            chunks.append(clip)
            dur = (s1 - s0) / sr

            cue_rows.append([i, s0 / sr, s1 / sr, concat_cursor, concat_cursor + dur, dur])
            concat_cursor += dur
            if pad is not None and i < len(intervals_to_export):
                chunks.append(pad)
                concat_cursor += PAD_SILENCE_S

        concat_audio = np.concatenate(chunks) if len(chunks) else np.array([], dtype=y.dtype)
        all_wav = os.path.join(run_dir, "all_movements.wav")
        sf.write(all_wav, concat_audio, sr)

        # Save cue sheet
        import csv
        cue_csv = os.path.join(run_dir, "all_movements_cues.csv")
        with open(cue_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "src_start_s", "src_end_s", "concat_start_s", "concat_end_s", "duration_s"])
            for row in cue_rows:
                w.writerow(row)

        print(f"Saved single file: {all_wav}")
        print(f"Saved cue sheet:   {cue_csv}")

    # -----------------------------------
    # Option B: also save per-clip files and scalograms (optional)
    # -----------------------------------
    if SAVE_PER_CLIP or SAVE_SCALOGRAMS:
        print(f"\nProcessing and saving {len(intervals_to_export)} movements...")
        for i, (start, end) in enumerate(tqdm(intervals_to_export, desc="Processing Movements")):
            clip_num = i + 1
            y_clip = y[start:end]
            duration = (end - start) / sr

            if SAVE_PER_CLIP:
                clip_path = os.path.join(audio_clips_dir, f"movement_{clip_num}.wav")
                sf.write(clip_path, y_clip, sr)

            if SAVE_SCALOGRAMS:
                processed_clip, fs = preprocess_audio_clip(y_clip.astype('float32'), sr)
                freqs = make_freq_grid(fs, fmin=FMIN, fmax=FMAX, voices_per_octave=VOICES_PER_OCT)
                power, t = calculate_cwt_power(processed_clip, fs, freqs, wavelet=WAVELET)
                scalogram_path = os.path.join(scalograms_dir, f"movement_{clip_num}_scalogram.png")
                title = f"Movement {clip_num} Scalogram (fs={fs} Hz, dur={duration:.2f} s)"
                save_scalogram_plot(power, freqs, t, scalogram_path, title=title)

    print("\nProcessing complete!")
    print(f"All files saved in: {run_dir}")
    print(f"- Diagnostic: {diag_path}")
    if EXPORT_SINGLE_FILE:
        print("- all_movements.wav and all_movements_cues.csv present")
    if SAVE_PER_CLIP:
        print(f"- Clips: {audio_clips_dir}")
    if SAVE_SCALOGRAMS:
        print(f"- Scalograms: {scalograms_dir}")

if __name__ == "__main__":
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Audio file not found at '{AUDIO_FILE}'")
    else:
        y, sr = librosa.load(AUDIO_FILE, sr=None, mono=True)
        process_and_save_movements(AUDIO_FILE, y, sr, base_output_dir=OUTPUT_ROOT, top_db=TOP_DB)
