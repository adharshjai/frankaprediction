# Franka Prediction

Predicts the **rotation direction** (clockwise or counter-clockwise) and **speed** of a Franka robot movement from the sound it makes.

There are two main steps:

1. **`data_prep.py`** takes one long audio recording, cuts it into clips with one movement each, and turns each clip into a wavelet **scalogram** image.
2. **`predictor.py`** trains a **hybrid neural network** on two inputs: the raw audio waveform and the scalogram image.

```
long .wav recording
        │
        ▼
 data_prep.py ──► movement_N.wav  (raw audio clip)
              └─► movement_N_scalogram.png  (time–frequency image)
        │
        ▼
 predictor.py ──► hybrid CRNN + CNN ──► direction (CW/CCW) + speed (rad/s)
```

---

## `data_prep.py`: cut the recording and make scalograms

### 1. Find the movements
The script loads the full recording (`AUDIO_FILE`) and runs `librosa.effects.split`. This marks every stretch of audio louder than `TOP_DB` (30 dB) below the peak as a sound event. Quiet gaps between movements become the cut points.

### 2. Merge fragments
One movement can break into several short events. `merge_intervals` joins events separated by less than `MAX_GAP_S` (30 ms), as long as the merged clip stays under `MAX_MERGED_DUR_S` (2.5 s).

### 3. Filter by duration
Clips that are too short or too long (clicks, noise, overlapping moves) are dropped. With `AUTO_DURATION = True`, the allowed range comes from the data itself: the 10th–90th percentile of clip lengths, widened by 10%. If there are fewer than 20 clips, the fixed range `MIN_DUR_S`–`MAX_DUR_S` is used instead.

### 4. Export
Each run is written to `robot_movements_dataset/run_<timestamp>/`:

| Output | Description |
|---|---|
| `diagnostic_intervals.png` | Waveform with the detected (orange) and kept (green) movements highlighted |
| `all_movements.wav` | All kept clips joined together, with short silences in between |
| `all_movements_cues.csv` | Where each clip came from in the original file and where it sits in the joined file |
| `audio_clips/movement_N.wav` | One file per movement (when `SAVE_PER_CLIP = True`) |
| `scalograms/movement_N_scalogram.png` | One scalogram per movement (when `SAVE_SCALOGRAMS = True`) |

### 5. Scalograms
A scalogram is the image produced by a **Continuous Wavelet Transform (CWT)**. It shows how much energy the sound has at each frequency over time, similar to a spectrogram. Wavelets give better time detail at high frequencies and better frequency detail at low frequencies, which suits short mechanical sounds.

For each clip:
- The audio is resampled to 16 kHz, the DC offset is removed, and the peak is normalized (`preprocess_audio_clip`).
- A log-spaced frequency grid runs from 50 Hz up to 8 kHz, with 24 steps per octave (`make_freq_grid`).
- A complex Morlet wavelet (`cmor1.5-1.0`) CWT is computed with PyWavelets. Power is |coefficients|² (`calculate_cwt_power`).
- Power is converted to dB, clipped to the 1st–99th percentile, and saved as a `turbo` colormap image (`save_scalogram_plot`).

> **Note:** `predictor.py` needs the per-clip WAVs and the scalograms. Set `SAVE_PER_CLIP = True` and `SAVE_SCALOGRAMS = True` before running data prep for training.

---

## `predictor.py`: the hybrid model

### Labels
Clips are labeled in recording order to match the experiment plan (`generate_ground_truth_labels`):
- speeds of 5–10 RPM,
- 3 repetitions at each speed,
- clockwise then counter-clockwise each time.

Speeds are converted to rad/s.

### Inputs
- **Raw audio:** each clip's waveform, zero-padded to the length of the longest clip.
- **Visual:** each clip's scalogram PNG, resized to 64×64 RGB and scaled to [0, 1].

### Architecture
The model has two branches that learn from different views of the same sound. Their outputs are joined before the prediction heads.

**Audio branch (CRNN):** works on the raw waveform
- Conv1D(16, k=7) → MaxPool(4) → Conv1D(32, k=5) → MaxPool(4)
- Bidirectional LSTM(32) captures how the sound changes over time
- Dropout 0.4

**Scalogram branch (CNN):** works on the time–frequency image
- Conv2D(16, 3×3) → MaxPool(2×2) → Conv2D(32, 3×3) → GlobalMaxPool
- Dense(32) → Dropout 0.4

**Fusion and outputs**
- Concatenate both branches → Dense(64) → Dropout 0.5
- `direction_output`: 2-class softmax (CW / CCW), categorical cross-entropy loss
- `speed_output`: a single regression value, MSE loss, trained on speeds scaled to 0–1 with MinMax scaling

The losses are weighted 0.5 for direction and 1.0 for speed. The optimizer is Adam (learning rate 1e-3).

### Training and evaluation
- 80/20 train/test split, stratified by direction, batch size 8, up to 30 epochs.
- Early stopping (patience 10) and learning-rate reduction on plateau (patience 5).
- Results are shown in three plots: a **confusion matrix** for direction, a **predicted vs. actual speed** scatter plot, and **training/validation loss** curves.

---

## Other files
- `predictorpi.py`, `predictorsj.py`: other versions of the predictor.
- `test_cwt.py`, `test_load.py`, `test_split.py`: small scripts for checking the CWT, data loading, and splitting.

## Usage
```bash
pip install numpy librosa soundfile pywavelets scipy matplotlib tqdm tensorflow scikit-learn pillow
```
1. Set `AUDIO_FILE` in `data_prep.py`, turn on `SAVE_PER_CLIP` and `SAVE_SCALOGRAMS`, then run:
   ```bash
   python data_prep.py
   ```
2. Set `run_folder` in `predictor.py` to the new `robot_movements_dataset/run_<timestamp>` folder, then run:
   ```bash
   python predictor.py
   ```
