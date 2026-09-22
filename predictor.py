import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import soundfile as sf
from PIL import Image
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog


# =============================================================================
# --- 1. GUI FILE CHOOSER & DATA LOADING ---
# =============================================================================

def select_run_folder():
    """Opens a GUI dialog to select the data folder."""
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title='Please select the run folder from your dataset')
    root.destroy()
    return folder_path


def generate_ground_truth_labels(num_labels):
    """
    Generates the theoretical direction and speed labels to match the
    experiment plan in conductor.cpp.
    """
    speeds_to_test = [5, 6, 7, 8, 9, 10]  # Speeds in RPM
    reps_per_setting = 3
    experiment_plan = []

    for speed in speeds_to_test:
        for _ in range(reps_per_setting):
            experiment_plan.append({'direction': 0, 'speed': speed})  # 0 for cw
            experiment_plan.append({'direction': 1, 'speed': speed})  # 1 for ccw

    labels_in_rads = []
    for run in experiment_plan:
        speed_rads = (run['speed'] * 2 * np.pi) / 60.0
        labels_in_rads.append({'direction': run['direction'], 'speed': speed_rads})

    return labels_in_rads[:num_labels]


def load_and_label_data(run_folder_path, img_size=(64, 64)):
    """Loads audio clips and scalogram images and pairs them with generated labels."""
    audio_dir = os.path.join(run_folder_path, "audio_clips")
    scalo_dir = os.path.join(run_folder_path, "scalograms")

    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')],
                         key=lambda x: int(x.split('_')[1].split('.')[0]))

    num_movements = len(audio_files)
    if num_movements == 0:
        return None

    print(f"Found {num_movements} movements. Generating labels...")
    labels = generate_ground_truth_labels(num_movements)

    all_clips, all_scalograms, all_directions, all_speeds = [], [], [], []

    for i in tqdm(range(num_movements), desc="Loading data"):
        clip_num = i + 1
        audio_path = os.path.join(audio_dir, f"movement_{clip_num}.wav")
        clip, _ = sf.read(audio_path)
        all_clips.append(clip)

        scalo_path = os.path.join(scalo_dir, f"movement_{clip_num}_scalogram.png")
        img = Image.open(scalo_path).convert('RGB')
        img = img.resize(img_size)
        all_scalograms.append(np.array(img) / 255.0)

        all_directions.append(labels[i]['direction'])
        all_speeds.append(labels[i]['speed'])

    return all_clips, np.array(all_scalograms), np.array(all_directions), np.array(all_speeds)


# =============================================================================
# --- 2. TENSORFLOW & MODELING ---
# =============================================================================

def build_hybrid_model(input_shape_audio, input_shape_scalogram):
    """
    A true CRNN model that uses an LSTM for the audio branch and a simple CNN
    for the image branch.
    """
    print(f"Building Hybrid CRNN model...")

    # Audio branch (CRNN)
    audio_input = layers.Input(shape=input_shape_audio, name='audio_input')
    x1 = layers.Conv1D(16, 7, activation='relu', padding='causal')(audio_input)
    x1 = layers.MaxPooling1D(4)(x1)
    x1 = layers.Conv1D(32, 5, activation='relu', padding='causal')(x1)
    x1 = layers.MaxPooling1D(4)(x1)
    x1 = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x1)
    x1 = layers.Dropout(0.4)(x1)

    # Scalogram branch (CNN)
    scalo_input = layers.Input(shape=input_shape_scalogram, name='scalogram_input')
    x2 = layers.Conv2D(16, (3, 3), activation='relu')(scalo_input)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.Conv2D(32, (3, 3), activation='relu')(x2)
    x2 = layers.GlobalMaxPooling2D()(x2)
    x2 = layers.Dense(32, activation='relu')(x2)
    x2 = layers.Dropout(0.4)(x2)

    # Combine branches
    concatenated = layers.concatenate([x1, x2])
    dense = layers.Dense(64, activation='relu')(concatenated)
    dropout = layers.Dropout(0.5)(dense)

    # Output heads
    direction_output = layers.Dense(2, activation='softmax', name='direction_output')(dropout)
    speed_output = layers.Dense(1, name='speed_output')(dropout)

    model = models.Model(inputs=[audio_input, scalo_input], outputs=[direction_output, speed_output])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss={'direction_output': 'categorical_crossentropy', 'speed_output': 'mean_squared_error'},
                  metrics={'direction_output': 'accuracy', 'speed_output': 'mae'},
                  loss_weights={'direction_output': 0.5, 'speed_output': 1.0})
    return model


def plot_evaluation_results(y_true_dir, y_pred_dir, y_true_speed, y_pred_speed, history=None):
    """Plots the confusion matrix, regression results, and training history."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Confusion matrix
    cm = confusion_matrix(y_true_dir, y_pred_dir)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CW', 'CCW'])
    disp.plot(ax=axes[0], cmap=plt.cm.Blues)
    axes[0].set_title('Direction Classifier Accuracy')

    # Speed prediction
    axes[1].scatter(y_true_speed, y_pred_speed, alpha=0.6, edgecolors='w', label='Predictions')
    lims = [min(y_true_speed.min(), y_pred_speed.min()), max(y_true_speed.max(), y_pred_speed.max())]
    axes[1].plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction')
    axes[1].set_xlabel("Actual Speed (rad/s)")
    axes[1].set_ylabel("Predicted Speed (rad/s)")
    axes[1].set_title("Speed Prediction: Predicted vs. Actual")
    axes[1].legend()
    axes[1].grid(True)

    # Training history
    if history:
        axes[2].plot(history.history['loss'], label='Training Loss')
        axes[2].plot(history.history['val_loss'], label='Validation Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training History')
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()
    plt.show()


# =============================================================================
# --- 3. MAIN SCRIPT EXECUTION ---
# =============================================================================
if __name__ == "__main__":
    run_folder = 'robot_movements_dataset/run_2025-09-25_15-49-03'

    if not run_folder:
        print("No folder selected. Exiting.")
    else:
        print(f"Selected data folder: {run_folder}")
        data = load_and_label_data(run_folder)

        if data:
            clips, scalograms, directions, speeds = data

            # Pad the audio clips to a uniform length
            print("Padding audio clips...")
            max_len = max(c.shape[0] for c in clips) if clips[0].ndim > 1 else max(len(c) for c in clips)
            if clips[0].ndim > 1:  # Multi-channel audio
                padded_clips = np.array([np.pad(c, ((0, max_len - c.shape[0]), (0, 0)), 'constant') for c in clips])
            else:  # Mono audio
                padded_clips = np.array([np.pad(c, (0, max_len - len(c)), 'constant') for c in clips])
                padded_clips = np.expand_dims(padded_clips, axis=-1)

            # Normalize the speed values
            speed_scaler = MinMaxScaler()
            speeds_scaled = speed_scaler.fit_transform(speeds.reshape(-1, 1))

            # Split data
            X_train_clips, X_test_clips, X_train_scalo, X_test_scalo, y_train_dir, y_test_dir, y_train_speed, y_test_speed = train_test_split(
                padded_clips, scalograms, directions, speeds_scaled, test_size=0.2, random_state=42, stratify=directions
            )

            print(f"Train set: {X_train_clips.shape[0]} samples | Test set: {X_test_clips.shape[0]} samples")

            # Create TF Datasets
            train_ds = tf.data.Dataset.from_tensor_slices((
                {'audio_input': X_train_clips, 'scalogram_input': X_train_scalo},
                {'direction_output': tf.one_hot(y_train_dir, 2), 'speed_output': y_train_speed}
            )).batch(8).prefetch(tf.data.AUTOTUNE)

            test_ds = tf.data.Dataset.from_tensor_slices((
                {'audio_input': X_test_clips, 'scalogram_input': X_test_scalo},
                {'direction_output': tf.one_hot(y_test_dir, 2), 'speed_output': y_test_speed}
            )).batch(8).prefetch(tf.data.AUTOTUNE)

            # Build the hybrid model
            model = build_hybrid_model(padded_clips.shape[1:], scalograms.shape[1:])
            model.summary()

            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
            ]

            # Train model
            print("\n--- Starting Model Training ---")
            history = model.fit(train_ds, epochs=30, validation_data=test_ds, verbose=1, callbacks=callbacks)

            # Evaluate and plot
            print("\n--- Evaluating on Test Set ---")
            predict_ds = tf.data.Dataset.from_tensor_slices((
                {'audio_input': X_test_clips, 'scalogram_input': X_test_scalo}
            )).batch(8)

            dir_probs, speed_preds_scaled = model.predict(predict_ds)

            # Inverse transform the predictions to get the real speed values
            speed_preds = speed_scaler.inverse_transform(speed_preds_scaled)
            dir_preds = np.argmax(dir_probs, axis=1)
            y_test_speed_original = speed_scaler.inverse_transform(y_test_speed)

            plot_evaluation_results(y_test_dir, dir_preds, y_test_speed_original, speed_preds.flatten(), history)