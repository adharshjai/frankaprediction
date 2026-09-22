import librosa

try:
    y, sr = librosa.load('250806_010_gain_80_cleaned_2.wav', sr=None, mono=True)
    print("Successfully loaded audio file.")
    print(f"Sample rate: {sr}")
    print(f"Number of samples: {len(y)}")
except Exception as e:
    print(f"Error loading audio file: {e}")