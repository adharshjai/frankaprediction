import librosa

try:
    y, sr = librosa.load('250806_010_gain_80_cleaned_2.wav', sr=None, mono=True)
    print("Successfully loaded audio file.")
    
    non_silent_intervals = librosa.effects.split(y, top_db=30)
    print(f"Successfully split audio file into {len(non_silent_intervals)} non-silent intervals.")

except Exception as e:
    print(f"Error splitting audio file: {e}")