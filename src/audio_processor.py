import streamlit as st
from moviepy.editor import VideoFileClip
import librosa
import numpy as np
import os
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

def extract_audio(video_path, output_dir="outputs"):
    """Extract audio from video."""
    audio_path = os.path.join(output_dir, "audio.wav")
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.close()
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

@st.cache_resource
def load_siren_model(model_path="resources/models/siren_cnn.h5"):
    """Load pre-trained siren detection model."""
    if load_model is None:
        print("TensorFlow not installed")
        return None
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"Error loading siren model: {e}")
        return None

def detect_siren(audio_path, model_path="resources/models/siren_cnn.h5"):
    """Detect siren in audio file."""
    try:
        y, sr = librosa.load(audio_path)
        y_harmonic, _ = librosa.effects.hpss(y)  # Noise reduction
        mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13, hop_length=int(sr*0.01))
        features = mfccs.T
        if features.shape[0] < 100:
            return False
        features = features[:100][np.newaxis, ..., np.newaxis]
        model = load_siren_model(model_path)
        if model is None:
            return False
        predictions = model.predict(features)
        return np.mean(predictions) > 0.5
    except Exception as e:
        print(f"Error detecting siren: {e}")
        return False

# Optional: Training code (uncomment to train a new model)
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
def create_siren_cnn(input_shape=(100, 13, 1)):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
# Example: model = create_siren_cnn(); model.fit(train_data, train_labels, epochs=10); model.save("siren_cnn.h5")
"""

if __name__ == "__main__":
    audio_path = extract_audio("test_video.mp4")
    if audio_path:
        siren_detected = detect_siren(audio_path)
        print(f"Siren detected: {siren_detected}")