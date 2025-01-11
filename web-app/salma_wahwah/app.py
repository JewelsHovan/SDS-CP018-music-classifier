import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import tempfile
import os
import joblib
import tensorflow as tf
import sounddevice as sd
from scipy.io.wavfile import write

# Set the parameters
duration = 30  # Duration of the recording in seconds
sample_rate = 44100  # Sample rate in Hz

# Load the trained model and scalers
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), 'music_genre_cnn_model_regularized.h5')
    scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')
    label_encoder_path = os.path.join(os.getcwd(), 'label_encoder.pkl')

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model()

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        rms = librosa.feature.rms(y=audio)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        
        features = np.hstack([ 
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1),
            np.mean(rms), np.std(rms),
            np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
        ])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to validate the model with an audio file
def validate_model(audio_file_path):
    features = extract_features(audio_file_path)
    if features is not None:
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_scaled = features_scaled.reshape(1, features_scaled.shape[1], 1)
        predictions = model.predict(features_scaled)
        predicted_class_index = np.argmax(predictions, axis=1)
        predicted_genre = label_encoder.inverse_transform(predicted_class_index)[0]
        confidence = np.max(predictions)
        return predicted_genre, confidence
    return None, None

# Streamlit UI
st.title("Music Genre Classification")
st.header("Record or Upload an Audio File to Classify the Genre")

# Function to record and save audio
def record_audio():
    try:
        st.info("Recording will start for 30 seconds. Please wait...")
        devices = sd.query_devices()
        input_device = None

        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # Look for devices with input capabilities
                input_device = idx
                break

        if input_device is None:
            st.error("No audio input device found.")
            return None

        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16', device=input_device)
        sd.wait()
        st.success("Recording finished!")
        
        temp_audio_path = "recorded_audio.wav"
        write(temp_audio_path, sample_rate, audio_data)
        st.success(f"Audio saved to {temp_audio_path}")
        return temp_audio_path

    except sd.PortAudioError as e:
        st.error(f"Error recording audio: {e}")
        return None

# Button to start recording
if st.button("Start Recording"):
    temp_audio_path = record_audio()
    if temp_audio_path:
        st.audio(temp_audio_path, format="audio/wav")

        predicted_genre, confidence = validate_model(temp_audio_path)
        if predicted_genre:
            st.success(f"Predicted Genre: {predicted_genre}")
            st.write(f"Confidence: {confidence:.2f}")
        else:
            st.error("Could not extract features from the recorded audio.")

# File upload option
audio_file = st.file_uploader("Or upload an audio file", type=["wav", "mp3"])

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        if audio_file.name.endswith(".mp3"):
            audio_data, sr = librosa.load(audio_file, sr=None)
            sf.write(temp_audio.name, audio_data, sr)
        else:
            temp_audio.write(audio_file.getvalue())
        temp_audio_path = temp_audio.name

    st.audio(temp_audio_path, format="audio/wav")

    predicted_genre, confidence = validate_model(temp_audio_path)
    if predicted_genre:
        st.success(f"Predicted Genre: {predicted_genre}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.error("Error extracting features from the uploaded audio file.")
