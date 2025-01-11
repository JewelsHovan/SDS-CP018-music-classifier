import tensorflow as tf
import numpy as np
import librosa
import sys
from attention import MultiHeadSelfAttention
import matplotlib.pyplot as plt

class MusicGenrePredictor:
    """
    A class for predicting the genre of music from audio files.

    This class handles audio processing, model loading, and genre prediction.
    It uses a pre-trained TensorFlow Keras model with a custom MultiHeadSelfAttention layer.
    """
    def __init__(self):
        """
        Initializes the MusicGenrePredictor with model parameters and loads the pre-trained model.
        """
        # Constants for audio processing and model input
        self.IMG_HEIGHT = 128
        self.IMG_WIDTH = 128
        self.NUM_SEGMENTS = 7
        self.MODEL_PATH = 'prod_model.keras'
        self.GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                      'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        # Load model at initialization
        self.model = self._load_model()

    def _load_model(self):
        """
        Loads the pre-trained Keras model from the specified path.

        Returns:
            tf.keras.Model: The loaded Keras model.

        Raises:
            RuntimeError: If the model fails to load.
        """
        try:
            return tf.keras.models.load_model(
                self.MODEL_PATH,
                custom_objects={"MultiHeadSelfAttention": MultiHeadSelfAttention}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def process_audio_file(self, audio_file, sr=22050, duration=30):
        """
        Processes an audio file to extract mel spectrogram segments.

        Args:
            audio_file (str or file-like object): Path to the audio file or a file-like object.
            sr (int, optional): Sampling rate for audio loading. Defaults to 22050.
            duration (int, optional): Duration of audio to load in seconds. Defaults to 30.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Array of mel spectrogram segments, shape (NUM_SEGMENTS, IMG_HEIGHT, IMG_WIDTH, 1).
                - np.ndarray: The full mel spectrogram in dB scale.

        Raises:
            RuntimeError: If audio processing fails.
        """
        try:
            # Load audio file
            y, _ = librosa.load(audio_file, sr=sr, duration=duration)
            
            # Generate mel spectrogram with consistent parameters
            mel_spect = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=128,
                n_fft=2048,
                hop_length=512,
                fmax=8000
            )
            
            # Convert to log scale
            mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
            
            # Calculate frames per segment (4 seconds each)
            frames_per_segment = int((4 * sr) / 512)  # 4 seconds * sr / hop_length
            
            # Segment the spectrogram
            segments = []
            num_segments = mel_spect_db.shape[1] // frames_per_segment
            
            for i in range(min(num_segments, self.NUM_SEGMENTS)):
                start_frame = i * frames_per_segment
                end_frame = start_frame + frames_per_segment
                segment = mel_spect_db[:, start_frame:end_frame]
                # Resize and add channel dimension
                segment = tf.image.resize(segment[..., np.newaxis], 
                                       (self.IMG_HEIGHT, self.IMG_WIDTH)).numpy()
                segments.append(segment)
            
            # Pad with zeros if we don't have enough segments
            while len(segments) < self.NUM_SEGMENTS:
                zero_segment = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 1))
                segments.append(zero_segment)
            
            return np.array(segments), mel_spect_db
            
        except Exception as e:
            raise RuntimeError(f"Failed to process audio: {str(e)}")

    def predict_genre(self, audio_segments):
        """
        Predicts the music genre from processed audio segments.

        Args:
            audio_segments (np.ndarray): Array of mel spectrogram segments, shape (NUM_SEGMENTS, IMG_HEIGHT, IMG_WIDTH, 1).

        Returns:
            dict: A dictionary containing:
                - 'predicted_genre' (str): The predicted genre.
                - 'confidence_scores' (dict): Confidence scores for each genre.

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            # Prepare input for model (add batch dimension)
            model_input = np.expand_dims(audio_segments, axis=0)
            
            # Get prediction
            prediction = self.model.predict(model_input, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence_scores = prediction[0]
            
            return {
                'predicted_genre': self.GENRES[predicted_class],
                'confidence_scores': {
                    genre: float(score) 
                    for genre, score in zip(self.GENRES, confidence_scores)
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to predict genre: {str(e)}")

    def generate_spectrogram_plot(self, mel_spect_db):
        """
        Generates a spectrogram plot from the mel spectrogram data.

        Args:
            mel_spect_db (np.ndarray): Mel spectrogram in dB scale.

        Returns:
            matplotlib.figure.Figure: The generated matplotlib figure.

        Raises:
            RuntimeError: If spectrogram generation fails.
        """
        try:
            plt.clf()  # Clear any existing plots
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(
                mel_spect_db,
                y_axis='mel',
                x_axis='time',
                ax=ax
            )
            plt.colorbar(img, format='%+2.0f dB')
            return fig
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate spectrogram: {str(e)}")