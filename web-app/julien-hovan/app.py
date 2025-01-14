import streamlit as st
from model_handler import MusicGenrePredictor
import matplotlib
import matplotlib.pyplot as plt
import librosa
import io
import soundfile as sf
import os

matplotlib.use('Agg')  # Required for streamlit

# Initialize the predictor (wrapped in cache_resource to prevent reloading)
@st.cache_resource
def get_predictor():
    """
    Initializes and returns the MusicGenrePredictor.
    
    This function is cached to prevent reloading the model on every rerun.
    """
    return MusicGenrePredictor()

# Page configuration
st.set_page_config(
    page_title="The Music Translator",
    page_icon="🎵",
    layout="wide"
)

# Main title and description
st.title("🎵 The Music Translator")
st.markdown("""
    Transform your music into visual patterns, display its spectrogram, and discover its genre! 
    Upload an audio file and let AI analyze its musical characteristics.
""")

# Initialize predictor
try:
    predictor = get_predictor()
    st.session_state['model_loaded'] = True
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.session_state['model_loaded'] = False

# Add this after the title and description
st.markdown("### Try these examples")
example_col1, example_col2 = st.columns([1, 2])

with example_col1:
    st.markdown("#### 🎵 Example Files")
    example_files = {
        "Classical Piano": "web-app/julien-hovan/examples/classical_example.wav",
        "Jazz Ensemble": "web-app/julien-hovan/examples/jazz_example.wav",
        "Rock Band": "web-app/julien-hovan/examples/rock_example.wav",
        "Hip Hop Beat": "web-app/julien-hovan/examples/hiphop_example.wav"
    }
    
    selected_example = st.selectbox(
        "Select an example track",
        [""] + list(example_files.keys()),
        index=0,
        help="Choose from our curated example files"
    )

with example_col2:
    if selected_example:
        example_path = example_files[selected_example]
        if os.path.exists(example_path):
            with open(example_path, 'rb') as f:
                st.audio(f, format='audio/wav')
                if st.button(f"Analyze {selected_example}", key=f"analyze_{selected_example}"):
                    # Read the example file into a BytesIO object
                    example_buffer = io.BytesIO()
                    with open(example_path, 'rb') as example_f:
                        example_buffer.write(example_f.read())
                    example_buffer.seek(0)
                    # Store in session state with a unique key
                    st.session_state['uploaded_file'] = example_buffer
                    # Indicate that a new file has been uploaded
                    st.session_state['new_file_uploaded'] = True

# Modify the file upload section to check session state and clear state if needed
if 'new_file_uploaded' in st.session_state and st.session_state['new_file_uploaded']:
    # Clear previous results when a new file is uploaded
    if 'processed_audio' in st.session_state:
        del st.session_state['processed_audio']
    if 'prediction_results' in st.session_state:
        del st.session_state['prediction_results']
    st.session_state['new_file_uploaded'] = False

if 'uploaded_file' in st.session_state:
    uploaded_file = st.session_state['uploaded_file']
else:
    uploaded_file = st.file_uploader(
        "Choose an MP3 or WAV file",
        type=["mp3", "wav"],
        help="Maximum file size: 200MB. Files longer than 30 seconds will be trimmed."
    )
    
    if uploaded_file is not None:
        st.success("File successfully uploaded!")
        # Indicate that a new file has been uploaded
        st.session_state['new_file_uploaded'] = True

# Initialize session state for processed audio and prediction results if not present
if 'processed_audio' not in st.session_state:
    st.session_state['processed_audio'] = None
if 'prediction_results' not in st.session_state:
    st.session_state['prediction_results'] = None

# Create columns for main content
col1, col2 = st.columns([1, 1])

# Spectrogram section
with col1:
    st.subheader("Audio Spectrogram")
    if uploaded_file is not None:
        with st.spinner("Processing audio..."):
            try:
                # Load and check audio duration
                audio_bytes = uploaded_file.read()
                uploaded_file.seek(0)  # Reset file pointer
                
                # Load audio with librosa
                audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
                duration = librosa.get_duration(y=audio, sr=sr)
                
                # Only show warning and trim if duration is significantly over 30 seconds
                if duration > 30.5:  # Adding small buffer for rounding
                    st.warning(f"Audio duration ({duration:.1f}s) exceeds 30 seconds. Only the first 30 seconds will be analyzed.")
                    # Trim audio to 30 seconds
                    samples_to_keep = int(30 * sr)
                    audio = audio[:samples_to_keep]
                    # Convert back to bytes
                    trimmed_file = io.BytesIO()
                    sf.write(trimmed_file, audio, sr, format='wav')
                    trimmed_file.seek(0)
                    uploaded_file = trimmed_file
                
                # Process the audio file
                audio_segments, mel_spect = predictor.process_audio_file(uploaded_file)
                
                # Display spectrogram
                fig = predictor.generate_spectrogram_plot(mel_spect)
                fig.set_size_inches(6, 4)
                plt.close(fig)
                st.pyplot(fig)
                
                # Store processed audio in session state
                st.session_state['processed_audio'] = audio_segments
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

# Results section
with col2:
    st.subheader("Analysis Results")
    
    # Check if we have processed audio and the model is loaded before attempting prediction
    if (st.session_state['processed_audio'] is not None and
            st.session_state.get('model_loaded', False)):
        # Only run prediction if results are not already stored or a new file has been uploaded
        if st.session_state['prediction_results'] is None:
            with st.spinner("Analyzing audio..."):
                try:
                    # Get prediction
                    results = predictor.predict_genre(st.session_state['processed_audio'])
                    # Store prediction results in session state
                    st.session_state['prediction_results'] = results
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

        # Display results if available
        if st.session_state['prediction_results'] is not None:
            results = st.session_state['prediction_results']
            st.markdown("### Predicted Genre")
            st.info(f"{results['predicted_genre'].title()}", icon="🎵")
            
            st.markdown("### Confidence Scores")
            for genre, score in results['confidence_scores'].items():
                st.progress(score, text=f"{genre.title()}: {score:.1%}")

# Additional information
with st.expander("About The Music Translator"):
    st.markdown("""
        This application uses deep learning to analyze audio files and predict their musical genre.
        The model has been trained on the GTZAN dataset, featuring 1000 audio tracks across 10 genres.
        
        **Supported Genres:**
        - Blues 🎸
        - Classical 🎻
        - Country 🤠
        - Disco 🕺
        - Hip Hop 🎤
        - Jazz 🎷
        - Metal 🤘
        - Pop 🎵
        - Reggae 🎼
        - Rock 🎸
        
        ---
        
        ### Model Architecture
        
        This project employs a sophisticated deep learning model that combines Convolutional Neural Networks (CNNs) 
        with multi-head attention mechanisms, achieving 85% accuracy on the test dataset.
        
        **The model processes audio in three main stages:**
        
        1. **Spatial Feature Extraction (CNN)**
           - Converts audio into spectrogram images
           - Processes 4-second segments through convolutional layers
           - Uses TimeDistributed layers to maintain sequential structure
        
        2. **Temporal Feature Extraction (Multi-Head Attention)**
           - Analyzes relationships between different time segments
           - Identifies key moments in the audio
           - Weighs the importance of different parts of the song
        
        3. **Genre Classification**
           - Processes the attention-weighted features
           - Makes final genre prediction through fully connected layers
        
        **Technical Details:**
        - Input: Spectrogram segments (4 seconds each)
        - Architecture: CNN + Multi-Head Attention
        - Performance: 85% accuracy on test set
        - Dataset: GTZAN (1000 tracks, 10 genres)
    """)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Julien Hovan")