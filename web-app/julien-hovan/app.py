import streamlit as st
from model_handler import MusicGenrePredictor
import matplotlib
import matplotlib.pyplot as plt
import librosa
import io
import soundfile as sf  # Add this import for audio writing
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

# Create two columns for layout
col1, col2 = st.columns([3, 2])

with col1:
    # File upload section
    st.subheader("Upload Your Audio")
    uploaded_file = st.file_uploader(
        "Choose an MP3 or WAV file",
        type=["mp3", "wav"],
        help="Maximum file size: 200MB. Files longer than 30 seconds will be trimmed."
    )
    
    if uploaded_file is not None:
        st.success("File successfully uploaded!")
        
        # Initialize session state for processed audio
        if 'processed_audio' not in st.session_state:
            st.session_state['processed_audio'] = None
        
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
                plt.close(fig)
                st.pyplot(fig)
                
                # Store processed audio in session state
                st.session_state['processed_audio'] = audio_segments
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

with col2:
    # Results section
    st.subheader("Analysis Results")
    
    # Check if we have processed audio before attempting prediction
    if (uploaded_file is not None and 
        st.session_state.get('model_loaded', False) and 
        st.session_state.get('processed_audio') is not None):
        with st.spinner("Analyzing audio..."):
            try:
                # Get prediction
                results = predictor.predict_genre(st.session_state['processed_audio'])
                
                # Display results
                st.markdown("### Predicted Genre")
                st.info(f"{results['predicted_genre'].title()}", icon="🎵")
                
                st.markdown("### Confidence Scores")
                for genre, score in results['confidence_scores'].items():
                    st.progress(score, text=f"{genre.title()}: {score:.1%}")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

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