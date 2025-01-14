import librosa
import soundfile as sf
import os
from pathlib import Path

def trim_audio(input_path, output_path, duration=30):
    """
    Trim audio file to specified duration in seconds.
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to save trimmed audio file
        duration (int): Duration in seconds to trim to (default: 30)
    """
    # Load the audio file
    audio, sr = librosa.load(input_path, sr=None)
    
    # Calculate samples for 30 seconds
    samples_to_keep = int(duration * sr)
    
    # Trim the audio
    trimmed_audio = audio[:samples_to_keep]
    
    # Save the trimmed audio
    sf.write(output_path, trimmed_audio, sr)

def process_directory(input_dir, output_dir):
    """
    Process all WAV files in a directory.
    
    Args:
        input_dir (str): Input directory containing WAV files
        output_dir (str): Output directory for trimmed files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each WAV file
    for file in os.listdir(input_dir):
        if file.endswith('.wav'):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, f"30sec_{file}")
            
            print(f"Processing: {file}")
            trim_audio(input_path, output_path)
            print(f"Saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual paths
    input_directory = "web-app/julien-hovan/examples"
    output_directory = "web-app/julien-hovan/examples_30sec"
    
    process_directory(input_directory, output_directory)
