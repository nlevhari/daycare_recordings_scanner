import librosa
import numpy as np

def analyze_audio_tone(audio_file: str) -> dict:
    """
    Analyzes an audio file to detect 'tone' using acoustic features 
    like average amplitude (loudness) and pitch (fundamental frequency).
    
    :param audio_file: Path to the audio file (e.g., 'processed.wav').
    :return: A dictionary with amplitude and pitch stats, along with simple flags.
    """
    # 1. Load audio
    y, sr = librosa.load(audio_file, sr=None)  # sr=None -> use file's native sample rate

    # 2. Compute average amplitude (loudness)
    amplitude = np.abs(y)
    avg_amplitude = np.mean(amplitude)

    # 3. Detect pitch (fundamental frequency) using librosa's pyin
    #    pyin requires specifying pitch range; adjust as needed
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        sr=sr, 
        fmin=librosa.note_to_hz('C2'),  # ~65.4 Hz
        fmax=librosa.note_to_hz('C6')   # ~1046.5 Hz (adjust if needed)
    )
    
    # 4. Calculate average pitch across voiced frames
    if voiced_flag is not None and any(voiced_flag):
        avg_pitch = np.nanmean(f0[voiced_flag])
    else:
        avg_pitch = 0.0

    # 5. Simple thresholds for "loud" or "high pitch" detection (adjust as needed)
    is_loud = avg_amplitude > 0.1          # Example amplitude threshold
    is_high_pitch = avg_pitch > 220.0      # Example pitch threshold (Hz)

    # 6. Return results
    return {
        "average_amplitude": float(avg_amplitude),
        "average_pitch_hz": float(avg_pitch),
        "tone_flags": {
            "loud": is_loud,
            "high_pitch": is_high_pitch
        }
    }

if __name__ == "__main__":
    # Example usage
    file_path = "processed.wav"
    result = analyze_audio_tone(file_path)
    print(result)
