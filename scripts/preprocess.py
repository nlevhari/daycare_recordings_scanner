from pydub import AudioSegment
import noisereduce as nr
import numpy as np

def preprocess_audio(input_file: str, output_file: str = "processed.wav") -> str:
    """
    Preprocess an audio file by converting to mono, 
    reducing noise, and saving as a WAV file.
    
    :param input_file: Path to the input audio file (e.g., 'recording.mp3' or 'recording.wav').
    :param output_file: Path where the processed file will be saved.
    :return: The path to the processed audio file.
    """
    # 1. Load the audio with pydub
    audio = AudioSegment.from_file(input_file)

    # 2. Convert to mono channel
    audio_mono = audio.set_channels(1)

    # 3. Convert AudioSegment to a NumPy array for noise reduction
    samples = np.array(audio_mono.get_array_of_samples(), dtype=np.float32)

    # 4. Apply spectral gating noise reduction
    reduced_noise = nr.reduce_noise(y=samples, sr=audio_mono.frame_rate)

    # 5. Convert the processed samples back to a pydub AudioSegment
    processed_audio = AudioSegment(
        reduced_noise.tobytes(),
        frame_rate=audio_mono.frame_rate,
        sample_width=audio_mono.sample_width,
        channels=1
    )

    # 6. Export the processed file as WAV
    processed_audio.export(output_file, format="wav")

    return output_file
