import os
import whisper

def transcribe_audio_file(
    input_file: str, 
    language_code: str = "he", 
    model_size: str = "medium", 
    output_path: str = "./",
    force_transcription: bool = False
) -> str:
    """
    Transcribes the given audio file using OpenAI Whisper, with support for Hebrew.
    Optionally saves the transcription to a text file.
    
    :param input_file: Path to the audio file (e.g., "processed.wav").
    :param language_code: Language code (default 'he' for Hebrew).
    :param model_size: Whisper model size (default 'medium').
    :param output_transcript: Where to save the transcribed text (default 'transcript.txt').
    :param force_transcription: If True, re-run Whisper even if the transcript file exists.
    :return: The path to the transcription file.
    """
    # If transcript already exists and we don't want to force re-transcription, skip
    output_transcript = os.path.join(output_path, 'transcript.txt')
    if not force_transcription and os.path.exists(output_transcript):
        print(f"Transcript file '{output_transcript}' already exists. Skipping transcription.")
        return output_transcript

    # Load a multilingual Whisper model.
    model = whisper.load_model(model_size)

    # Transcribe and specify the language to help the model.
    result = model.transcribe(input_file, language=language_code)
    transcribed_text = result["text"]

    # Save the transcribed text
    with open(output_transcript, "w", encoding="utf-8") as f:
        f.write(transcribed_text)

    print(f"Transcription saved to '{output_transcript}'.")
    return output_transcript

if __name__ == "__main__":
    # Example usage
    transcript_file = transcribe_audio_file(
        input_file="processed.wav",
        language_code="he",
        model_size="medium",
        output_transcript="my_transcript.txt",
        force_transcription=False
    )

    print(f"Transcript file path: {transcript_file}")

