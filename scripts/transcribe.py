import whisper

def transcribe_audio_file(input_file: str, language_code: str = "he") -> str:
    """
    Transcribes the given audio file using OpenAI Whisper, with support for Hebrew.
    
    :param input_file: Path to the audio file (e.g., "processed.wav").
    :param language_code: Language code to help Whisper transcribe accurately ("he" for Hebrew).
    :return: The transcribed text.
    """
    # Load a multilingual Whisper model. "medium" or "large" models
    # often perform better with non-English languages.
    model = whisper.load_model("medium")

    # Transcribe and specify the language to help the model.
    result = model.transcribe(input_file, language=language_code)

    return result["text"]

if __name__ == "__main__":
    # Example usage
    transcription = transcribe_audio_file("processed.wav", language_code="he")
    print("Transcribed Text (Hebrew):")
    print(transcription)
