#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import List, Dict

# --- Import your modules/functions ---
# Adjust these imports to match your actual file/module names:
from scripts.preprocess import maybe_preprocess_audio, preprocess_audio
from scripts.analyze_text_english import analyze_english_text
from scripts.analyze_text_hebrew import analyze_hebrew_text
from scripts.transcribe import transcribe_audio_file
from scripts.analyze_tone import analyze_audio_tone

# For translation to English, if you want to also analyze the text in English
# (You can also do direct Hebrew-only analysis if you prefer)
try:
    from googletrans import Translator
except ImportError:
    print("Please install googletrans if you need translation: pip install googletrans==4.0.0-rc1")
    Translator = None


def chunk_transcript_with_timestamps(transcript_data: Dict) -> List[Dict]:
    """
    Given Whisper's transcription output, return a list of segments with
    start/end timestamps and text for each segment.
    
    :param transcript_data: The full dictionary returned by Whisper,
                           which usually contains a "segments" list.
    :return: A list of dicts like [
              { "start": float, "end": float, "text": str },
              ...
            ]
    """
    segments = transcript_data.get("segments", [])
    results = []
    for seg in segments:
        results.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })
    return results


def analyze_segment_text(
    text: str, 
    translator=None
) -> Dict:
    """
    Analyze text in Hebrew. Optionally translate to English and analyze again.
    Returns a dictionary with Hebrew + English analysis results.
    
    :param text: The Hebrew text to analyze.
    :param translator: An optional Translator instance for translating to English.
    :return: A dict with results from Hebrew analysis and (optionally) English analysis.
    """
    # 1) Hebrew analysis
    hebrew_result = analyze_hebrew_text(text)

    # 2) Optional: If we want English-based analysis as well, we can translate:
    english_result = {}
    if translator:
        translation_obj = translator.translate(text, src="he", dest="en")
        english_text = translation_obj.text
        english_result = analyze_english_text(english_text)

    # Combine results
    return {
        "hebrew_analysis": hebrew_result,
        "english_analysis": english_result,
    }


def is_segment_problematic(seg_analysis: Dict, tone_analysis: Dict) -> bool:
    """
    Decide if a segment is "problematic" based on text or tone features.
    This is a simple example. You can refine your conditions/thresholds.
    
    :param seg_analysis: Dict from analyze_segment_text.
    :param tone_analysis: Dict from analyze_audio_tone (for the chunk).
    :return: True if flagged as problematic, else False.
    """
    # 1) Check Hebrew keywords
    if seg_analysis["hebrew_analysis"]["found_keywords"]:
        return True

    # 3) Check English (if used)
    if "english_analysis" in seg_analysis:
        # Found keywords?
        if seg_analysis["english_analysis"].get("found_keywords"):
            return True
        # Toxic?
        if (seg_analysis["english_analysis"].get("toxicity_label") == "toxic" and
            seg_analysis["english_analysis"].get("toxicity_score", 0) > 0.5):
            return True

    # 4) Check volume/pitch from tone analysis
    # Example thresholds for "loud" or "high pitch"
    if tone_analysis["tone_flags"]["loud"] or tone_analysis["tone_flags"]["high_pitch"]:
        # This might or might not be considered problematic. 
        # You can refine the logic as needed.
        return True
    
    # If none of the above conditions are triggered, consider it not problematic
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Process and analyze a daycare audio recording."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input recording (e.g. recording.wav or recording.mp3)."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to the output JSON file where results will be saved."
    )
    parser.add_argument(
        "--language_code",
        default="he",
        help="Language code for transcription. Default: he (Hebrew)."
    )
    parser.add_argument(
        "--model_size",
        default="medium",
        help="Whisper model size to load (tiny, base, small, medium, large). Default=medium."
    )
    parser.add_argument(
        "--use_translation",
        action="store_true",
        help="If specified, we also translate Hebrew to English and analyze the English text."
    )
    args = parser.parse_args()

    input_file = args.input
    output_path = args.output
    language_code = args.language_code
    model_size = args.model_size
    use_translation = args.use_translation

    input_file_name = input_file.split('/')[-1]
    stripped_input_file_name = input_file_name.split('/')[-1][:input_file_name.find('.')] if input_file_name.find('.') != -1 else input_file_name
    output_path = os.path.join(output_path, stripped_input_file_name)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving files to {output_path}")

    # 1) Preprocess audio (noise reduction, mono, etc.)
    print("Preprocessing audio...")
    processed_path = os.path.join(output_path, f"processed.wav")
    _ = maybe_preprocess_audio(input_file, processed_path)

    # 2) Transcribe the processed audio with Whisper
    print("Transcribing audio with Whisper...")
    transcript_data = transcribe_audio_file(
        processed_path, 
        language_code=language_code, 
        model_size=model_size,
        output_path=output_path
    )
    # transcript_data might be a dict with {"text": "...", "segments": [...]}
    # if you used Whisper's "transcribe()" that returns segments.

    # 3) Break down transcript into segments (with timestamps)
    segments = []
    if isinstance(transcript_data, dict) and "segments" in transcript_data:
        # Whisper "transcribe" in Python typically returns segments in result["segments"]
        segments = chunk_transcript_with_timestamps(transcript_data)
    else:
        # Fallback: if you only got text, treat everything as one segment with no timestamps
        segments = [{"start": 0.0, "end": 0.0, "text": transcript_data}]

    # 4) Analyze TONE for the entire audio (or for each chunk)
    #    Typically, you might do a single global tone analysis 
    #    or chunk your audio manually. Here we'll do a single global analysis.
    #    If you want per-segment tone, you need to chunk the audio by time and re-run.
    print("Analyzing tone (global)...")
    tone_result = analyze_audio_tone(processed_path)

    # 5) Prepare translator if needed
    translator = None
    if use_translation and Translator is not None:
        translator = Translator()

    # 6) Analyze each segment's text (Hebrew, plus English if translation is used)
    print("Analyzing segments for text-based problems...")
    analyzed_segments = []
    for seg in segments:
        seg_text = seg["text"]
        seg_analysis = analyze_segment_text(seg_text, translator=translator)

        # Decide if the segment is "problematic"
        # We currently use the same "tone_result" for all segments 
        # (since we didn't chunk the audio for tone). 
        # In a more advanced approach, you'd chunk the audio in parallel with text segments.
        problem_flag = is_segment_problematic(seg_analysis, tone_result)

        analyzed_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg_text,
            "text_analysis": seg_analysis,
            "tone_analysis": tone_result,
            "problematic": problem_flag
        })

    # 7) Gather only problematic segments
    problematic_segments = [seg for seg in analyzed_segments if seg["problematic"]]
    num_problems = len(problematic_segments)

    # 8) Print summary
    print(f"\nTotal segments: {len(analyzed_segments)}")
    print(f"Problematic segments: {num_problems}")
    if num_problems > 0:
        print("First problematic segment example:")
        first_problem = problematic_segments[0]
        print(json.dumps(first_problem, indent=2, ensure_ascii=False))

    # 9) Save entire segment list (including analysis) to JSON
    print(f"\nSaving results to {output_path}/results.json...")
    with open(os.path.join(output_path, 'results.json'), "w", encoding="utf-8") as f:
        json.dump(analyzed_segments, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
