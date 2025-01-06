# Daycare Audio Monitor

A Python-based pipeline to:
1. **Record and preprocess** daycare audio.
2. **Transcribe** speech (in Hebrew, or any other supported language).
3. **Analyze text** (Hebrew and/or English) for concerning keywords, sentiment, or toxicity.
4. **Analyze tone** (loudness/pitch).
5. **Identify problematic moments** and export results to a JSON file.

> **Disclaimer**: Please ensure you have the legal right and necessary consent to record and analyze daycare audio. Comply with all privacy regulations in your jurisdiction.

---

## Features

1. **Preprocessing**  
   - Converts audio to mono, reduces background noise, outputs a cleaned WAV file.

2. **Speech-to-Text**  
   - Uses [OpenAI Whisper](https://github.com/openai/whisper) to transcribe Hebrew (or any other language) audio.

3. **Text Analysis**  
   - **Hebrew**: Uses a Hebrew BERT model for sentiment and a Hebrew toxicity classifier.  
   - **English**: Uses a standard English sentiment pipeline and a toxic language classifier (e.g., `unitary/toxic-bert`).  
   - (Optional) Translates Hebrew to English for additional checks (via `googletrans`).

4. **Tone Analysis**  
   - Checks for loudness (amplitude) and pitch (fundamental frequency) to detect shouting or harsh intonation.

5. **Problematic Moments**  
   - Flags segments containing concerning keywords, toxic or abusive language, negative or hostile sentiment, or loud/high-pitched tone.

6. **Report Generation**  
   - Saves a JSON file of all analyzed segments and prints a summary of how many were flagged as problematic.

---

## Installation

1. **Clone the Repo**
   ```bash
   git clone https://github.com/your-username/my-daycare-monitor.git
   cd my-daycare-monitor
    ```
    
2. **Set Up Conda Environment**  

    ```
    conda env create -f environment.yml
    conda activate daycare-env
    ```
    Or install packages manually:
    ```
    conda create -n daycare-env python=3.9
    conda activate daycare-env
    # Then install dependencies (example):
    pip install -r requirements.txt
    ```
   
3. **FFmpeg**  

    - On macOS (via Homebrew): `brew install ffmpeg`  
    - On Linux (apt-based): `sudo apt-get install ffmpeg`  
    - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html), place `ffmpeg.exe` in your PATH.  

4. **Optional**: If you want translation:  
   
    ```
    pip install googletrans==4.0.0-rc1
    ```

---

## Usage

Run the main script from the CLI:
```
python main.py --input /path/to/recording.mp3 \
               --output /path/to/output.json \
               --language_code he \
               --model_size medium \
               --use_translation
```
---

### Arguments

| Argument           | Description                                                                                           | Example                                   |
|--------------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------|
| `--input, -i`      | Path to the input recording (WAV/MP3/etc.)                                                            | `recording.wav` or `recording.mp3`        |
| `--output, -o`     | Path to the JSON file where results will be saved                                                    | `results.json`                            |
| `--language_code`  | Language code for transcription (default: `he` for Hebrew).                                           | `en`, `he`, etc.                          |
| `--model_size`     | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`). Default: `medium`.                  | `small`                                   |
| `--use_translation`| If provided, translates Hebrew text to English for additional analysis (keyword & toxicity checks).   | *Flag only; no argument*                  |

---


- **`main.py`**: Orchestrates the entire pipeline (preprocess, transcribe, analyze text & tone).  
- **`preprocess_audio.py`**: Noise reduction, mono conversion, and WAV export.  
- **`transcribe.py`**: Speech-to-text using Whisper.  
- **`hebrew_analysis.py`**: Hebrew keyword, sentiment, and toxicity checks.  
- **`english_analysis.py`**: English keyword, sentiment, and toxicity checks.  
- **`analyze_tone.py`**: Audio tone analysis (loudness, pitch).

---

## Example Workflow

1. **Record Audio**  
   - Place the audio file in a known location (e.g., `recording.wav`).

2. **Preprocess & Transcribe**  
   - Run `main.py`, which will preprocess the audio, then transcribe with Whisper.

3. **Text & Tone Analysis**  
   - If the audio is Hebrew, it’s analyzed with a Hebrew NLP model (optionally translated to English).  
   - Checks for keywords, abusive language, negative sentiment, or loud/high-pitched speech.

4. **Results & JSON Output**  
   - A JSON file is generated containing all segments, any flagged issues, and a summary printed to the console.

---

## Example Output

**Console Example**  

```
Preprocessing audio... 

Transcribing audio with Whisper... 

Analyzing tone... 

Analyzing segments for text-based problems...

Total segments: 10 

Problematic segments: 2 

First problematic segment example: { "start": 12.3, "end": 18.7, "text": "היי ילד טיפש, תסתום כבר!", "text_analysis": { "hebrew_analysis": { "text_hebrew": "היי ילד טיפש, תסתום כבר!", "found_keywords": ["טיפש", "תסתום"], "sentiment_label": "negative", "sentiment_score": 0.85, "toxicity_label": "toxic", "toxicity_score": 0.92 }, "english_analysis": { "text_english": "Hey stupid boy, shut up already!", "found_keywords": ["stupid", "shut up"], "sentiment_label": "NEGATIVE", "sentiment_score": 0.99, "toxicity_label": "toxic", "toxicity_score": 0.88 } }, "tone_analysis": { "average_amplitude": 0.12, "average_pitch_hz": 280.0, "tone_flags": { "loud": true, "high_pitch": true } }, "problematic": true }

Saving results to /path/to/results.json... Done.
```

---

## Contributing

1. **Fork** the repo on GitHub.  
2. **Create** a new branch (`feat/new-feature`).  
3. **Commit** your changes and **push** to GitHub.  
4. **Open a Pull Request** to discuss and merge.

---

## License

This project is available under the [MIT License](https://opensource.org/licenses/MIT).  
Please consult your local laws regarding **recording** and **analyzing** audio of third parties.

---

## Contact

For questions, issues, or feature requests, please open an [issue on GitHub](https://github.com/your-username/my-daycare-monitor/issues) or reach out to:
- **Name**: Your Name  
- **Email**: your.email@example.com  

**Happy Monitoring!**  
Remember to use responsibly and ethically.
