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

python main.py --input /path/to/recording.mp3 \
               --output /path/to/output.json \
               --language_code he \
               --model_size medium \
               --use_translation

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

## Project Structure

