# combined_analysis.py

from analyze_text_hebrew import analyze_hebrew_text
from analyze_text_english import analyze_english_text

def run_combined_analysis(hebrew_text, english_text):
    # Hebrew analysis
    he_result = analyze_hebrew_text(hebrew_text)
    
    # English analysis
    en_result = analyze_english_text(english_text)

    # Combine or store the results
    return {
        "hebrew_analysis": he_result,
        "english_analysis": en_result
    }

if __name__ == "__main__":
    # Example usage
    hebrew_sample = "היי ילד טיפש, אל תבכה בבקשה."
    english_sample = "Hey, don't be stupid. Stop crying."
    
    combined_result = run_combined_analysis(hebrew_sample, english_sample)
    print(combined_result)
