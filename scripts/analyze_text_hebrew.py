import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Example Hebrew keywords:
from common.consts import HEBREW_KEYWORDS


# 1) Load Hebrew sentiment model (example: heBERT)
HEBREW_MODEL_NAME = "avichr/heBERT_sentiment_analysis"
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("DGurgurov/xlm-r_hebrew_sentiment")
model = AutoModelForSequenceClassification.from_pretrained("DGurgurov/xlm-r_hebrew_sentiment")

# how to use?
hebrew_sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="DGurgurov/xlm-r_hebrew_sentiment",
    tokenizer="DGurgurov/xlm-r_hebrew_sentiment",
    return_all_scores = True
)


def analyze_hebrew_text(text: str) -> dict:
    """
    Analyzes Hebrew text for:
      1) Hebrew keyword detection
      2) Hebrew sentiment analysis
      3) Hebrew toxic/abusive language classification
    
    :param text: The Hebrew text to analyze
    :return: A dictionary with found keywords, sentiment, and toxicity results
    """
    # --- 1) Keyword Detection ---
    found_keywords = []
    for kw in HEBREW_KEYWORDS:
        pattern = rf"\b{re.escape(kw)}\b"
        if re.search(pattern, text):
            found_keywords.append(kw)

    # --- 2) Sentiment Analysis ---
    sentiment_results = hebrew_sentiment_pipeline(text)
    if sentiment_results:
        scores = sentiment_results[0]
        # Sort by descending score
        scores_sorted = sorted(scores, key=lambda x: x["score"], reverse=True)
        sentiment_label = scores_sorted[0]["label"]
        sentiment_score = scores_sorted[0]["score"]
    else:
        sentiment_label = "unknown"
        sentiment_score = 0.0

    # --- 3) Toxic/Abusive Classification ---
    # Each pipeline run might return something like:
    # [
    #   [
    #     {"label": "toxic", "score": 0.9},
    #     {"label": "non-toxic", "score": 0.1}
    #   ]
    # ]

    return {
        "text_hebrew": text,
        "found_keywords": found_keywords,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
    }

if __name__ == "__main__":
    # Quick test
    sample_text_he = "היי ילד טיפש, תסתום כבר!"
    analysis = analyze_hebrew_text(sample_text_he)
    print(analysis)
