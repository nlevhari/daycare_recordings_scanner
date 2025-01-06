import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Example Hebrew keywords:
from consts import HEBREW_KEYWORDS


# 1) Load Hebrew sentiment model (example: heBERT)
HEBREW_MODEL_NAME = "avichr/heBERT_sentiment"
hebrew_tokenizer = AutoTokenizer.from_pretrained(HEBREW_MODEL_NAME)
hebrew_model = AutoModelForSequenceClassification.from_pretrained(HEBREW_MODEL_NAME)
hebrew_sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=hebrew_model,
    tokenizer=hebrew_tokenizer,
    return_all_scores=True
)

# 2) Load Hebrew toxic classification model (example: alexmickey/HebToxicClassifier)
HEBREW_TOXIC_MODEL_NAME = "alexmickey/HebToxicClassifier"
hebrew_toxic_tokenizer = AutoTokenizer.from_pretrained(HEBREW_TOXIC_MODEL_NAME)
hebrew_toxic_model = AutoModelForSequenceClassification.from_pretrained(HEBREW_TOXIC_MODEL_NAME)
hebrew_toxic_pipeline = pipeline(
    "text-classification",
    model=hebrew_toxic_model,
    tokenizer=hebrew_toxic_tokenizer,
    return_all_scores=True
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
    toxicity_results = hebrew_toxic_pipeline(text)
    # Each pipeline run might return something like:
    # [
    #   [
    #     {"label": "toxic", "score": 0.9},
    #     {"label": "non-toxic", "score": 0.1}
    #   ]
    # ]
    if toxicity_results and isinstance(toxicity_results, list) and len(toxicity_results[0]) > 0:
        toxicity_sorted = sorted(toxicity_results[0], key=lambda x: x["score"], reverse=True)
        toxic_label = toxicity_sorted[0]["label"]
        toxic_score = toxicity_sorted[0]["score"]
    else:
        toxic_label = "unknown"
        toxic_score = 0.0

    return {
        "text_hebrew": text,
        "found_keywords": found_keywords,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "toxicity_label": toxic_label,
        "toxicity_score": toxic_score,
    }

if __name__ == "__main__":
    # Quick test
    sample_text_he = "היי ילד טיפש, תסתום כבר!"
    analysis = analyze_hebrew_text(sample_text_he)
    print(analysis)
