import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Example list of English keywords/phrases to flag
from consts import ENGLISH_KEYWORDS

# 1) Load an English sentiment pipeline
english_sentiment_pipeline = pipeline("sentiment-analysis")

# 2) Load an English toxic/abusive classification model
TOXIC_MODEL_NAME = "unitary/toxic-bert"
toxic_tokenizer = AutoTokenizer.from_pretrained(TOXIC_MODEL_NAME)
toxic_model = AutoModelForSequenceClassification.from_pretrained(TOXIC_MODEL_NAME)
english_toxic_pipeline = pipeline(
    "text-classification",
    model=toxic_model,
    tokenizer=toxic_tokenizer,
    return_all_scores=True
)

def analyze_english_text(text: str) -> dict:
    """
    Analyzes English text for:
      1) English keyword detection
      2) English sentiment analysis
      3) Toxic/abusive language classification
    
    :param text: The English text to analyze
    :return: A dictionary with found keywords, sentiment, and toxicity results
    """
    # --- 1) Keyword Detection ---
    found_keywords = []
    for kw in ENGLISH_KEYWORDS:
        pattern = rf"\b{re.escape(kw.lower())}\b"
        if re.search(pattern, text.lower()):
            found_keywords.append(kw)

    # --- 2) Sentiment Analysis ---
    sentiment_results = english_sentiment_pipeline(text)
    if sentiment_results:
        sentiment_label = sentiment_results[0]["label"]
        sentiment_score = sentiment_results[0]["score"]
    else:
        sentiment_label = "unknown"
        sentiment_score = 0.0

    # --- 3) Toxic/Abusive Classification ---
    toxicity_raw = english_toxic_pipeline(text)
    # This pipeline returns a list of dicts, e.g.:
    # [
    #   [
    #     {"label": "toxic", "score": 0.7}, 
    #     {"label": "not toxic", "score": 0.3}
    #   ]
    # ]
    # Some models have different or more granular labels. 
    # We'll pick the label with the highest score:
    if toxicity_raw and isinstance(toxicity_raw, list) and len(toxicity_raw[0]) > 0:
        # Sort by descending score
        toxicity_sorted = sorted(toxicity_raw[0], key=lambda x: x["score"], reverse=True)
        top_toxic_label = toxicity_sorted[0]["label"]
        top_toxic_score = toxicity_sorted[0]["score"]
    else:
        top_toxic_label = "unknown"
        top_toxic_score = 0.0

    return {
        "text_english": text,
        "found_keywords": found_keywords,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "toxicity_label": top_toxic_label,
        "toxicity_score": top_toxic_score,
    }

if __name__ == "__main__":
    # Quick test
    sample_text_en = "Hey kid, you are so stupid. Shut up!"
    analysis = analyze_english_text(sample_text_en)
    print(analysis)
