# utils/sentiment_analysis.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict

# model choice (works well for finance)
FINBERT_MODEL = "yiyanghkust/finbert-tone"  # alternative: "ProsusAI/finbert"

# lazy singletons so model loads only once per process
_tokenizer = None
_model = None
_pipe = None

def get_finbert_pipeline():
    global _tokenizer, _model, _pipe
    if _pipe is None:
        _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        _pipe = pipeline("sentiment-analysis", model=_model, tokenizer=_tokenizer, truncation=True)
    return _pipe

def analyze_headlines(headlines: List[str], batch_size: int = 16) -> List[Dict]:
    """
    Input: list of headline strings
    Output: list of dicts with keys:
      - label (str) e.g. 'positive'/'negative'/'neutral'
      - score (float) model confidence
      - numeric_sentiment (float) mapped: positive -> +score, negative -> -score, neutral -> 0.0
    """
    if not headlines:
        return []

    pipe = get_finbert_pipeline()
    results = pipe(headlines, batch_size=batch_size)

    out = []
    for r in results:
        label = r.get("label", "").lower()
        score = float(r.get("score", 0.0))
        if "neg" in label:
            numeric = -score
        elif "pos" in label:
            numeric = score
        else:
            numeric = 0.0
        out.append({"label": label, "score": score, "numeric_sentiment": numeric})
    return out
