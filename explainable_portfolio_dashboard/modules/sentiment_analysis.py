import streamlit as st

@st.cache_resource
def _load_transformers(model_name="yiyanghkust/finbert-tone"):
    """Try to load a transformers pipeline. If transformers isn't installed, raise ImportError."""
    try:
        from transformers import pipeline
        # This will download if not present (optional). We do it lazily.
        sentiment = pipeline("text-classification", model=model_name, top_k=None)
        return sentiment
    except Exception as e:
        # Re-raise so callers can fallback; logging via streamlit
        st.warning("Transformers/FinBERT not available or failed to load: %s" % e)
        raise

def get_sentiment(headlines):
    """
    headlines : list[str] or pd.Series
    Returns list of sentiment scores in [-1, 1] (negative -> positive).
    Behavior:
      - If transformers available: map classes to numerical score (positive=+1, neutral=0, negative=-1) weighted by score.
      - If not: simple heuristic fallback.
    """
    texts = list(headlines)
    try:
        pipe = _load_transformers()
        out = pipe(texts, truncation=True)
        # pipe may return list of dicts or list of list-of-dicts depending on version
        def _score(item):
            # item might be [{'label':'positive','score':0.9},...]
            if isinstance(item, list):
                item = item[0]
            label = item.get("label","").lower()
            score = float(item.get("score", 0.0))
            if "pos" in label:
                return score
            if "neg" in label:
                return -score
            return 0.0
        return [_score(o) for o in out]
    except Exception:
        # fallback heuristic: simple keywords
        results = []
        for t in texts:
            txt = str(t).lower()
            s = 0.0
            if any(w in txt for w in ["gain","up","positive","beat","raises","higher","expands","outperform","upbeat"]):
                s += 0.6
            if any(w in txt for w in ["miss","cut","concern","down","lower","pulls back","warning","regulatory","misses"]):
                s -= 0.6
            if any(w in txt for w in ["profit taking","profit-taking","pull back","pulls back"]):
                s -= 0.2
            # bound
            if s > 1: s = 1.0
            if s < -1: s = -1.0
            results.append(s)
        return results
