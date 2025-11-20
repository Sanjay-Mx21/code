# utils/ticker_mapper.py
from pathlib import Path
import pandas as pd
import re
from difflib import get_close_matches
from typing import Dict, List, Set

def build_ticker_lookup(prices_dir: Path) -> Dict[str, Dict]:
    """
    Scans prices_dir for CSVs and builds a lookup dict:
      { "AAPL": {"ticker":"AAPL", "name": "Apple Inc", "aliases": ["apple","apple inc"]}, ... }
    Tries to read a 'name' or 'Name' column inside CSVs to get a company name; falls back to ticker only.
    """
    lookup = {}
    prices_dir = Path(prices_dir)
    for csv_path in sorted(prices_dir.glob("*.csv")):
        ticker = csv_path.stem.upper()
        name = None
        try:
            df = pd.read_csv(csv_path, nrows=5)
            cols = [c.lower().strip() for c in df.columns]
            # try to guess a name column
            if "name" in cols:
                name = df.iloc[0][[c for c in df.columns if c.lower().strip()=="name"][0]]
            elif "company" in cols:
                name = df.iloc[0][[c for c in df.columns if c.lower().strip()=="company"][0]]
        except Exception:
            name = None

        # normalize name to string
        if pd.isna(name) or name is None:
            name = None
        else:
            name = str(name).strip()

        aliases = set()
        aliases.add(ticker.lower())
        if name:
            # create simple aliases: full name, lower, individual tokens
            aliases.add(name.lower())
            for token in re.findall(r"[A-Za-z0-9]+", name.lower()):
                if len(token) > 2:
                    aliases.add(token)

        lookup[ticker] = {"ticker": ticker, "name": name, "aliases": sorted(list(aliases))}
    return lookup

def detect_tickers_in_text(text: str, lookup: Dict[str, Dict]) -> List[str]:
    """
    Detect possible tickers mentioned in `text` based on:
      1. $TICKER pattern (e.g. $AAPL)
      2. Exact ticker token (word boundary)
      3. Company name substring or alias match
      4. Fuzzy matching fallback (difflib) using alias tokens
    Returns a list of matched tickers (unique, in arbitrary order).
    """
    if not text or not lookup:
        return []

    text_lower = text.lower()
    found: Set[str] = set()

    # 1) $TICKER pattern
    for match in re.findall(r"\$([A-Za-z]{1,6})", text):
        t = match.upper()
        if t in lookup:
            found.add(t)

    # 2) exact ticker token (word boundary)
    for t, meta in lookup.items():
        pattern = r"\b" + re.escape(t.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.add(t)

    # 3) company name / alias substring match
    for t, meta in lookup.items():
        for alias in meta.get("aliases", []):
            if alias and alias in text_lower:
                found.add(t)
                break

        # if we already have it, skip other checks
        if t in found:
            continue

        # try name full substring (if available)
        name = meta.get("name")
        if name and name.lower() in text_lower:
            found.add(t)

    # 4) fuzzy fallback: check tokens against aliases using difflib
    if not found:
        # build alias->ticker reverse map
        alias_map = {}
        alias_list = []
        for t, meta in lookup.items():
            for a in meta.get("aliases", []):
                alias_map[a] = t
                alias_list.append(a)
        # check each significant token in text
        tokens = re.findall(r"[A-Za-z0-9]{3,}", text_lower)
        for tok in tokens:
            matches = get_close_matches(tok, alias_list, n=3, cutoff=0.85)
            for m in matches:
                found.add(alias_map.get(m))

    # final: return sorted list
    return sorted([t for t in found if t])

# small helper to expand rows if you want each ticker per row (optional)
def explode_news_by_ticker(df_news, lookup):
    """
    df_news expected to have 'title' column.
    Returns a new DataFrame with one row per (original_row, detected_ticker).
    """
    rows = []
    for _, row in df_news.reset_index(drop=True).iterrows():
        title = str(row.get("title", ""))
        detected = detect_tickers_in_text(title, lookup)
        if detected:
            for t in detected:
                new = row.to_dict()
                new["detected_ticker"] = t
                rows.append(new)
        else:
            new = row.to_dict()
            new["detected_ticker"] = None
            rows.append(new)
    return pd.DataFrame(rows)
