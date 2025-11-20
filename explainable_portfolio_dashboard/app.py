# app.py
import streamlit as st
from utils.data_loader import load_price_panel
from utils.news_loader import parse_rss_feed
from utils.sentiment_analysis import analyze_headlines
from utils.ticker_mapper import build_ticker_lookup, detect_tickers_in_text, explode_news_by_ticker
from utils.price_fetcher import fetch_live_prices
from pathlib import Path
from datetime import timedelta, datetime
import pandas as pd

st.set_page_config(page_title="Explainable Portfolio Dashboard", layout="wide")
st.title("Explainable Portfolio Dashboard — Data Loader Test")

@st.cache_data
def get_price_data():
    return load_price_panel()

# ---------- Event-window helper ----------
def compute_event_windows(exploded_events_df, price_df, forward_days=3):
    """
    exploded_events_df: DataFrame with columns ['detected_ticker', 'event_date', ...]
    price_df: DataFrame with columns ['ticker', 'date', 'close'] where date is datetime.date
    Returns: exploded_events_df with added columns ar_1, ar_2, ar_3, car_1_3 (forward simple returns)
    """
    # ensure price_df date is date (not datetime)
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
    price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # compute daily simple returns per ticker
    price_df["prev_close"] = price_df.groupby("ticker")["close"].shift(1)
    price_df["daily_return"] = (price_df["close"] / price_df["prev_close"]) - 1
    price_df = price_df.drop(columns=["prev_close"])

    # prepare a fast lookup dict: (ticker) -> DataFrame for that ticker
    ticker_groups = {t: g.set_index("date") for t, g in price_df.groupby("ticker")}

    # add columns
    out = exploded_events_df.copy().reset_index(drop=True)
    for d in range(1, forward_days + 1):
        out[f"ar_{d}"] = None

    out["car_1_3"] = None

    # iterate events (ok for small N)
    for i, row in out.iterrows():
        t = row.get("detected_ticker") or row.get("detected_tickers")  # handles both names
        # if detected_tickers is list, pick first (exploded should have single ticker)
        if isinstance(t, (list, tuple)) and len(t) > 0:
            t = t[0]
        if not t or pd.isna(t):
            # no ticker detected
            for d in range(1, forward_days + 1):
                out.at[i, f"ar_{d}"] = None
            out.at[i, "car_1_3"] = None
            continue

        t = str(t).upper()
        if t not in ticker_groups:
            # no price data for ticker
            for d in range(1, forward_days + 1):
                out.at[i, f"ar_{d}"] = None
            out.at[i, "car_1_3"] = None
            continue

        price_ts = ticker_groups[t]  # indexed by date
        event_date = row["event_date"]
        # compute forward simple returns for days 1..forward_days
        ar_vals = []
        for d in range(1, forward_days + 1):
            target_date = event_date + timedelta(days=d)
            if target_date in price_ts.index:
                ret = price_ts.at[target_date, "daily_return"]
                # if multiple rows for same date (unlikely), this may be an array/Series
                if isinstance(ret, (pd.Series, pd.DataFrame)):
                    ret = float(ret.iloc[0])
                ar_vals.append(ret)
            else:
                ar_vals.append(None)
            out.at[i, f"ar_{d}"] = ar_vals[-1]

        # compute CAR from day1 to dayN if any values present (skip None)
        numeric_vals = [v for v in ar_vals if v is not None]
        if numeric_vals:
            out.at[i, "car_1_3"] = sum(numeric_vals)
        else:
            out.at[i, "car_1_3"] = None

    return out

def main():

    # ---------------- PRICE DATA SECTION ----------------
    st.subheader("📈 Price Data Preview")
    try:
        price_df = get_price_data()
        st.success("✅ Loaded price data successfully.")
        st.markdown(f"**Rows:** {price_df.shape[0]} — **Tickers:** {', '.join(sorted(price_df['ticker'].unique()))}")
        st.dataframe(price_df.head(50))
    except Exception as e:
        st.error(f"❌ Error while loading price data: {e}")
        import traceback
        tb = traceback.format_exc()
        st.text_area("Traceback", tb, height=300)

    # ---------------- NEWS SECTION ----------------
    st.subheader("📰 Live News Feed (Real-Time)")

    # You can change to any company or keyword:
    rss_url = "https://news.google.com/rss/search?q=Apple+stock&hl=en-US&gl=US&ceid=US:en"

    try:
        news_items = parse_rss_feed(rss_url, max_items=10)

        if not news_items:
            st.warning("No news fetched.")
        else:
            df_news = pd.DataFrame(news_items)
            st.dataframe(df_news[["published", "title", "source", "link"]])

            # ---------------- SENTIMENT (FinBERT) ----------------
            st.markdown("**Sentiment (FinBERT)** — running on headlines below")
            headlines = df_news["title"].fillna("").astype(str).tolist()
            if headlines:
                with st.spinner("Analyzing headlines with FinBERT..."):
                    sent = analyze_headlines(headlines)
                s_df = pd.DataFrame(sent)
                merged = pd.concat([df_news.reset_index(drop=True), s_df], axis=1)
                st.dataframe(merged[["published", "title", "label", "score", "numeric_sentiment", "source", "link"]])
            else:
                st.info("No headlines to analyze.")

            # ---------------- TICKER DETECTION (dynamic) ----------------
            try:
                project_root = Path(__file__).resolve().parent
                prices_dir = project_root / "data" / "prices"

                # build lookup from CSV files present in data/prices
                ticker_lookup = build_ticker_lookup(prices_dir)

                # detect tickers for every headline and attach as a list
                merged["detected_tickers"] = merged["title"].fillna("").astype(str).apply(
                    lambda t: detect_tickers_in_text(t, ticker_lookup)
                )
                merged["event_date"] = pd.to_datetime(merged["published"]).dt.date

                # show detected tickers in the UI (including event_date)
                st.subheader("🔎 Detected tickers for each headline")
                st.dataframe(merged[["published","event_date", "title", "detected_tickers", "label", "numeric_sentiment", "source", "link"]])

                # ---------- Event study: explode & compute forward returns ----------
                # explode headlines so each row has one detected ticker
                exploded = []
                for _, r in merged.reset_index(drop=True).iterrows():
                    tickers = r.get("detected_tickers") or []
                    if isinstance(tickers, (list, tuple)) and len(tickers) > 0:
                        for tk in tickers:
                            newr = r.copy()
                            newr["detected_ticker"] = tk
                            exploded.append(newr)
                    else:
                        # skip headlines with no detected ticker for event study
                        pass

                exploded_df = pd.DataFrame(exploded)

                if exploded_df.empty:
                    st.info("No exploded events (no detected tickers) — cannot compute event windows.")
                else:
                    # Determine tickers we need live prices for (unique detected_ticker values)
                    needed_tickers = sorted(set(exploded_df["detected_ticker"].dropna().astype(str).str.upper().tolist()))
                    if not needed_tickers:
                        st.info("No tickers detected to fetch live prices for.")
                        price_df_for_events = price_df.copy()
                        price_df_for_events["date"] = pd.to_datetime(price_df_for_events["date"]).dt.date
                    else:
                        # choose a date range that covers events and forward window
                        min_event = min(exploded_df["event_date"])
                        max_event = max(exploded_df["event_date"])
                        start = datetime.combine(min_event - timedelta(days=5), datetime.min.time())
                        end = datetime.combine(max_event + timedelta(days=5), datetime.max.time())

                        st.info(f"Fetching live prices for: {', '.join(needed_tickers)} from {start.date()} to {end.date()} (programmatic, no manual download)")
                        try:
                            live_prices = fetch_live_prices(needed_tickers, start=start, end=end)
                            # merge with your historical price_df if you want; prefer live_prices for events
                            price_df_for_events = live_prices.copy()
                        except Exception as e:
                            st.error(f"Failed to fetch live prices (yfinance): {e}")
                            # fallback to local CSVs
                            price_df_for_events = price_df.copy()
                            price_df_for_events["date"] = pd.to_datetime(price_df_for_events["date"]).dt.date

                    # compute event windows (ar_1..ar_3 and car_1_3)
                    event_windows_df = compute_event_windows(exploded_df, price_df_for_events, forward_days=3)

                    st.subheader("Event study (forward returns 1..3 days and CAR 1-3)")
                    st.dataframe(event_windows_df[[
                        "event_date", "detected_ticker", "title", "label", "numeric_sentiment",
                        "ar_1", "ar_2", "ar_3", "car_1_3", "source", "link"
                    ]])

            except Exception as ex:
                st.error(f"❌ Error during ticker detection / event study: {ex}")

    except Exception as e:
        st.error(f"❌ Error while fetching news: {e}")

if __name__ == "__main__":
    main()
