# utils/news_loader.py
import feedparser
from datetime import datetime, timezone
import time

def parse_rss_feed(url: str, max_items:int=50):
    """
    Returns a list of dicts with keys:
      - published (datetime or None)
      - title (str)
      - summary (str)
      - link (str)
      - source (str)
    """
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:max_items]:
        published = None
        if hasattr(e, "published_parsed") and e.published_parsed:
            published = datetime.fromtimestamp(time.mktime(e.published_parsed), tz=timezone.utc)
        elif hasattr(e, "updated_parsed") and e.updated_parsed:
            published = datetime.fromtimestamp(time.mktime(e.updated_parsed), tz=timezone.utc)

        items.append({
            "published": published,
            "title": getattr(e, "title", "") or "",
            "summary": getattr(e, "summary", "") or "",
            "link": getattr(e, "link", "") or "",
            "source": feed.feed.get("title", "")
        })
    return items
