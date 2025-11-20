# run_news_test.py
from utils.news_loader import parse_rss_feed

rss_url = "https://news.google.com/rss/search?q=Apple+stock&hl=en-US&gl=US&ceid=US:en"

items = parse_rss_feed(rss_url, max_items=20)

print(f"Fetched {len(items)} items. Showing first 5 titles:\n")

for i, it in enumerate(items[:5], 1):
    pub = it['published'].isoformat() if it['published'] else "no-time"
    print(f"{i}. [{pub}] {it['title']}")
    print(f"   link: {it['link']}\n")
