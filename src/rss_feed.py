'''THIS IS AN ACTUAL RSS FEED PARSER. DO NOT USE IT BECAUSE 
THE CODEBASE IS CURRENTLY WORKING PURELY ON SIMULATION'''

import feedparser

def fetch_espn_news():
    # This is ESPN's top news RSS feed URL
    rss_url = "https://www.espn.com/espn/rss/news"
    
    # feedparser reads the URL and parses the XML
    feed = feedparser.parse(rss_url)
    
    # Create an empty list to store our clean data
    latest_news = []
    
    # Loop through the first 5 articles in the feed
    for entry in feed.entries[:10]:
        news_item = {
            "title": entry.title,
            "summary": entry.description,
            "link": entry.link,
            "published_time": entry.published
        }
        latest_news.append(news_item)
        
    return latest_news

# Test it
news_data = fetch_espn_news()
for item in news_data:
    print(f"Headline: {item['title']}\nSummary: {item['summary']}\n---")