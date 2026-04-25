"""
Async Reddit Crawler  —  Production Grade
=====================================================
Features:
  • Fully asynchronous I/O via AsyncPRAW
  • Exponential backoff for network resilience
  • Memory-efficient async generators for large data streams
  • Structured logging and type hinting
  • Clean teardown of HTTP sessions
"""

import asyncio
import logging
import os
from functools import wraps
from typing import AsyncGenerator, Dict, Any, List

import asyncpraw
from asyncprawcore.exceptions import AsyncPrawcoreException

# ── Logging Configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reddit_crawler")

# ── Retry Decorator ───────────────────────────────────────────────────────────
def async_retry(retries: int = 3, base_delay: float = 2.0):
    """
    Exponential backoff decorator for async functions.
    Catches AsyncPrawcoreException (network/API errors).
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return await func(*args, **kwargs)
                except AsyncPrawcoreException as e:
                    attempt += 1
                    delay = base_delay * (2 ** (attempt - 1))
                    log.warning(
                        "API Error on %s: %s. Retrying %d/%d in %.1fs...",
                        func.__name__, str(e), attempt, retries, delay
                    )
                    if attempt >= retries:
                        log.error("Max retries reached for %s. Failing.", func.__name__)
                        raise
                    await asyncio.sleep(delay)
        return wrapper
    return decorator


# ── Crawler Core ──────────────────────────────────────────────────────────────
class AsyncRedditCrawler:
    def __init__(self):
        """
        Initializes the crawler using environment variables.
        Required ENV vars:
            REDDIT_CLIENT_ID
            REDDIT_CLIENT_SECRET
            REDDIT_USER_AGENT (e.g., "script:my_crawler:v1.0 (by u/your_username)")
        """
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT")

        if not all([client_id, client_secret, user_agent]):
            raise ValueError(
                "Missing required credentials. Ensure REDDIT_CLIENT_ID, "
                "REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT are set."
            )

        self.reddit = asyncpraw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        self.reddit.read_only = True
        log.info("Initialized AsyncRedditCrawler (Read-Only Mode)")

    async def close(self):
        """Must be called to gracefully close the underlying aiohttp session."""
        await self.reddit.close()
        log.info("Closed Reddit API session.")

    @async_retry(retries=3, base_delay=2.0)
    async def fetch_subreddit_posts(
        self, 
        subreddit_name: str, 
        limit: int = 100, 
        sort: str = "hot"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Async generator that yields formatted dictionary representations of posts.
        """
        log.info("Fetching top %d '%s' posts from r/%s...", limit, sort, subreddit_name)
        subreddit = await self.reddit.subreddit(subreddit_name)
        
        # Determine sorting method
        if sort == "new":
            submissions = subreddit.new(limit=limit)
        elif sort == "top":
            submissions = subreddit.top(limit=limit)
        else:
            submissions = subreddit.hot(limit=limit)

        count = 0
        async for post in submissions:
            yield {
                "id": post.id,
                "title": post.title,
                "author": str(post.author) if post.author else "[deleted]",
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
                "num_comments": post.num_comments,
                "created_utc": post.created_utc,
                "url": post.url,
                "selftext": post.selftext,
                "is_video": post.is_video,
            }
            count += 1

        log.info("Finished yielding %d posts from r/%s.", count, subreddit_name)


# ── Execution Pipeline ────────────────────────────────────────────────────────
async def main():
    # 1. Ensure your environment variables are set before running
    # os.environ["REDDIT_CLIENT_ID"] = "your_id"
    # os.environ["REDDIT_CLIENT_SECRET"] = "your_secret"
    # os.environ["REDDIT_USER_AGENT"] = "script:data_ingestion:v1.0"

    crawler = None
    try:
        crawler = AsyncRedditCrawler()
        
        # 2. Define targets
        target_subreddits = ["computervision", "machinelearning"]
        post_limit = 50
        
        # 3. Create concurrent scraping tasks
        # We use asyncio.gather to scrape multiple subreddits at the exact same time
        async def process_subreddit(sub_name: str):
            results = []
            async for post in crawler.fetch_subreddit_posts(sub_name, limit=post_limit, sort="hot"):
                results.append(post)
                # In a real pipeline, you might push this dict directly to a DB,
                # Kafka topic, or JSON file here instead of appending to a list.
            return sub_name, len(results)

        log.info("Starting concurrent crawl for: %s", ", ".join(target_subreddits))
        
        tasks = [process_subreddit(sub) for sub in target_subreddits]
        completed_crawls = await asyncio.gather(*tasks)
        
        # 4. Report
        log.info("-" * 40)
        for sub, count in completed_crawls:
            log.info("Successfully scraped %d posts from r/%s", count, sub)
        log.info("-" * 40)

    except ValueError as ve:
        log.error("Configuration Error: %s", ve)
    except Exception as e:
        log.exception("Pipeline encountered a fatal error: %s", e)
    finally:
        if crawler:
            await crawler.close()


if __name__ == "__main__":
    # Windows-specific fix for asyncio Event Loop policies
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Crawler manually interrupted by user.")