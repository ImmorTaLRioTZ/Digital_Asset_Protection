import os
import logging
from typing import List, Dict
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure production-level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # You can add a FileHandler here to save logs to a file in production
        # logging.FileHandler("asset_monitor.log") 
    ]
)
logger = logging.getLogger("AssetMonitor")

class YouTubeAssetMonitor:
    """
    A robust client for monitoring YouTube for intellectual property infringements.
    """
    
    def __init__(self):
        """Initializes the YouTube API client securely using environment variables."""
        # Load environment variables from the .env file
        load_dotenv()
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        
        if not self.api_key:
            logger.critical("YOUTUBE_API_KEY not found in environment variables.")
            raise ValueError("Missing YOUTUBE_API_KEY. Please verify your .env file.")
            
        try:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            logger.info("YouTube API client initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to build YouTube client: {e}")
            raise

    def search_content(self, query: str, max_pages: int = 1, results_per_page: int = 50) -> List[Dict]:
        """
        Searches YouTube for specific keywords and handles pagination safely.
        
        Args:
            query (str): The search term (e.g., "Team A vs Team B Live Stream").
            max_pages (int): How many pages of results to fetch to control API quota.
            results_per_page (int): Max is 50 per YouTube API docs.
            
        Returns:
            List[Dict]: A structured list of videos matching the search.
        """
        logger.info(f"Starting search for query: '{query}' (Max Pages: {max_pages})")
        
        suspect_videos = []
        next_page_token = None
        pages_fetched = 0

        while pages_fetched < max_pages:
            try:
                request = self.youtube.search().list(
                    q=query,
                    part='id,snippet',
                    type='video',
                    maxResults=results_per_page,
                    pageToken=next_page_token
                )
                response = request.execute()

                items = response.get('items', [])
                for item in items:
                    video_id = item['id']['videoId']
                    suspect_videos.append({
                        'video_id': video_id,
                        'title': item['snippet']['title'],
                        'channel_id': item['snippet']['channelId'],
                        'channel_name': item['snippet']['channelTitle'],
                        'published_at': item['snippet']['publishedAt'],
                        'url': f"https://www.youtube.com/watch?v={video_id}"
                    })

                pages_fetched += 1
                next_page_token = response.get('nextPageToken')
                
                logger.info(f"Fetched page {pages_fetched}. Extracted {len(items)} items.")

                # If there are no more pages, break the loop early
                if not next_page_token:
                    logger.info("No more pages of results available.")
                    break

            except HttpError as e:
                # This specifically catches API quota limits, bad requests, or auth issues
                logger.error(f"An HTTP error occurred interacting with the YouTube API: {e.resp.status} - {e.content}")
                break
            except Exception as e:
                # Catch-all for unexpected pipeline failures (network drops, etc.)
                logger.error(f"An unexpected error occurred: {e}", exc_info=True)
                break

        logger.info(f"Search complete. Found {len(suspect_videos)} total suspect videos.")
        return suspect_videos

# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    try:
        # Initialize the monitor (will automatically read from .env)
        monitor = YouTubeAssetMonitor()
        
        # Define the exact broadcast or asset you are protecting
        target_query = "IPL Rajasthan royals vs Sunrisers Hyderabad"
        
        # Fetch up to 2 pages of results (up to 100 videos)
        results = monitor.search_content(query=target_query, max_pages=2)
        
        # Process the results (e.g., save to database, pass to an AI filter)
        for video in results:
            # Here you would typically add logic to skip your OWN official channels
            # if video['channel_id'] == 'YOUR_OFFICIAL_CHANNEL_ID': continue
            
            print(video["title"])
            
    except ValueError as ve:
        # Graceful exit if the environment isn't configured right
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"System failure: {e}")