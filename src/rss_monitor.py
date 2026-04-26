import time
import os
import urllib.request
import feedparser
from pathlib import Path

# Import your pipeline classes
from pipeline.pipeline import DetectionPipeline, PipelineConfig

# --- Configuration ---
RSS_URL = "http://127.0.0.1:8000/feed/suspects.xml"
DOWNLOAD_DIR = "videos/incoming"
POLL_INTERVAL_SECONDS = 10

# Keep track of processed IDs in memory so we don't scan the same video twice
PROCESSED_GUIDS = set()

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def setup_pipeline() -> DetectionPipeline:
    """Initializes the pipeline and registers the official reference videos."""
    print("⚙️ Initializing Detection Pipeline...")
    
    # Point to where your SSCD model is stored
    config = PipelineConfig(sscd_model_dir=Path("models/"))
    pipeline = DetectionPipeline(config)
    
    # --- REGISTER YOUR OFFICIAL ASSETS HERE ---
    # The pipeline needs to know what it is protecting before it can check suspects.
    official_video_path = Path("videos/official_BW_short.mp4")
    
    if official_video_path.exists():
        print(f"📥 Registering official asset: {official_video_path.name}")
        # Assuming you want to skip audio for now based on previous testing
        pipeline.register_asset(
            official_video_path, 
            asset_id="official_BW_short", 
            skip_audio=True 
        )
    else:
        print(f"⚠️ WARNING: Official reference video not found at {official_video_path}")
        print("Pipeline has no references to check against!")
        
    return pipeline

def poll_feed(pipeline: DetectionPipeline):
    """Fetches the RSS feed, downloads new videos, and runs them through the pipeline."""
    try:
        feed = feedparser.parse(RSS_URL)
    except Exception as e:
        print(f"⚠️ Failed to connect to RSS feed: {e}")
        return

    # Check if the feed is valid (feedparser sets 'bozo' to 1 on malformed XML/connection errors)
    if feed.bozo and not feed.entries:
        print("📡 Waiting for RSS server... (Is mock_rss_server.py running?)")
        return

    for entry in feed.entries:
        video_id = entry.guid
        s3_link = entry.link
        
        # Skip if we have already checked this exact video
        if video_id in PROCESSED_GUIDS:
            continue
            
        print("\n" + "="*60)
        print(f"🚨 NEW SUSPECT DETECTED: {entry.title}")
        
        # We can also pull the custom source metadata we injected into the XML
        source = getattr(entry, 'source', 'Unknown')
        print(f"   Source: {source}")
        print(f"   Downloading: {s3_link}")
        
        filename = s3_link.split("/")[-1]
        local_path = Path(DOWNLOAD_DIR) / filename
        
        try:
            # 1. Download the video
            urllib.request.urlretrieve(s3_link, local_path)
            
            # 2. Feed it into the three-stage cascade
            print(f"   Running pipeline check...")
            result = pipeline.check(local_path)
            
            # 3. Output the verdict
            if result.is_pirated:
                print(f"   ⛔ PIRACY ALERT: Matched '{result.asset_id}'")
                print(f"      Caught via: {result.detected_by}")
                print(f"      Confidence: {result.confidence}")
            else:
                print(f"   ✅ CLEAN: Video passed all active stages.")
                
        except Exception as e:
            print(f"   ❌ Pipeline execution failed for {filename}: {e}")
            
        finally:
            # 4. Clean up the disk (Delete the suspect video to save space)
            if local_path.exists():
                local_path.unlink()
                print(f"   🧹 Cleaned up temporary file: {filename}")
            
            # Mark as processed regardless of outcome so it doesn't loop infinitely
            PROCESSED_GUIDS.add(video_id)

if __name__ == "__main__":
    detector = setup_pipeline()
    
    print(f"\n🚀 Starting continuous monitor. Polling every {POLL_INTERVAL_SECONDS}s.")
    print("Press Ctrl+C to stop.")
    print("-" * 60)
    
    try:
        while True:
            poll_feed(detector)
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped gracefully.")