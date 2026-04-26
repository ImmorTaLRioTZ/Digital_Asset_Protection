import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# --- Configuration ---
BUCKET_NAME = "digital-asset-protection"  # Just the name, not the full URL
DOWNLOAD_DIR = "aws_videos"
ALLOWED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def fetch_all_videos_dynamically():
    # Configure boto3 to make anonymous/unsigned requests
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    print(f"Connecting to bucket '{BUCKET_NAME}'...")
    
    try:
        # We use a paginator in case you have more than 1000 files in the bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME)
        
        video_keys = []
        
        # Scan through the bucket and find the videos
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                # Filter to only grab video files
                if key.lower().endswith(ALLOWED_EXTENSIONS):
                    video_keys.append(key)
        
        if not video_keys:
            print("No videos found in the bucket.")
            return

        print(f"Found {len(video_keys)} videos. Starting download...")
        
        # Download each video
        for key in video_keys:
            download_path = os.path.join(DOWNLOAD_DIR, key)
            
            if os.path.exists(download_path):
                print(f"⏭️ Skipping {key} (already downloaded)")
                continue
                
            print(f"⬇️ Downloading {key}...")
            # Download the file directly from S3
            s3_client.download_file(BUCKET_NAME, key, download_path)
            print(f"✅ Saved to {download_path}")
            
    except s3_client.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDenied':
            print("\n❌ ACCESS DENIED: Your bucket allows public file access, but blocks public directory listing.")
        else:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    fetch_all_videos_dynamically()