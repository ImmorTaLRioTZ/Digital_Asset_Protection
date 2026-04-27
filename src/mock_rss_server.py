from fastapi import FastAPI, Response, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- IMPORT THIS
from datetime import datetime
import boto3
import os
from dotenv import load_dotenv
import uuid

load_dotenv()
app = FastAPI()

# --- Configuration ---
BUCKET_NAME = os.getenv("BUCKET_NAME")
REGION = os.getenv("REGION")
ALLOWED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (React, etc.). Restrict this in production!
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

# Initialize the boto3 client using your configured AWS credentials
s3_client = boto3.client('s3', region_name=REGION)

def get_videos_from_s3():
    """
    Dynamically fetches video keys from S3, retrieves their custom metadata, 
    and constructs their public URLs.
    
    NOTE: Using head_object inside this loop creates an N+1 API bottleneck. 
    This is perfect for simulation and small buckets, but will slow down 
    if the bucket contains thousands of videos.
    """
    video_list = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME)
        
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                
                # Filter for video files only
                if key.lower().endswith(ALLOWED_EXTENSIONS):
                    
                    # Fetch metadata (extra API call per video)
                    head = s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
                    metadata = head.get("Metadata", {})
                    
                    uploader = metadata.get("uploader-name", "Unknown")
                    source = metadata.get("upload-source", "Unknown")
                    
                    public_url = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{key}"
                    pub_date = obj['LastModified'].strftime("%a, %d %b %Y %H:%M:%S GMT")
                    
                    video_list.append({
                        "id": key,
                        "title": f"Suspect: {key} (Uploaded by {uploader})",
                        "source": source,
                        "s3_url": public_url,
                        "pub_date": pub_date
                    })
                    
    except Exception as e:
        print(f"Error fetching from S3: {e}")
        
    return video_list

@app.get("/feed/suspects.xml")
def get_rss_feed():
    """Generates a dynamic XML RSS feed directly from the S3 bucket contents."""
    
    # 1. Dynamically fetch the latest list of videos and metadata
    dynamic_videos = get_videos_from_s3()
    
    # 2. Build the XML items
    items_xml = ""
    for video in dynamic_videos:
        items_xml += f"""
        <item>
            <title>{video['title']}</title>
            <link>{video['s3_url']}</link>
            <guid isPermaLink="false">{video['id']}</guid>
            <pubDate>{video['pub_date']}</pubDate>
            <source>{video['source']}</source>
        </item>
        """

    # 3. Wrap items in the RSS channel structure
    rss_xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
    <rss version="2.0">
        <channel>
            <title>Dynamic S3 Suspect Tracker</title>
            <link>http://localhost:8000</link>
            <description>Live feed of suspect videos and metadata directly from AWS.</description>
            {items_xml}
        </channel>
    </rss>
    """
    return Response(content=rss_xml, media_type="application/xml")

@app.post("/upload-official")
def upload_official_video(
    file: UploadFile = File(...),
    asset_id: str = Form(...),          # Now we ask for a specific ID (e.g., "movie_001")
    uploader_name: str = Form(...),
    uploader_email: str = Form(...)
):
    """Receives an OFFICIAL video from the frontend and uploads it to S3."""
    
    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    # Prefix with 'official_' to distinguish them in your S3 bucket
    safe_filename = f"official_{asset_id}_{uuid.uuid4().hex[:6]}_{file.filename}"

    try:
        print(f"⬆️ Uploading Official Asset '{asset_id}' to S3...")
        s3_client.upload_fileobj(
            file.file,
            BUCKET_NAME,
            safe_filename,
            ExtraArgs={
                'ContentType': file.content_type or 'video/mp4',
                'Metadata': {
                    'asset-id': asset_id,
                    'uploader-name': uploader_name,
                    'uploader-email': uploader_email,
                    'asset-type': 'official-reference' # Tag it clearly
                }
            }
        )
        
        public_url = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{safe_filename}"
        return {
            "message": "Official Asset Registered", 
            "asset_id": asset_id,
            "filename": safe_filename, 
            "url": public_url
        }

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload to S3")

@app.post("/upload-suspect")
def upload_suspect_video(
    file: UploadFile = File(...),
    uploader_name: str = Form(...),
    uploader_email: str = Form(...),
    upload_source: str = Form("web-frontend")
):
    """Receives a video from the frontend and uploads it to S3 with metadata."""
    
    # 1. Validate the file type
    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be a video.")

    # 2. Generate a unique safe filename
    safe_filename = f"suspect_{uuid.uuid4().hex[:8]}_{file.filename}"

    try:
        # 3. Stream the upload directly to S3
        print(f"⬆️ Uploading {safe_filename} to S3...")
        s3_client.upload_fileobj(
            file.file,
            BUCKET_NAME,
            safe_filename,
            ExtraArgs={
                'ContentType': file.content_type or 'video/mp4',
                'Metadata': {
                    'uploader-name': uploader_name,
                    'uploader-email': uploader_email,
                    'upload-source': upload_source
                }
            }
        )
        
        # 4. Return the new public URL
        public_url = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{safe_filename}"
        return {
            "message": "Upload successful", 
            "filename": safe_filename, 
            "url": public_url
        }

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload to S3")

@app.get("/api/videos")
def get_all_videos():
    """Returns a JSON list of all videos and their metadata with debug logging."""
    print("📡 Frontend requested /api/videos. Fetching from AWS...")
    video_list = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME)
        
        for page_num, page in enumerate(pages, 1):
            contents = page.get("Contents", [])
            print(f"📄 Scanning S3 Page {page_num}: Found {len(contents)} total objects.")
            
            for obj in contents:
                key = obj["Key"]
                
                if key.lower().endswith(ALLOWED_EXTENSIONS):
                    print(f"   🔍 Fetching metadata for: {key}")
                    
                    head = s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
                    metadata = head.get("Metadata", {})
                    
                    video_list.append({
                        "id": key,
                        "filename": key,
                        "uploader_name": metadata.get("uploader-name", "Unknown"),
                        "asset_type": metadata.get("asset-type", "suspect"),
                        "is_dmca_struck": metadata.get("dmca-struck", "false").lower() == "true",
                        "url": f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{key}",
                        "date": obj['LastModified'].strftime("%b %d, %Y - %H:%M")
                    })
                    
        print(f"✅ Successfully prepared {len(video_list)} videos for the frontend.")
        return {"videos": video_list}
        
    except Exception as e:
        print(f"❌ FATAL ERROR in get_all_videos: {str(e)}")
        # Print the full error traceback to the terminal
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Runs the simulated feed on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)