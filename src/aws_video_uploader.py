from pathlib import Path
import os
from dotenv import load_dotenv
from metadata.add_aws_object_metadata import upload_with_metadata
load_dotenv()

BUCKET = os.getenv("BUCKET_NAME")

video_dir = Path("Digital_Asset_Protection/videos/")

video_name = "batch1_suspect.mp4"  # change to your desired file name
video_object = video_dir / video_name

# print(Path.exists(video_object))
# print(BUCKET)

metadata = {
            'Metadata': {
                'uploader-email': 'admin@example.com',
                'uploader-name': 'Jit Sarkar',
                'asset-id': 'asset_001'
            },
            'ContentType': 'video/mp4',   # please do include the extension of the file
            # 'ACL': 'public-read'        # optional, may cause errors
        }

upload_with_metadata(video_object, BUCKET, "batch2_official.mp4", metadata)