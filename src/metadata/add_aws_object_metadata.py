import boto3
from pathlib import Path

s3_client = boto3.client('s3')

def upload_with_metadata(local_file_path, bucket, object_name):
    s3_client.upload_file(
        local_file_path, 
        bucket, 
        object_name,
        ExtraArgs={
            'Metadata': {
                'uploader-email': 'admin@example.com',
                'uploader-name': 'Jit Sarkar',
                'asset-id': 'asset_001'
            },
            'ContentType': 'video/mp4',   # important
            # 'ACL': 'public-read'        # optional
        }
    )
    print(f"✅ Uploaded {object_name} with custom metadata.")

video_dir = Path("/home/immortalriotz/PYTHON/Digital_Asset_Protection/Digital_Asset_Protection/videos/")

video_name = "batch1_suspect.mp4"  # change to your desired file name
video_object = video_dir / video_name

print(Path.exists(video_object))

BUCKET = "digital-asset-protection"

#upload_with_metadata(video_object, BUCKET, "batch1_suspect.mp4")

