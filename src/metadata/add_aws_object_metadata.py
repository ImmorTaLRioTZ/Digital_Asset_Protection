import boto3
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("BUCKET_NAME")
s3_client = boto3.client('s3')

def upload_with_metadata(local_file_path, bucket, object_name, metadata):
    s3_client.upload_file(
        local_file_path, 
        bucket, 
        object_name,
        ExtraArgs=metadata
    )
    print(f"✅ Uploaded {object_name} with custom metadata.")



