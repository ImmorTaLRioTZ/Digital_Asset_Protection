import boto3

s3_client = boto3.client('s3')

def upload_with_metadata(local_file_path, bucket, object_name):
    s3_client.upload_file(
        local_file_path, 
        bucket, 
        object_name,
        ExtraArgs={
            'Metadata': {
                'uploader-email': 'user1@example.com',
                'uploader-name': 'Jane Doe',
                'asset-id': 'asset_002'
            },
            'ContentType': 'video/mp4',   # important
            # 'ACL': 'public-read'        # optional
        }
    )
    print(f"✅ Uploaded {object_name} with custom metadata.")