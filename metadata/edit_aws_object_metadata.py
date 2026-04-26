import boto3

s3 = boto3.client('s3')

BUCKET = "digital_asset_protection"
KEY = "suspect1.mp4"

def update_metadata():
    # Step 1: Get existing metadata
    head = s3.head_object(Bucket=BUCKET, Key=KEY)
    existing_metadata = head.get('Metadata', {})

    # Step 2: Merge with new metadata
    new_metadata = {
        **existing_metadata,
        'uploader-email': 'admin@example.com',
        'uploader-name': 'John Doe',
        'upload-source': 'web-portal',
        'is-suspicious': 'true'
    }

    # Step 3: Copy with full metadata
    s3.copy_object(
        Bucket=BUCKET,
        Key=KEY,
        CopySource={'Bucket': BUCKET, 'Key': KEY},
        Metadata=new_metadata,
        MetadataDirective='REPLACE',
        ContentType=head['ContentType'],   # preserve
        ACL='public-read'                  # if needed
    )

    print("✅ Metadata safely updated")

