import time
import os
import boto3
from pathlib import Path
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pipeline.pipeline import DetectionPipeline, PipelineConfig

load_dotenv()

# --- Configuration ---
BUCKET_NAME = os.getenv("BUCKET_NAME")
REGION = os.getenv("REGION")
POLL_INTERVAL_SECONDS = 10
MAILTRAP_SMTP_HOST=os.getenv("MAILTRAP_SMTP_HOST")
MAILTRAP_SMTP_PORT=int(os.getenv("MAILTRAP_SMTP_PORT"))
MAILTRAP_SMTP_USER=os.getenv("MAILTRAP_SMTP_USER")
MAILTRAP_SMTP_PASS=os.getenv("MAILTRAP_SMTP_PASS")

OFFICIAL_DIR = Path("videos/officials")
SUSPECT_DIR = Path("videos/incoming")

OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
SUSPECT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize AWS Client
s3_client = boto3.client('s3', region_name=REGION)

OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
SUSPECT_DIR.mkdir(parents=True, exist_ok=True)

# Memory cache to prevent re-processing
PROCESSED_SUSPECTS = set()
REGISTERED_OFFICIALS = set()

def send_dmca_notice(target_email: str, filename: str, matched_asset: str, confidence: str):
    """Drafts and sends an automated DMCA takedown notice via Mailtrap."""
    if not target_email or target_email == "Unknown":
        print("   ⚠️ No valid email found for suspect. Cannot send DMCA.")
        return

    print(f"   ✉️ Drafting DMCA Notice to {target_email}...")
    
    subject = f"URGENT: Automated DMCA Takedown Notice - Copyright Infringement ({filename})"
    body = f"""Dear Site Administrator / Uploader,

This is an automated notification of copyright infringement pursuant to the Digital Millennium Copyright Act (DMCA).

Our automated Video Fingerprinting system has detected that the file you uploaded, identified as '{filename}', contains unauthorized copyrighted material matching our registered master asset '{matched_asset}' with a confidence score of {confidence}.

We have a good faith belief that the use of the material in the manner complained of is not authorized by the copyright owner, its agent, or the law. 

We hereby demand that you immediately disable access to or remove the infringing material.

Sincerely,
Digital Asset Protection Bot
Automated Enforcement Division
"""

    # Construct the email
    msg = MIMEMultipart()
    msg['From'] = MAILTRAP_SMTP_USER
    msg['To'] = target_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to Mailtrap and send
        server = smtplib.SMTP(MAILTRAP_SMTP_HOST, MAILTRAP_SMTP_PORT)
        server.starttls() 
        server.login(MAILTRAP_SMTP_USER, MAILTRAP_SMTP_PASS) # ⬅️ Uses Mailtrap credentials
        server.send_message(msg)
        server.quit()
        print(f"   ✅ Simulated DMCA Notice successfully caught by Mailtrap for {target_email}!")
    except Exception as e:
        print(f"   ❌ Failed to send simulated email: {e}")

def sync_official_assets(pipeline: DetectionPipeline):
    """Pulls new official reference videos from S3 and registers them."""
    print("🔄 Syncing official master assets from S3...")
    
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="official_")
        
        for obj in response.get("Contents", []):
            key = obj["Key"]
            
            if key in REGISTERED_OFFICIALS:
                continue  # Already registered in memory
                
            local_path = OFFICIAL_DIR / key
            
            # Download if we don't have it locally
            if not local_path.exists():
                print(f"   ⬇️ Downloading master asset: {key}")
                s3_client.download_file(BUCKET_NAME, key, str(local_path))
            
            # Fetch metadata to get the actual asset_id
            head = s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
            asset_id = head.get("Metadata", {}).get("asset-id", key)
            
            print(f"   🛡️ Registering asset '{asset_id}' into pipeline...")
            pipeline.register_asset(local_path, asset_id=asset_id, skip_audio=False)
            
            REGISTERED_OFFICIALS.add(key)
            
    except Exception as e:
        print(f"❌ Failed to sync official assets: {e}")


def poll_suspects(pipeline: DetectionPipeline):
    """Scans S3 for new suspect videos, downloads, and checks them."""
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="suspect_")
        
        for obj in response.get("Contents", []):
            key = obj["Key"]
            
            if key in PROCESSED_SUSPECTS:
                continue
                
            print("\n" + "="*60)
            print(f"🚨 NEW SUSPECT DETECTED: {key}")
            
            local_path = SUSPECT_DIR / key
            
            try:
                # Fetch Metadata first to get the target email
                head = s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
                metadata = head.get("Metadata", {})
                suspect_email = metadata.get("uploader-email", "Unknown")

                # 1. Download suspect video
                print(f"   ⬇️ Downloading suspect video for analysis...")
                s3_client.download_file(BUCKET_NAME, key, str(local_path))
                
                # 2. Run Pipeline
                print(f"   🔍 Running SSCD & pHash analysis...")
                result = pipeline.check(local_path)
                
                # 3. Output Verdict
                if result.is_pirated:
                    print(f"   ⛔ PIRACY ALERT: Matched Master '{result.asset_id}'")
                    print(f"      Caught via: {result.detected_by}")
                    print(f"      Confidence: {result.confidence}")

                    print(f"   🏷️ Tagging S3 file as DMCA Struck...")
                    new_metadata = {
                        **metadata,  # Keep all existing tags (like email and uploader)
                        'dmca-struck': 'true'
                    }

                    s3_client.copy_object(
                        Bucket=BUCKET_NAME,
                        Key=key,
                        CopySource={'Bucket': BUCKET_NAME, 'Key': key},
                        Metadata=new_metadata,
                        MetadataDirective='REPLACE',
                        ContentType=head.get('ContentType', 'video/mp4')
                    )

                    send_dmca_notice(
                        target_email=suspect_email, 
                        filename=key, 
                        matched_asset=result.asset_id,
                        confidence=str(result.confidence)
                    )
                else:
                    print(f"   ✅ CLEAN: Video passed all checks.")
                    
            except Exception as e:
                print(f"   ❌ Analysis failed for {key}: {e}")
                
            finally:
                # 4. Clean up disk to save space
                if local_path.exists():
                    local_path.unlink()
                
                PROCESSED_SUSPECTS.add(key)
                
    except Exception as e:
        print(f"❌ Failed to poll suspects: {e}")


if __name__ == "__main__":
    print("⚙️ Initializing Detection Pipeline Engine...")
    config = PipelineConfig(sscd_model_dir=Path("models/"))
    detector = DetectionPipeline(config)
    
    print(f"\n🚀 Starting Continuous Detection Worker. Polling every {POLL_INTERVAL_SECONDS}s.")
    print("Press Ctrl+C to stop.")
    print("-" * 60)
    
    try:
        while True:
            # Always ensure we have the latest masters before checking suspects
            sync_official_assets(detector)
            
            # Check for piracy
            poll_suspects(detector)
            
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n🛑 Worker stopped gracefully.")