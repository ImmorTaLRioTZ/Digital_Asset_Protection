import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import os

def extract_keyframes(video_path, fps_interval=1):
    """Extracts 1 frame per second from a video."""
    print(f"Extracting frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps * fps_interval))
    
    frames = []
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)
            
        current_frame += 1

    cap.release()
    print(f"Extracted {len(frames)} keyframes.")
    return frames

def get_sequence_vectors(frames, model, processor, device, batch_size=32):
    """
    Processes frames in batches to prevent GPU memory overflow.
    Returns a tensor of shape [Num_Frames, 512].
    """
    all_vectors = []
    
    # Process frames in small chunks (e.g., 32 at a time)
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        
        with torch.no_grad():
            batch_vectors = model.get_image_features(**inputs)
            # Normalize the batch vectors immediately
            batch_vectors = F.normalize(batch_vectors, p=2, dim=1)
            
        all_vectors.append(batch_vectors)
        
    # Concatenate all batches back into one continuous tensor timeline
    return torch.cat(all_vectors, dim=0)

def find_clip_in_long_video(official_vectors, suspect_vectors, threshold=0.95):
    """Slides the suspect window across the official timeline."""
    num_official_frames = official_vectors.shape[0]
    num_suspect_frames = suspect_vectors.shape[0]
    
    if num_suspect_frames == 0 or num_official_frames == 0:
        return 0.0, -1

    # if num_suspect_frames > num_official_frames:
    #     print("Error: Suspect video is longer than the official video.")
    #     return 0.0, -1

    print(f"\nSearching for a {num_suspect_frames}s clip inside a {num_official_frames}s broadcast...")

    best_match_score = 0.0
    best_timestamp = -1

    for i in range(num_official_frames - num_suspect_frames + 1):
        official_window = official_vectors[i : i + num_suspect_frames]
        
        # Calculate similarity for the entire block simultaneously
        similarities = F.cosine_similarity(official_window, suspect_vectors)
        window_score = similarities.mean().item()
        
        if window_score > best_match_score:
            best_match_score = window_score
            best_timestamp = i

    print(f"\n--- SEQUENCE MATCH RESULTS ---")
    if best_match_score >= threshold:
        print("🚨 VERDICT: PIRACY DETECTED.")
        print(f"Match Confidence: {best_match_score:.4f}")
        print(f"Stolen clip found at timestamp: {best_timestamp} seconds into the official broadcast.")
    else:
        print("✅ VERDICT: CLEAR.")
        print(f"Highest similarity found was {best_match_score:.4f}, below threshold.")
        
    return best_match_score, best_timestamp

def verify_variable_video_piracy(official_video_path: str, suspect_video_path: str):
    print("Loading Vision Model into memory...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
    model = CLIPModel.from_pretrained(model_id)

    # Automatically use the dedicated GPU if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    try:
        # 1. Extract frames from both videos
        official_frames = extract_keyframes(official_video_path)
        suspect_frames = extract_keyframes(suspect_video_path)
        
        # 2. Generate the sequential vectors (with batching to protect memory)
        print("\nGenerating AI Vector Sequences...")
        vector_official = get_sequence_vectors(official_frames, model, processor, device)
        vector_suspect = get_sequence_vectors(suspect_frames, model, processor, device)
        
        # Free up VRAM after inference is complete
        torch.cuda.empty_cache() 

        # 3. Execute the sliding window mathematical search
        find_clip_in_long_video(vector_official, vector_suspect)

    except Exception as e:
        print(f"Pipeline Error: {e}")

# Run the pipeline
if __name__ == "__main__":
    # 1. Get the directory of THIS script (the 'src' folder)
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # 2. Go UP one level to the main project folder, then into 'videos'
    PROJECT_ROOT = SCRIPT_DIR.parent
    official_vid = PROJECT_ROOT / "videos" / "official.mp4"
    suspect_vid = PROJECT_ROOT / "videos" / "official_BW.mp4"

    print(f"Targeting official video at: {official_vid}")
    
    # 3. Quick sanity check before running the heavy model
    if not official_vid.exists():
        print(f"CRITICAL ERROR: Cannot find the official video at {official_vid}")
        print("Check your folder structure and ensure the filename is exactly 'official.mp4' (lowercase).")
        exit()
        
    if not suspect_vid.exists():
        print(f"CRITICAL ERROR: Cannot find the suspect video at {suspect_vid}")
        exit()

    # 4. Pass them to your function
    verify_variable_video_piracy(str(official_vid), str(suspect_vid))