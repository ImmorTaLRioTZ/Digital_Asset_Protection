from pathlib import Path
from src.pipeline.pipeline import DetectionPipeline, PipelineConfig

def main():

    # Initialize configuration, pointing to where your SSCD model weights are/will be saved
    config = PipelineConfig(sscd_model_dir=Path("models/"))

    # Create the pipeline orchestrator
    pipeline = DetectionPipeline(config)

    videos_dir = Path("videos")

    official_video = videos_dir / "batch1_official.mp4"
    suspect_video = videos_dir / "batch1_suspect.mp4"

    # Pass the path to your official video and give it a unique ID
    pipeline.register_asset(official_video, asset_id="official_movie_01", skip_audio=False)
    # Pass the path to the suspect video
    result = pipeline.check(suspect_video)

    print(result)

if __name__ == "__main__":
    main()
