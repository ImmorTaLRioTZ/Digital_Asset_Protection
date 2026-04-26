from pathlib import Path
import urllib.request

def ensure_sscd_model_exists(model_dir: Path):
    """Downloads the SSCD TorchScript model directly from Meta if it isn't cached."""
    model_path = model_dir / "sscd_disc_mixup.torchscript.pt"
    
    if not model_path.exists():
        print("SSCD Model not found locally. Downloading from Facebook Research (~100MB)...")
        url = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
        # Create directory if it doesn't exist
        model_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(model_path))
        print("✅ Download complete!")
        print(f"SSCD downloaded at {model_path}")
    return str(model_path)




#(Self-Supervised Copy Detection)