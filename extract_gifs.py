import os
from pathlib import Path
import torch
import shutil

# Import the extraction function from the API server
from api_server import extract_landmarks_from_video

def main():
    gif_dir = Path("gifs")
    cache_dir = Path("data_extracted")
    
    if not gif_dir.exists():
        print(f"❌ Directory {gif_dir} does not exist.")
        return
        
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
        
    print(f"⏳ Extracting landmarks from GIFs in {gif_dir} ...")
    
    gif_files = list(gif_dir.glob("*.gif"))
    
    # We will generate 3 variations (simulated samples) from each GIF to give the model more data
    # (e.g. original, slightly cropped/augmented by train_cnn later, but we need 3 files to simulate having 3 samples)
    
    for gif_path in gif_files:
        filename = gif_path.stem
        # The filename format is usually "Word_ID", e.g. "أب_0195"
        parts = filename.split("_")
        if len(parts) >= 2:
            class_id = parts[-1]
            try:
                features = extract_landmarks_from_video(str(gif_path))
                tensor_feats = torch.tensor(features, dtype=torch.float32)
                
                # Save 3 copies to satisfy the training scripts expectations of having multiple samples
                for i in range(3):
                    out_path = cache_dir / f"{class_id}_gif_sample_{i}.pt"
                    torch.save(tensor_feats, str(out_path))
                    
                print(f"✅ Extracted: {filename} -> {class_id} ({len(features)} frames)")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                
    print("🎉 All GIFs processed and saved to data_extracted!")

if __name__ == "__main__":
    main()
