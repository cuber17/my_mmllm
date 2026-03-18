import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
OUTPUT_DIR = "./vis_outputs"  # All images will be saved here

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def load_npy(path):
    """Load .npy file safely."""
    try:
        arr = np.load(path)
        arr = arr.astype(np.float32)
        # Squeeze channel dim if needed: (1, H, W) -> (H, W) or (3, H, W) -> (H, W) if grayscale
        if arr.ndim == 3 and arr.shape[0] in [1, 3]: 
            arr = arr[0] 
        return arr
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def visualize_single_file(npy_path):
    """
    Visualize a single .npy file and save the image.
    """
    # 1. Resolve absolute path
    file_path = Path(npy_path).resolve()
    
    if not file_path.exists():
        print(f"❌ Error: File not found at {file_path}")
        return

    # 2. Load Data
    data = load_npy(file_path)
    if data is None: return

    # 3. Determine Title based on filename convention
    filename = file_path.name
    stem = file_path.stem # e.g., "011761_td_aug1"
    
    title = f"File: {filename}"
    if "_td" in stem: title += "\n(Range-Doppler)"
    elif "_tr" in stem: title += "\n(Range-Time)"
    elif "_ta" in stem: title += "\n(Range-Angle)"

    # 4. Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use 'jet' colormap typical for radar heatmaps
    im = ax.imshow(data, cmap='jet', aspect='auto', interpolation='nearest')
    
    ax.set_title(f"{title}\nShape: {data.shape} | Range: [{data.min():.2f}, {data.max():.2f}]")
    fig.colorbar(im, ax=ax, label='Amplitude')
    plt.tight_layout()
    
    # 5. Save
    ensure_output_dir()
    save_filename = f"vis_{stem}.png"
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    
    plt.savefig(save_path)
    plt.close()
    
    print(f"✅ Saved visualization to: {save_path}")

if __name__ == "__main__":
    print("-" * 50)
    print("Single Radar NPY Visualizer")
    print(f"Outputs will be saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("-" * 50)

    # Mode 1: Command Line Argument
    # Example: python visualize_radar.py ./data/sample.npy
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        visualize_single_file(input_path)
        
    # Mode 2: Interactive Loop
    else:
        while True:
            # Get input
            user_input = input("\nEnter path to .npy file (or 'q' to quit): ").strip()
            
            # Remove quotes if user dragged & dropped file in terminal
            user_input = user_input.strip("'").strip('"')
            
            if user_input.lower() in ['q', 'exit']:
                break
            
            if not user_input:
                continue
                
            visualize_single_file(user_input)