import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from build_sam import build_sam2
from sam2_image_predictor import SAM2ImagePredictor

import shutil

from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_dir, initialize
from omegaconf import DictConfig, OmegaConf

# Define the mapping of detection labels to segmentation mask values
LABEL_MAPPING = {
    "gallbladder": 1,
    "grasper": 2,
    "bipolar": 5,
    "hook": 6,
    "scissors": 7,
    "clipper": 8,
    "irrigator": 9,
    "specimenbag": 10
}

# Define a color palette for 11 classes
PALETTE = [
    0, 0, 0,        # Class 0: Black
    255, 0, 0,      # Class 1: Red
    0, 255, 0,      # Class 2: Green
    0, 0, 255,      # Class 3: Blue
    255, 255, 0,    # Class 4: Yellow
    255, 0, 255,    # Class 5: Magenta
    0, 255, 255,    # Class 6: Cyan
    128, 128, 128,  # Class 7: Gray
    128, 0, 0,      # Class 8: Maroon
    0, 128, 0,      # Class 9: Dark Green
    0, 0, 128,      # Class 10: Navy Blue
]
PALETTE += [0, 0, 0] * (256 - len(PALETTE) // 3)

def setup_hydra():
    """Ensure Hydra is initialized properly."""
    # Clear any previous Hydra instance
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize(config_path="configs/sam2", version_base="1.2")

def setup_device():
    """Sets up the correct device (CUDA, MPS, or CPU) and configures options."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        # Use bfloat16 precision for CUDA
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # Enable TF32 precision for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA "
            "and might give numerically different outputs and sometimes degraded performance. "
            "See https://github.com/pytorch/pytorch/issues/84936 for details."
        )
    
    return device

def process_frame(predictor, frame_path, details, output_path):
    """Process a single frame and save the segmentation mask."""
    frame_image = Image.open(frame_path).convert("RGB")  # Ensure the image is in RGB format
    predictor.set_image(frame_image)
    h, w = frame_image.size[::-1]  # Get height and width
    combined_mask = np.zeros((h, w, 1), dtype=np.uint8)

    if details:
        for obj_key, obj_details in details.items():
            label = obj_details.get("detection_label", "").lower()
            mask_value = LABEL_MAPPING.get(label)
            if mask_value is None:
                print(f"Unknown label '{label}' for {obj_key}, skipping...")
                continue

            # Prepare bounding box input
            detection_box = obj_details.get("detection_box")
            if detection_box:
                bbox = np.array(detection_box, dtype=np.float32)  # Ensure it's a NumPy array
                
                # Predict mask using the bounding box
                masks, scores, logits = predictor.predict(
                    box=bbox,
                    multimask_output=False  # Use single mask for less ambiguity
                )

                # Use the first mask (or the only mask when multimask_output=False)
                if masks is not None and len(masks) > 0:
                    mask = (masks[0] * mask_value).reshape(h, w, 1).astype(np.uint8)
                    combined_mask = np.maximum(combined_mask, mask)  # Combine masks
    
    # Save the combined mask as a .png file
    combined_mask_image = Image.fromarray(combined_mask.squeeze(axis=-1))
    combined_mask_image.putpalette(PALETTE)
    combined_mask_image.save(output_path)

def load_keyframes(json_path):
    """Load and parse keyframes from the given JSON file."""
    with open(json_path, 'r') as f:
        keyframes = json.load(f)
    return keyframes         

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SAM2 video inference.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory.")
    parser.add_argument("--input_spec", type=str, default="images", help="Path to the output directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--output_spec", type=str, default="masks_sam", help="Path to the output directory.")
    parser.add_argument("--prompts_type", type=str, default="grounding_prompts.json", help="prompts type")
    
    return parser.parse_args()
    

def main():
    args = parse_arguments()
    
    video_dir = os.path.join(args.input_dir, args.input_spec)
    keyframes_json_path = os.path.join(args.input_dir, args.prompts_type)
    output_dir = os.path.join(args.output_dir, args.output_spec)
        
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Setup Hydra properly
    setup_hydra()

    device = setup_device()

    # Build the SAM2 video predictor
    sam2_checkpoint = "/mnt/disk0/haoding/SAM2/checkpoints/sam2_hiera_large.pt"
    cfg_path = "sam2_hiera_l.yaml"
    print(cfg_path)  # Verify that the config is loaded correctly
    sam2_model = build_sam2(cfg_path, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("Predictor initialized successfully.")
        
    # Load keyframes from the provided JSON file
    keyframes = load_keyframes(keyframes_json_path)
    
    # iterate over each frame:
    for frame, details in keyframes.items():
        frame_path = os.path.join(video_dir, frame)
        if not os.path.exists(frame_path):
            print(f"Frame {frame} not found, skipping...")
            continue

        output_path = os.path.join(output_dir, f"{os.path.splitext(frame)[0]}.png")
        process_frame(predictor, frame_path, details, output_path)


if __name__ == "__main__":
    main()