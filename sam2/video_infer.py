import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from build_sam import build_sam2_video_predictor
import shutil

from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_dir, initialize
from omegaconf import DictConfig, OmegaConf

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

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

def predict_video(predictor, video_dir, output_dir, keyframes):

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    # initialize state
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    
    for keyframe in keyframes:
        # Load the keyframe
        #frame_file = keyframe["frame_file"]
        #keyframe_path = os.path.join(video_dir, frame_file)
        #keyframe_image = Image.open(keyframe_path)
        
        objects = keyframe["objects"]
        prompts = {}
        #ann_frame_idx = int(keyframe["frame_id"])
        
        for i, object in enumerate(objects):
            points = []
            labels = []
            for j, obj in enumerate(objects):
                if i == j:
                    points += obj["points"]
                    labels += obj["labels"]
                else:
                    for l in range(len(obj["labels"])):
                        if obj["labels"][l] == 0:
                            points.append(obj['points'][l])
                            labels.append(0)
            
            print(f"points: {points}")
            print(f"labels: {labels}")
            ann_obj_id = object['obj_id']
            points = np.array(points, dtype=np.float32)
            labels = np.array(labels, np.int32)
            prompts[ann_obj_id] = points, labels
            
            # predict
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

        
    # Define a color palette for 11 classes
    palette = [
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

    # Extend the palette to have 256 classes (with 3 values per class).
    # Fill the rest with black (or any default color).
    palette += [0, 0, 0] * (256 - len(palette) // 3)
    
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
    # Save the combined mask results of every frame to output_dir
    for out_frame_idx in range(len(frame_names)):
        frame_name = frame_names[out_frame_idx]
        frame_path = os.path.join(video_dir, frame_name)
        frame_image = Image.open(frame_path)
        h, w = frame_image.size[::-1]  # Get original height and width
        combined_mask = np.zeros((h, w, 1), dtype=np.uint8)  # Initialize an empty mask with class labels
        
        # Combine all masks for the current frame
        for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
            mask = (out_mask * out_obj_id).reshape(h, w, 1).astype(np.uint8)
            combined_mask = np.maximum(combined_mask, mask)  # Combine masks, keeping the highest label value
        
        # Save the combined mask as a .png file
        output_mask_path = os.path.join(output_dir, f"{os.path.splitext(frame_name)[0]}.png")
        combined_mask_image = Image.fromarray(combined_mask.squeeze(axis=-1))
        
        # Apply the palette
        combined_mask_image.putpalette(palette)
        combined_mask_image.save(output_mask_path)
    
    print(f"All masks saved to {output_dir}") 

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
    parser.add_argument("--output_spec", type=str, default="masks_sam_gpt", help="Path to the output directory.")
    
    return parser.parse_args()
    

def main():
    args = parse_arguments()
    
    video_dir = os.path.join(args.input_dir, args.input_spec)
    keyframes_json_path = os.path.join(args.input_dir, "prompts.json")
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
    predictor = build_sam2_video_predictor(cfg_path, sam2_checkpoint, device=device)
    print("Predictor initialized successfully.")
        
    # Load keyframes from the provided JSON file
    keyframes = load_keyframes(keyframes_json_path)
    
    # sort keyframs by frame_id
    keyframes.sort(key=lambda x: x["frame_id"])
    frame_ids = [keyframe["frame_id"] for keyframe in keyframes]
    
    # Save masks as all 0 for frames before the first keyframe
    first_frame_file = keyframes[0]["frame_file"]
    # convert the first filename to digit
    first_frame_file_num = int(os.path.splitext(first_frame_file)[0])
    frame_names = sorted([
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ], key=lambda p: int(os.path.splitext(p)[0]))
    
    #  check frame names before first_frame_file
    for frame_name in frame_names:
        frame_idx = int(os.path.splitext(frame_name)[0])
        if frame_idx < first_frame_file_num:
            frame_path = os.path.join(video_dir, frame_name)
            frame_image = Image.open(frame_path)
            h, w = frame_image.size[::-1]  # Get original height and width
            combined_mask = np.zeros((h, w, 1), dtype=np.uint8)  # Initialize an empty mask with class labels
            output_mask_path = os.path.join(output_dir, f"{frame_name.split('.')[0]}.png")
            combined_mask_image = Image.fromarray(combined_mask.squeeze(axis=-1))
            combined_mask_image.save(output_mask_path)
        else:
            break
            
    # Create a temporary folder to process images between keyframes
    temp_folder = os.path.join(args.input_dir, "temp")
    if os.path.exists(temp_folder):
        for temp_file in os.listdir(temp_folder):
            temp_file_path = os.path.join(temp_folder, temp_file)
            os.remove(temp_file_path)
        # Delete the temp folder after processing
        os.rmdir(temp_folder)

    os.makedirs(temp_folder)
    
    # Run prediction for each segment between keyframes
    for i in range(len(keyframes)):
        start_frame_id = keyframes[i]["frame_id"]
        end_frame_id = keyframes[i + 1]["frame_id"] if i + 1 < len(keyframes) else len(frame_names)

        print(f"Processing segment {i + 1} between frames {start_frame_id} and {end_frame_id}")
        # Copy frames between start_frame_id and end_frame_id to the temp folder
        for frame_idx in range(start_frame_id, end_frame_id):
            frame_name = frame_names[frame_idx]
            src_path = os.path.join(video_dir, frame_name)
            dst_path = os.path.join(temp_folder, frame_name)
            shutil.copy(src_path, dst_path)
        
        # if the temp folder is not empty
        if len(os.listdir(temp_folder)) != 0:  
            # Run prediction on the current segment
            predict_video(predictor, temp_folder, output_dir, [keyframes[i]])
        
        # Clear the temp folder for the next iteration
        for temp_file in os.listdir(temp_folder):
            temp_file_path = os.path.join(temp_folder, temp_file)
            os.remove(temp_file_path)
    
    # Delete the temp folder after processing
    os.rmdir(temp_folder)

if __name__ == "__main__":
    main()