import os
import json
import glob

# TODO: replace this with actual implementation
def generate_gpt_output_for_frame(frame_id, frame_file):
    """
    Placeholder function simulating GPT-4o output.
    Replace with actual GPT inference code.

    This function should return a Python dictionary structured like:
    {
       "frame_id": <int>,
       "frame_file": <string>,
       "objects": [
         {
           "obj_id": <int>,
           "points": [[x1, y1], [x2, y2], ...],
           "labels": [0 or 1, 0 or 1, ...]
         },
         ...
       ]
    }
    """
    # example of returned data.
    return {
        "frame_id": frame_id,
        "frame_file": frame_file,
        "objects": [
            {
                "obj_id": 1,
                "points": [
                    [434, 163],
                    [494, 218],
                    [293, 190],
                    [596, 143],
                    [610, 438]
                ],
                "labels": [1, 1, 0, 0, 0]
            }
            # more objects
        ]
    }

def process_frames(input_dir, output_json, step=25):
    """
    - input_dir: Directory containing frames (images).
    - output_json: Path to the output JSON file (prompts.json).
    - step: The interval for selecting key frames. Default every 25 frames.
    """

    # Find all jpg images in the directory. Adjust extension if needed.
    # Sort them to ensure correct sequential order.
    frame_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))

    # List to hold all GPT results
    all_results = []

    # Iterate through frames, picking every 'step'th frame
    for i in range(0, len(frame_files), step):
        frame_path = frame_files[i]
        
        # Extract frame filename (e.g. "0000000.jpg")
        frame_file = os.path.basename(frame_path)
        
        # Extract a numeric frame_id from the index i (optional, can be just i)
        frame_id = i

        # Here you would integrate your GPT pipeline code.
        # For now, we call our placeholder.
        gpt_output = generate_gpt_output_for_frame(frame_id, frame_file)

        # Append the result to our list
        all_results.append(gpt_output)

    # Once done, write the final JSON file
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    input_dir = "/path/to/your/frames"  # Replace with your frames directory
    output_json = "prompts.json"

    process_frames(input_dir, output_json, step=25)