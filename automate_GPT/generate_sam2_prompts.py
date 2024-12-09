#%pip install openai==1.55.3 httpx==0.27.2
import os
import json
import glob
import base64
import random
import numpy as np
from openai import OpenAI
import argparse

# Retrieve the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key exists
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# remove excessive characters from chatgpt's response
def strip_JSON(response):
    result = response.strip("```")
    result = result.strip("json")
    result = json.loads(result)
    return result

# converting chatgpt's response into our ideal JSON format
def format_JSON(coordinates,frame_id,frame_file):
    objects_list = []
    for obj_id, bbox in coordinates.items():
        # Convert relative bbox to pixel coordinates (assuming width 854x height 480)
        width, height = 854, 480
        top_left_x = int(bbox[0] * width)
        top_left_y = int(bbox[1] * height)
        bottom_right_x = int(bbox[2] * width)
        bottom_right_y = int(bbox[3] * height)

        # Create center point as the label 1 point
        points = [
            [int(np.mean([top_left_x,bottom_right_x])),int(np.mean([top_left_y, bottom_right_y]))]
        ]

        # Create far away points as the label 0 point based on the tool (due to different geometries and orientations)
        match obj_id:
            case "2" :
                points.append([top_left_x -random.randint(5,top_left_x-20),bottom_right_y + random.randint(5, height-bottom_right_y-10)])
                points.append([bottom_right_x + random.randint(5,width - bottom_right_x-20), top_left_y -random.randint(5,top_left_y-10)])
            case "8" :
                points.append([top_left_x -random.randint(5,top_left_x-20),bottom_right_y + random.randint(5, height-bottom_right_y-10)])
                points.append([bottom_right_x + random.randint(5,width - bottom_right_x-20), top_left_y -random.randint(5,top_left_y-10)])
            case "1" :
                points.append([top_left_x -random.randint(20,top_left_x-20),top_left_y])
                points.append([bottom_right_x + random.randint(20,width - bottom_right_x-20), bottom_right_y])
            case "10" :
                points.append([top_left_x -random.randint(20,top_left_x-20),top_left_y])
                points.append([bottom_right_x + random.randint(20,width - bottom_right_x-20), bottom_right_y])
            case _:
                points.append([top_left_x -random.randint(5,top_left_x-20),top_left_y -random.randint(5,top_left_y-10)])
                points.append([bottom_right_x + random.randint(5,width - bottom_right_x-20), bottom_right_y + random.randint(5, height-bottom_right_y-10)])

        # Labels corresponding to the 1 label 1 point and 2 label 0 points
        labels = [1,0,0]

        # Create dictionary objects for JSON output
        obj_dict = {
            "obj_id": int(obj_id),
            "points": points,
            "labels": labels
        }
        objects_list.append(obj_dict)

    frame_result = {
        "frame_id": frame_id,
        "frame_file": frame_file,
        "objects": objects_list
    }
    return frame_result

def generate_gpt_output_for_frame(input_dir, frame_id, frame_file):
    # for first gpt inquiry
    prompt1 = """I am going to show you images of a gallblader removal surgery,
    These images contain a white gallblader in the middle of the image, and the following tools only:
    Left Grasper, Top Grasper, Right Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, Specimen Bag.
    We are interested in segmenting the gallblader and the aforementioned tools, and your job is to locate them and provide coordinates in pixels."""
    # for second gpt inquiry
    prompt2 = """Now locate the objects you have just identified, on the same image.
    Return the response **only** in valid JSON format with the following structure:
    1. Each surgical tool should be identified by a number with the following correspondance to predefined classes:
    - Gallbladder: 1
    - Left Grasper: 2
    - Top Grasper: 3
    - Right Grasper: 4
    - Bipolar: 5
    - Hook: 6
    - Scissors: 7
    - Clipper: 8
    - Irrigator: 9
    - Specimen Bag: 10

    2. For each detected tool, provide bounding box coordinates (relative to the image width and height, between 0 and 1 and of 3 decimal points precision) in the format:
    - [top_left_x, top_left_y, bottom_right_x, bottom_right_y].

    3. If no tools are detected, return an empty JSON object (`{}`).

    ### Rules:
    - Respond strictly in JSON format.
    - Do not include any explanations, comments, or text outside of the JSON response.
    - all identifier numbers must have quotation marks.
    ### Example Output:
    {
        "6": [0.052, 0.024, 0.854, 0.953],
        "2": [0.014, 0.045, 0.917, 0.854]
    }
    """
    all_results = []
    image_path = os.path.join(input_dir, frame_file)
    # Encode the image for GPT
    base64_image = encode_image(image_path)

    # First GPT call (analysis)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI specialized in surgical phase recognition."},
            {"role": "system", "content": prompt1},
            {"role": "system", "content": "This is one of the images."},
            {"role": "user", "content": [
                {
                "type": "image_url",
                "image_url": {"url":  f"data:image/jpeg;base64,{base64_image}"}
                }
            ]}
            #{"role": "system", "content": "List only the classes you see in this image."},
        ],
    )
    print("Analysis response for frame", frame_id, ":", response.choices[0].message.content)
    assistant_response1 = response.choices[0].message.content

    # Second GPT call (coordinates)
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an AI specialized in surgical phase recognition."},
        {"role": "system", "content": prompt1},
        {"role": "system", "content": "This is an example of the image."},
        {"role": "user", "content": [
            {
            "type": "image_url",
            "image_url": {"url":  f"data:image/jpeg;base64,{base64_image}"}
            }
        ]},
        {"role": "assistant", "content": assistant_response1},
        {"role": "system", "content": prompt2},
    ]
    )

    json_response = response.choices[0].message.content
    print("JSON Response for frame", frame_id, ":", json_response)

    # Convert to Python dictionary
    try:
        coordinates = strip_JSON(json_response)

    except json.JSONDecodeError:
        coordinates = {}
        
    if not coordinates:
        return assistant_response1, None
    return assistant_response1, format_JSON(coordinates,frame_id,frame_file)

# Getting frame ids from a input JSON file
def get_frame_ids(frame_ids, input_json):
    with open(input_json, 'r') as file:
        input_content = json.load(file)
    for frame in input_content:
        frame_ids.append(frame["frame_id"])
    return frame_ids

# CURRENTLY NOT USED: function to detect new frames to log into prompt.JSON via ChatGPT
def det_new_frame(input_dir,frame_file_old,frame_file_new,assistant_response1):
    prompt2 = """Given your analysis of the previous image, now determine whether the new uploaded image contains new surgical tools or some of the tools present in the old image is missing?
    ### Rules:
    - Respond "Yes" if any new surgical tools are present in the new image, or if any surgical tools in the previous image is missing.
    - Respond "No" if no new surgical tools are present in the new image or no surgical tools in the previous image is missing.
    """
    # The previously analyzed image
    image_path_old = input_dir + frame_file_old
    base64_image_old = encode_image(image_path_old)
    # The new image to be compared
    image_path_new = input_dir + frame_file_new
    base64_image_new = encode_image(image_path_new)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            #from previous chatgpt analysis
            {"role": "system", "content": "You are an AI specialized in surgical phase recognition."},
            {"role": "user", "content": [
                {
                "type": "image_url",
                "image_url": {"url":  f"data:image/jpeg;base64,{base64_image_old}"}
                }
            ]},
            {"role": "assistant", "content": assistant_response1},
            #comparison between images
            {"role": "user", "content": [
                {
                "type": "image_url",
                "image_url": {"url":  f"data:image/jpeg;base64,{base64_image_new}"}
                }
            ]},
            {"role": "system", "content": prompt2}
        ],
        max_tokens=1
    )
    return response.choices[0].message.content #Yes if tools are different, No if not different.

# Main loop over video frames and extract tool data into prompt.JSON
def process_frames(input_dir, output_json, step=25, input_json = None):
    """
    - input_dir: Directory containing frames (images).
    - output_json: Path to the output JSON file (prompts.json).
    - step: The interval for selecting key frames. Default every 25 frames. If using autodetection, set to 1!
    """
    # Find all jpg images in the directory. Adjust extension if needed.
    # Sort them to ensure correct sequential order.
    frame_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    # List to hold all GPT results
    all_results = []
    frame_ids = []
    """
    # placeholder for initial GPT response (for auto identifying new frames)
    assistant_response1 = ""
    """

    # Iterate through frames, picking every 'step'th frame
    if input_json:
        frame_ids = get_frame_ids(frame_ids, input_json)
    else:
        frame_ids = range(0, len(frame_files), step)

    for i in frame_ids:
        frame_path = frame_files[i]
        # Extract frame filename (e.g. "0000000.jpg")
        frame_file = os.path.basename(frame_path)
        # Extract a numeric frame_id from the index i
        frame_id = i
        
        """
        # auto identificying new frames via ChatGPT
        if (i != 0):
            new_frame = det_new_frame(input_dir, frame_file_prev,frame_file, assistant_response1)
            print(frame_file, new_frame)
            if new_frame == "No": 
                continue
        frame_file_prev = frame_file
        """

        # Here integrates GPT pipeline .
        assistant_response1, gpt_output = generate_gpt_output_for_frame(input_dir, frame_id, frame_file)

        if gpt_output:
            # Append the result to our list
            all_results.append(gpt_output)

    # Ensure the output JSON file exists in the input JSON's directory
    output_dir = os.path.dirname(output_json)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the final JSON file
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video frames and generate a GPT-based JSON file.")
    parser.add_argument("--input_dir", required=True, help="Directory containing video frames (images).")
    parser.add_argument("--input_json", required=True, help="Path to the input JSON file.")
    parser.add_argument("--step", type=int, default=25, help="Interval for selecting frames (default: 25).")
    args = parser.parse_args()

    input_dir = args.input_dir
    input_json = args.input_json

    # Define output JSON file path
    output_json = os.path.join(os.path.dirname(input_json), "gpt_generated_prompts.json")
    process_frames(input_dir, output_json, args.step, input_json)