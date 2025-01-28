import os
import json
import glob
import base64
import numpy as np
from PIL import Image
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

def classify_binary_mask(image_path):
    # GPT prompt for classification
    prompt = """
    I am providing you with binary segmentation masks for surgical images. 
    Each binary mask represents one of the following classes:
    - Gallbladder (class 1)
    - Left Grasper (class 2)
    - Top Grasper (class 3)
    - Right Grasper (class 4)
    - Bipolar (class 5)
    - Hook (class 6)
    - Scissors (class 7)
    - Clipper (class 8)
    - Irrigator (class 9)
    - Specimen Bag (class 10)

    The binary mask you will analyze belongs to one of these classes. Your task is to identify the class of the binary mask and respond with a JSON object in the following format:
    {
        "class_id": <detected_class_id>
    }

    Respond only in JSON format.
    """

    # Encode the image
    base64_image = encode_image(image_path)

    # Make GPT call
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an AI specialized in surgical tool classification."},
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Here is the binary mask image as a base64 string:"},
        {"role": "user", "content": f"data:image/jpeg;base64,{base64_image}"}
    ]
    )

    # Parse response
    json_response = response.choices[0].message.content
    try:
        result = json.loads(json_response)
    except json.JSONDecodeError:
        result = {"class_id": None}

    return result

def process_binary_masks(input_folder, output_json):
    all_results = []
    correct_counts = {class_id: 0 for class_id in range(1, 11)}
    total_counts = {class_id: 0 for class_id in range(1, 11)}

    for class_id in range(1, 11):
        class_folder = os.path.join(input_folder, f"class_{class_id}")

        if not os.path.exists(class_folder):
            print(f"Class folder {class_folder} does not exist, skipping...")
            continue

        for mask_file in os.listdir(class_folder):
            mask_path = os.path.join(class_folder, mask_file)

            if not mask_file.endswith(".png"):
                continue

            total_counts[class_id] += 1

            # Classify the binary mask
            result = classify_binary_mask(mask_path)
            detected_class_id = result.get("class_id")

            # Compare detected class with ground truth
            is_correct = detected_class_id == class_id
            if is_correct:
                correct_counts[class_id] += 1

            all_results.append({
                "mask_file": mask_file,
                "ground_truth": class_id,
                "detected_class": detected_class_id,
                "is_correct": is_correct
            })

    # Calculate accuracy per class
    accuracy_results = {
        class_id: (correct_counts[class_id] / total_counts[class_id]) * 100 if total_counts[class_id] > 0 else 0
        for class_id in range(1, 11)
    }

    # Save results to output JSON
    with open(output_json, "w") as f:
        json.dump({"results": all_results, "accuracy": accuracy_results}, f, indent=2)

    print("Accuracy per class:")
    for class_id, accuracy in accuracy_results.items():
        print(f"Class {class_id}: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify binary masks using GPT.")
    parser.add_argument("--input_folder", required=True, help="Path to the binary_mask folder containing class subfolders.")
    parser.add_argument("--output_json", required=True, help="Path to save the classification results JSON.")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_json = args.output_json

    process_binary_masks(input_folder, output_json)
