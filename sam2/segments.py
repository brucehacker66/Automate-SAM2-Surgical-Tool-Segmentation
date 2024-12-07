import os
import sys

def extract_phases(txt_file_path):
    phases_dict = {}

    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines[1:]:
        frame_index, phase = line.strip().split()
        frame_index = int(frame_index)

        if frame_index % 25 == 0:
            subsample_index = frame_index // 25
            if phase not in phases_dict:
                phases_dict[phase] = []
            phases_dict[phase].append(frame_index)

    return phases_dict

# output a dict with phase as the key and a list of segments as the value, len(segments) <= 300
def segment_by_phases(folder_path, txt_file_path):

    # Extract phases, key will be the phase and value will be a list of indices
    phases_dict = extract_phases(txt_file_path)

    # Store phase-images segments
    segments_dict = {phase: [[]] for phase in phases_dict}

    jpg_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

    # Map the phase dict to the png files
    for phase, indices in phases_dict.items():
        for index in indices:
            jpg_file = f"{index:07d}.jpg"
            if jpg_file in jpg_files:
                if len(segments_dict[phase][-1]) >= 300:
                    segments_dict[phase].append([])
                segments_dict[phase][-1].append(jpg_file)

    return segments_dict

def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python segments.py <input_path> <txt_file_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    txt_file_path = sys.argv[2]

    original_images_path = os.path.join(input_path, 'ws_0', 'images')
    segments_dict = segment_by_phases(original_images_path, txt_file_path)
    
    ws_idx = 1
    # create directory to store the segments as images
    for phase, segment_lists in segments_dict.items():
        print(phase, len(segment_lists))
        for segment in segment_lists:
            os.makedirs(os.path.join(input_path, f'ws_{ws_idx}'), exist_ok=True)
            segment_dir = os.path.join(input_path, f'ws_{ws_idx}', 'images')
            os.makedirs(segment_dir, exist_ok=True)
            for image in segment:
                image_path = os.path.join(original_images_path, image)
                os.system(f"cp {image_path} {segment_dir}")
            print(f"Segment {ws_idx} created for phase {phase}",'\n')
            ws_idx += 1
        


if __name__ == "__main__":
    main()