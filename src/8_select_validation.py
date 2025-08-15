import shared_utils
import os 
from pathlib import Path 
from collections import Counter, defaultdict
import re
import sys
import shutil  # Add this import

def get_base_name(filepath):
    basename = os.path.basename(filepath)
    # Split on ' - ' first if it exists, take first part
    name = basename.split(' - ')[0]
    # Remove .midi extension if present
    name = name.replace('.midi', '')
    # Convert to lowercase
    name = name.lower()
    # Keep only alphanumeric and spaces
    name = re.sub(r'[^a-z0-9 ]', '', name)
    return name

def main():
    folder = "/Users/matthewyk/src/ai-music/data/midi/"
    out_folder = "/Users/matthewyk/src/ai-music/data/10_piano_solos/"
    assert os.path.exists(folder)
    assert os.path.exists(out_folder)

    files = shared_utils.find_files(Path(folder), '.midi')
    assert len(files) > 0, f"Failed to find midi in {folder}"
    print(f"Found {len(files)} midi files")
    files.sort(key=lambda x: os.path.basename(x))
    
    # Create mapping of cleaned names to list of original paths
    name_to_paths = defaultdict(list)
    for file in files:
        clean_name = get_base_name(file)
        name_to_paths[clean_name].append(file)
    
    # Print counts and original paths, sorted by count descending
    sorted_items = sorted(name_to_paths.items(), key=lambda x: len(x[1]), reverse=False)
    for clean_name, paths in sorted_items[-20:]:
        print(f"{clean_name}")
        
        # Sort paths by file size and take the smallest one
        src_path = sorted(paths, key=lambda x: os.path.getsize(x))[0]
        dst_path = os.path.join(out_folder, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        print(f"Copied {src_path} ({os.path.getsize(src_path)} bytes) to {dst_path}")

if __name__ == "__main__":
    main()