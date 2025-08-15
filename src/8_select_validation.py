import shared_utils
import os 
from pathlib import Path 
from collections import Counter, defaultdict
import re

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
    assert os.path.exists(folder)
    files = shared_utils.find_files(Path(folder), '.midi')
    assert len(files) > 0, f"Failed to find midi in {folder}"
    print(f"Found {len(files)} midi files")
    files.sort(key=lambda x: os.path.basename(x))
    
    # Create mapping of cleaned names to list of original paths
    name_to_paths = defaultdict(list)
    for file in files:
        clean_name = get_base_name(file)
        name_to_paths[clean_name].append(file)
    
    # Print counts and original paths
    for clean_name, paths in sorted(name_to_paths.items()):
        print(f"{clean_name}: {len(paths)}")
        # for path in paths:
        #     print(f"  → {path}")

if __name__ == "__main__":
    main()