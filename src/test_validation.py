import shared_utils
from pathlib import Path


# files = shared_utils.find_text_files(Path('data/tiny_dataset/training/'))
files = shared_utils.find_text_files(Path('/Users/matthewyk/src/ai-music/data/small-lm-datasets/midi_hawthorne_all_keys/validation'))
print(f"Retrieved {len(files)} files.")

# files = ["val/file1.txt",     "val/file2.txt"]
ds = shared_utils.MiddleP2WindowDataset(files, min_p2=4, max_p2=256)
print(f"Retrieved {len(ds)} data points.")
print(f"Token counts : {[len(d) for d in ds]}")
# print(f"HEre is the first one,{ds[0]}")

      