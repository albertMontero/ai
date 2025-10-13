import os
from PIL import Image
import imagehash

HASH_FUNC = imagehash.phash  # options: phash, dhash, whash, ahash
HASH_SIZE = 16              # larger = more sensitive (default: 8)
MAX_DISTANCE = 5            # smaller = stricter duplicate check

def collect_images(root):
    """Recursively collect all image file paths."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                yield os.path.join(dirpath, name)

def compute_hashes(paths):
    """Compute perceptual hashes for a list of image paths."""
    hashes = {}
    for path in paths:
        try:
            img = Image.open(path)
            h = HASH_FUNC(img, hash_size=HASH_SIZE)
            hashes[path] = h
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return hashes

def find_duplicates(hashes):
    paths = list(hashes.keys())
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            dist = hashes[paths[i]] - hashes[paths[j]]
            if dist <= MAX_DISTANCE:
                print(f"Possible duplicate (distance {dist}):")
                print(f"  {paths[i]}")
                print(f"  {paths[j]}")


if __name__ == "__main__":

    ROOT_FOLDER = "../data/raw"

    print(os.getcwd())

    print(f"Scanning folder: {ROOT_FOLDER}")
    img_paths = list(collect_images(ROOT_FOLDER))
    print(f"Found {len(img_paths)} image(s). Computing hashes...")
    hashes = compute_hashes(img_paths)
    print("Checking for duplicates...")
    find_duplicates(hashes)
    print("Done.")
