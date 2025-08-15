import pandas as pd
import random
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
import os
import multiprocessing as mp

# --- Configuration ---
NUM_CATEGORIES = 1000
IMAGES_PER_CATEGORY = 2
IMAGE_SIZE = 300
OUTPUT_DIR = r"/Volumes/AHRDrive/TyPictures/data"
# --- End Configuration ---

# STEP 1: Get the official class list
url = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
print(f"Downloading official class list from {url}...")
df = pd.read_csv(url)
all_classes = df['DisplayName'].tolist()
print(f"Successfully loaded {len(all_classes)} class names.")
chosen_classes = random.sample(all_classes, NUM_CATEGORIES)
print(f"Randomly selected {len(chosen_classes)} categories to sample from...")

# STEP 2: Download the data efficiently, IGNORING SEGMENTATIONS
# Do NOT delete the output directory — just create it if missing
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
print(f"Output directory '{OUTPUT_DIR}' is ready.")

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],  # <-- critical fix
    classes=chosen_classes,
    max_samples_per_class=IMAGES_PER_CATEGORY,
    only_matching=True,
    dataset_name="dissimilar_image_set_final_fast"
)
print(f"\nSuccessfully downloaded {len(dataset)} images.")

# STEP 3: Process and save images in parallel
def process_image(args):
    index, filepath = args
    try:
        img = Image.open(filepath)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        width, height = img.size
        left = (width - IMAGE_SIZE) / 2
        top = (height - IMAGE_SIZE) / 2
        right = (width + IMAGE_SIZE) / 2
        bottom = (height + IMAGE_SIZE) / 2

        cropped_img = img.crop((left, top, right, bottom))
        output_path = os.path.join(OUTPUT_DIR, f"{index + 1}.jpg")
        cropped_img.save(output_path, "JPEG")

        # Progress update every 100 images
        if (index + 1) % 100 == 0:
            print(f"Saved {index + 1} images so far...")

        return None
    except Exception as e:
        return f"Could not process file {filepath}: {e}"

filepaths = dataset.values("filepath")
args_list = list(enumerate(filepaths))
num_cores = mp.cpu_count()
print(f"\nStarting parallel processing with {num_cores} cores...")

with mp.Pool(processes=num_cores) as pool:
    results = pool.map(process_image, args_list)

errors = [r for r in results if r is not None]
if errors:
    print("\nSome files could not be processed:")
    for error in errors:
        print(error)

print(f"\n✅ Done! Processed and saved images to '{OUTPUT_DIR}'.")

# Clean up the FiftyOne dataset
fo.delete_dataset("dissimilar_image_set_final_fast")
print("Cleaned up FiftyOne dataset.")
