import os
import random
import shutil
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
NUM_POSITIVE_CASES = 500
NUM_NEGATIVE_CASES = 500
VAL_SPLIT_RATIO = 0.2
# ---------------------

# Paths
base_dir = 'dataset/'
image_dir = os.path.join(base_dir, 'images/')
label_dir = os.path.join(base_dir, 'labels/')
csv_path = 'Train_Labels.csv'

# Create train/val directories
for set_type in ['train', 'val']:
    os.makedirs(os.path.join(image_dir, set_type), exist_ok=True)
    os.makedirs(os.path.join(label_dir, set_type), exist_ok=True)

print(f"Reading {csv_path} to find cases...")
df = pd.read_csv(csv_path)

# Get unique patient IDs for positive (Target=1) and negative (Target=0) cases
positive_ids = df[df['Target'] == 1]['patientId'].unique()
negative_ids = df[df['Target'] == 0]['patientId'].unique()

print(f"Found {len(positive_ids)} total positive cases.")
print(f"Found {len(negative_ids)} total negative cases.")

# Shuffle and take samples
random.shuffle(positive_ids)
random.shuffle(negative_ids)

positive_sample = list(positive_ids[:NUM_POSITIVE_CASES])
negative_sample = list(negative_ids[:NUM_NEGATIVE_CASES])

print(f"Using {len(positive_sample)} positive samples.")
print(f"Using {len(negative_sample)} negative samples.")

# Combine and shuffle the final list
final_file_list = positive_sample + negative_sample
random.shuffle(final_file_list)

# Split into training and validation
split_idx = int(len(final_file_list) * (1 - VAL_SPLIT_RATIO))
train_files = final_file_list[:split_idx]
val_files = final_file_list[split_idx:]

print(f"Total images in new sample: {len(final_file_list)}")
print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")

def move_files(file_list, set_type):
    for filename in tqdm(file_list, desc=f"Moving {set_type} files"):
        img_name = f"{filename}.png"
        label_name = f"{filename}.txt"
        
        # Move image
        src_img_path = os.path.join(image_dir, img_name)
        if os.path.exists(src_img_path):
            shutil.move(src_img_path, os.path.join(image_dir, set_type, img_name))
        
        # Move label if it exists (only positive cases will have one)
        src_label_path = os.path.join(label_dir, label_name)
        if os.path.exists(src_label_path):
            shutil.move(src_label_path, os.path.join(label_dir, set_type, label_name))

# Move the files
move_files(train_files, 'train')
move_files(val_files, 'val')

print("\nSmall dataset split complete!")