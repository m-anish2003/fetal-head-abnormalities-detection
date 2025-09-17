import os, shutil, re

# Paths to your raw dataset
raw_train_dir = r"C:\Users\MANISH\OneDrive\Desktop\fetal-head-project\raw_dataset\training_data"
raw_test_dir  = r"C:\Users\MANISH\OneDrive\Desktop\fetal-head-project\raw_dataset\test_data"

target_train_images = r"data/train/images"
target_train_masks  = r"data/train/masks"
target_test_images  = r"data/test/images"

os.makedirs(target_train_images, exist_ok=True)
os.makedirs(target_train_masks, exist_ok=True)
os.makedirs(target_test_images, exist_ok=True)

# --- Copy Training Data ---
for root, dirs, files in os.walk(raw_train_dir):
    for f in files:
        if f.startswith("."):
            continue
        src_path = os.path.join(root, f)

        if "annotation" in f.lower():
            clean_name = re.sub(r'_?annotation[s]?','', f, flags=re.IGNORECASE)
            dst_path = os.path.join(target_train_masks, clean_name)
        else:
            dst_path = os.path.join(target_train_images, f)

        shutil.copy(src_path, dst_path)

print("Training data copied & renamed successfully!")

# --- Copy Test Data ---
for root, dirs, files in os.walk(raw_test_dir):
    for f in files:
        if f.startswith("."):
            continue
        src_path = os.path.join(root, f)
        dst_path = os.path.join(target_test_images, f)
        shutil.copy(src_path, dst_path)

print("Test data copied successfully!")
