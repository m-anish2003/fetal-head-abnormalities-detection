import os
import glob

train_images = glob.glob("data/train/images/*")
train_masks = glob.glob("data/train/masks/*")

image_filenames = {os.path.basename(path) for path in train_images}
mask_filenames = {os.path.basename(path) for path in train_masks}

missing_masks = image_filenames - mask_filenames
missing_images = mask_filenames - image_filenames

print(f"Total training images: {len(image_filenames)}")
print(f"Total training masks: {len(mask_filenames)}")

if missing_masks:
    print("Images missing masks (up to 10 shown):")
    for m in list(missing_masks)[:10]:
        print(" -", m)
else:
    print("All images have corresponding masks!")

if missing_images:
    print("Masks missing images (up to 10 shown):")
    for m in list(missing_images)[:10]:
        print(" -", m)
else:
    print("No masks are orphaned.")

