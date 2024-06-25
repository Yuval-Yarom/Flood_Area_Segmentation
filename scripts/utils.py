import os
import pandas as pd

IMAGES_DIR = '../data/raw/images'
MASKS_DIR = '../data/raw/masks'
METADATA_PATH = '../Data/raw/metadata.csv'
AUGMENTED_METADATA_PATH = '../data/processed/augmentation/augmented_metadata.csv'
def get_processed_images_and_masks():
    metadata = pd.read_csv(METADATA_PATH)
    images = [os.path.join(IMAGES_DIR, img) for img in metadata['Image']]
    masks = [os.path.join(MASKS_DIR, mask) for mask in metadata['Mask']]
    return images, masks

def get_augmented_images_and_masks():
    augmented_metadata = pd.read_csv(AUGMENTED_METADATA_PATH)
    images = [os.path.join('data/processed/augmentation', img) for img in augmented_metadata['Image']]
    masks = [os.path.join('data/processed/augmentation', mask) for mask in augmented_metadata['Mask']]
    return images, masks


processed_images, processed_masks = get_processed_images_and_masks()
print("Processed images:", processed_images)
print("Processed masks:", processed_masks)

augmented_images, augmented_masks = get_augmented_images_and_masks()
print("Augmented images:", augmented_images)
print("Augmented masks:", augmented_masks)