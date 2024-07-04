import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt


def load_metadata(metadata_path):
    return pd.read_csv(metadata_path)


def preprocess_image(image_path, target_size=(256, 256)):
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])
        image = transform(image)
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def preprocess_mask(mask_path, target_size=(256, 256)):
    try:
        mask = Image.open(mask_path).convert('L')
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])
        mask = transform(mask)
        return mask
    except Exception as e:
        print(f"Error processing mask {mask_path}: {e}")
        return None


def plot_histogram(image, title, save_path):
    plt.figure()
    if image.shape[0] == 3:  # RGB image
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = torch.histc(image[i], bins=256, min=0, max=1)
            plt.plot(hist.cpu().numpy(), color=color)
    else:  # Grayscale image
        hist = torch.histc(image, bins=256, min=0, max=1)
        plt.plot(hist.cpu().numpy(), color='black')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()


def augment_translate(image, mask):
    try:
        transform = transforms.Compose([
            transforms.Pad((20, 20), fill=0),
            transforms.CenterCrop((image.size(1), image.size(2)))
        ])
        translated_image = transform(image)
        translated_mask = transform(mask)
        return translated_image, translated_mask
    except Exception as e:
        print(f"Error in augment_translate: {e}")
        return image, mask


def augment_scale(image, mask):
    try:
        scale_factor = 1.2
        new_size = [int(dim * scale_factor) for dim in image.shape[1:]]
        transform = transforms.Compose([
            transforms.Resize(new_size, interpolation=Image.BILINEAR),
            transforms.CenterCrop((image.size(1), image.size(2)))
        ])
        scaled_image = transform(image)
        scaled_mask = transform(mask)
        return scaled_image, scaled_mask
    except Exception as e:
        print(f"Error in augment_scale: {e}")
        return image, mask


def augment_flip(image, mask):
    try:
        transform = transforms.RandomHorizontalFlip(p=1.0)
        flipped_image = transform(image)
        flipped_mask = transform(mask)
        return flipped_image, flipped_mask
    except Exception as e:
        print(f"Error in augment_flip: {e}")
        return image, mask


def augment_rotate(image, mask):
    try:
        transform_image = transforms.Compose([
            transforms.RandomRotation(degrees=45, fill=0),
            transforms.CenterCrop((image.size(1), image.size(2)))
        ])
        transform_mask = transforms.Compose([
            transforms.RandomRotation(degrees=45, fill=0),
            transforms.CenterCrop((mask.size(1), mask.size(2)))
        ])
        rotated_image = transform_image(image)
        rotated_mask = transform_mask(mask)
        return rotated_image, rotated_mask
    except Exception as e:
        print(f"Error in augment_rotate: {e}")
        return image, mask


def augment_promotion(image, mask):
    try:
        transform = transforms.Compose([
            transforms.Pad((20, 20), fill=0),
            transforms.CenterCrop((image.size(1), image.size(2)))
        ])
        promoted_image = transform(image)
        promoted_mask = transform(mask)
        return promoted_image, promoted_mask
    except Exception as e:
        print(f"Error in augment_promotion: {e}")
        return image, mask


def save_augmented_data(images_dir, masks_dir, test_metadata):
    augmentation_funcs = {
        'translate': augment_translate,
        'scale': augment_scale,
        'flip': augment_flip,
        'rotate': augment_rotate,
        'promotion': augment_promotion
    }
    augmented_metadata = []
    base_path = Path('../data/processed/augmentation')
    base_path.mkdir(parents=True, exist_ok=True)

    for aug_name, aug_func in augmentation_funcs.items():
        aug_images_dir = base_path / aug_name / 'images'
        aug_masks_dir = base_path / aug_name / 'masks'
        aug_images_dir.mkdir(parents=True, exist_ok=True)
        aug_masks_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in test_metadata.iterrows():
            try:
                image_path = os.path.join(images_dir, row['Image'])
                mask_path = os.path.join(masks_dir, row['Mask'])
                image = preprocess_image(image_path)
                mask = preprocess_mask(mask_path)
                if image is None or mask is None:
                    continue
                augmented_image, augmented_mask = aug_func(image, mask)
                aug_image_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{aug_name}.png"
                aug_mask_filename = f"{os.path.splitext(os.path.basename(mask_path))[0]}_{aug_name}.png"
                aug_image_path = aug_images_dir / aug_image_filename
                aug_mask_path = aug_masks_dir / aug_mask_filename
                transforms.ToPILImage()(augmented_image).save(aug_image_path)
                transforms.ToPILImage()(augmented_mask).save(aug_mask_path)
                plot_histogram(augmented_image, f'{aug_name} Image Histogram',
                               aug_images_dir / f"{os.path.splitext(aug_image_filename)[0]}_hist.png")
                plot_histogram(augmented_mask, f'{aug_name} Mask Histogram',
                               aug_masks_dir / f"{os.path.splitext(aug_mask_filename)[0]}_hist.png")
                augmented_metadata.append({
                    'Image': str(aug_image_path.relative_to(base_path)),
                    'Mask': str(aug_mask_path.relative_to(base_path))
                })
                print(f"Saved augmented {aug_name} for image {idx + 1}/{len(test_metadata)}")
            except Exception as e:
                print(f"Error in save_augmented_data for image {row['Image']}: {e}")

    # Save new metadata to CSV
    augmented_metadata_df = pd.DataFrame(augmented_metadata)
    augmented_metadata_df.to_csv(base_path / 'augmented_metadata.csv', index=False)


def preprocess_data(metadata_path, images_dir, masks_dir, target_size=(256, 256)):
    metadata = load_metadata(metadata_path)
    print("Starting preprocessing...")
    images = []
    masks = []
    for idx, row in metadata.iterrows():
        try:
            image_path = os.path.join(images_dir, row['Image'])
            mask_path = os.path.join(masks_dir, row['Mask'])
            image = preprocess_image(image_path, target_size)
            mask = preprocess_mask(mask_path, target_size)
            if image is not None and mask is not None:
                images.append(image)
                masks.append(mask)
            print(f"Processed image {idx + 1}/{len(metadata)}")
        except Exception as e:
            print(f"Error processing data for image {row['Image']}: {e}")
    return images, masks, metadata


if __name__ == "__main__":
    metadata_path = '../data/raw/metadata.csv'
    images_dir = '../data/raw/images'
    masks_dir = '../data/raw/masks'
    images, masks, metadata = preprocess_data(metadata_path, images_dir, masks_dir)
    X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    Path('../data/processed').mkdir(parents=True, exist_ok=True)
    save_augmented_data(images_dir, masks_dir, metadata)
