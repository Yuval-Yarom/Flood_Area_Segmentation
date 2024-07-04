import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from scripts.preprocess import augment_translate, augment_scale, augment_flip, augment_rotate, augment_promotion, plot_histogram

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def load_mask(mask_path):
    return Image.open(mask_path).convert('L')

def save_image(image, path):
    image.save(path)

def save_images(original, augmented, base_filename, aug_name):
    save_dir = 'test/examples'
    os.makedirs(save_dir, exist_ok=True)

    original_image_path = os.path.join(save_dir, f'{base_filename}_before.png')
    augmented_image_path = os.path.join(save_dir, f'{base_filename}_{aug_name}.png')
    original_hist_path = os.path.join(save_dir, f'{base_filename}_before_hist.png')
    augmented_hist_path = os.path.join(save_dir, f'{base_filename}_{aug_name}_hist.png')

    save_image(original, original_image_path)
    save_image(augmented, augmented_image_path)

    plot_histogram(torch.tensor(np.array(original)), f'{base_filename} Before Histogram', original_hist_path)
    plot_histogram(torch.tensor(np.array(augmented)), f'{base_filename} {aug_name} Histogram', augmented_hist_path)

def pil_to_tensor(image):
    transform = transforms.ToTensor()
    return transform(image)

def tensor_to_pil(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor)

# Paths to an example image and mask
image_path = '../Data/raw/images/0.jpg'
mask_path = '../Data/raw/masks/0.png'

# Load the original image and mask
original_image = load_image(image_path)
original_mask = load_mask(mask_path)

# Convert to tensors
original_image_tensor = pil_to_tensor(original_image)
original_mask_tensor = pil_to_tensor(original_mask)

# Save the original image and mask once
save_images(original_image, original_image, 'image', 'before')
save_images(original_mask, original_mask, 'mask', 'before')

# Apply augmentations and save results
translated_image_tensor, translated_mask_tensor = augment_translate(original_image_tensor, original_mask_tensor)
scaled_image_tensor, scaled_mask_tensor = augment_scale(original_image_tensor, original_mask_tensor)
flipped_image_tensor, flipped_mask_tensor = augment_flip(original_image_tensor, original_mask_tensor)
rotated_image_tensor, rotated_mask_tensor = augment_rotate(original_image_tensor, original_mask_tensor)
promoted_image_tensor, promoted_mask_tensor = augment_promotion(original_image_tensor, original_mask_tensor)

# Convert tensors back to PIL images
translated_image = tensor_to_pil(translated_image_tensor)
translated_mask = tensor_to_pil(translated_mask_tensor)

scaled_image = tensor_to_pil(scaled_image_tensor)
scaled_mask = tensor_to_pil(scaled_mask_tensor)

flipped_image = tensor_to_pil(flipped_image_tensor)
flipped_mask = tensor_to_pil(flipped_mask_tensor)

rotated_image = tensor_to_pil(rotated_image_tensor)
rotated_mask = tensor_to_pil(rotated_mask_tensor)

promoted_image = tensor_to_pil(promoted_image_tensor)
promoted_mask = tensor_to_pil(promoted_mask_tensor)

# Save augmented images and masks
save_images(original_image, translated_image, 'image', 'translated')
save_images(original_mask, translated_mask, 'mask', 'translated')

save_images(original_image, scaled_image, 'image', 'scaled')
save_images(original_mask, scaled_mask, 'mask', 'scaled')

save_images(original_image, flipped_image, 'image', 'flipped')
save_images(original_mask, flipped_mask, 'mask', 'flipped')

save_images(original_image, rotated_image, 'image', 'rotated')
save_images(original_mask, rotated_mask, 'mask', 'rotated')

save_images(original_image, promoted_image, 'image', 'promoted')
save_images(original_mask, promoted_mask, 'mask', 'promoted')
