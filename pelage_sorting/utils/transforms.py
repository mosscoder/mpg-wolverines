import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image, ImageFilter
import random
from typing import Dict, Any, Optional


class CenterCrop(transforms.CenterCrop):
    """Custom center crop that handles various input types"""
    
    def __init__(self, size):
        super().__init__(size)
        
    def forward(self, img):
        if isinstance(img, Image.Image):
            return super().forward(img)
        # Handle other PIL modes or numpy arrays
        return super().forward(img)


class GrayscaleAugmentation:
    """Convert image to grayscale with specified probability"""
    
    def __init__(self, probability: float = 0.5):
        self.probability = probability
    
    def __call__(self, img):
        if random.random() > self.probability:
            return img
        
        # Convert to grayscale but keep 3 channels
        gray = img.convert('L')
        return Image.merge('RGB', (gray, gray, gray))


class AdaptiveBlur:
    """Apply blur with different intensities"""
    
    def __init__(self, blur_type: str = 'moderate', probability: float = 0.5):
        self.blur_type = blur_type
        self.probability = probability
        
        self.blur_params = {
            'none': 0,
            'moderate': 1.0,
            'high': 2.0
        }
    
    def __call__(self, img):
        if random.random() > self.probability or self.blur_type == 'none':
            return img
        
        radius = self.blur_params[self.blur_type]
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class CutMix:
    """CutMix augmentation for images"""
    
    def __init__(self, alpha: float = 1.0, probability: float = 0.5):
        self.alpha = alpha
        self.probability = probability
    
    def __call__(self, img):
        if random.random() > self.probability:
            return img
        
        # For single image, create a random patch removal
        w, h = img.size
        cut_w = int(w * np.random.beta(self.alpha, self.alpha) * 0.5)  # Smaller cuts
        cut_h = int(h * np.random.beta(self.alpha, self.alpha) * 0.5)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Fill with random noise or mean color
        img_array = np.array(img)
        mean_color = np.mean(img_array, axis=(0, 1))
        img_array[y1:y2, x1:x2] = mean_color
        
        return Image.fromarray(img_array.astype(np.uint8))


def create_base_transforms(crop_size: int, resize_size: int = 256):
    """Create base transforms with center crop and resize"""
    return transforms.Compose([
        CenterCrop(crop_size),
        transforms.Resize((resize_size, resize_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_augmented_transforms(
    crop_size: int,
    resize_size: int = 256,
    max_zoom: float = 1.0,
    h_flip_p: float = 0.0,
    grayscale_p: float = 0.0,
    blur_type: str = 'none',
    blur_p: float = 0.0,
    cutmix_p: float = 0.0
):
    """Create augmented transforms based on hyperparameters"""
    
    transform_list = []
    
    # Center crop first
    transform_list.append(CenterCrop(crop_size))
    
    # Random zoom (implemented as random resized crop)
    if max_zoom > 1.0:
        zoom_size = int(crop_size * max_zoom)
        transform_list.append(transforms.RandomResizedCrop(
            crop_size, 
            scale=(1.0/max_zoom**2, 1.0),
            ratio=(0.9, 1.1)
        ))
    
    # Horizontal flip
    if h_flip_p > 0:
        transform_list.append(transforms.RandomHorizontalFlip(p=h_flip_p))
    
    # Grayscale augmentation
    if grayscale_p > 0:
        transform_list.append(GrayscaleAugmentation(probability=grayscale_p))
    
    # Blur
    if blur_p > 0:
        transform_list.append(AdaptiveBlur(blur_type=blur_type, probability=blur_p))
    
    # CutMix
    if cutmix_p > 0:
        transform_list.append(CutMix(probability=cutmix_p))
    
    # Resize to final size
    transform_list.append(transforms.Resize((resize_size, resize_size), antialias=True))
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)


def get_validation_transforms(crop_size: int, resize_size: int = 256):
    """Get validation transforms (no augmentation)"""
    return create_base_transforms(crop_size, resize_size)


def create_transform_from_params(params: Dict[str, Any], is_train: bool = True):
    """Create transform from parameter dictionary"""
    
    crop_size = params.get('crop_size', 512)
    resize_size = params.get('resize_size', 256)
    
    if not is_train:
        return get_validation_transforms(crop_size, resize_size)
    
    # Training transforms with augmentation parameters
    return create_augmented_transforms(
        crop_size=crop_size,
        resize_size=resize_size,
        max_zoom=params.get('max_zoom', 1.0),
        h_flip_p=params.get('h_flip_p', 0.0),
        grayscale_p=params.get('grayscale_p', 0.0),
        blur_type=params.get('blur_type', 'none'),
        blur_p=params.get('blur_p', 0.0),
        cutmix_p=params.get('cutmix_p', 0.0)
    )