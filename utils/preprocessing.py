"""
Image preprocessing utilities for the wolverines workflow.
Handles standardized resizing and transforms after script 00 determines optimal size.
"""



def get_standard_transform(resize_size=512):
    """
    Get standard transform pipeline with resize and normalization.
    Uses ImageNet normalization constants.
    
    Args:
        resize_size: Target size for resize (width, height)
    
    Returns:
        torchvision.transforms.Compose with Resize + ToTensor + Normalize
    """
    from torchvision import transforms
    from PIL import Image
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_best_resize_size(results_dir='results/00_resize', default=512):
    """
    Get best resize size from script 00 results.
    
    Args:
        results_dir: Directory containing script 00 results
        default: Default resize size if no results found
        
    Returns:
        Best resize size based on validation F1 score
    """
    import json
    import glob
    import os
    
    if not os.path.exists(results_dir):
        print(f"Warning: {results_dir} not found, using default resize_size={default}")
        return default
    
    results_pattern = f"{results_dir}/*.json"
    result_files = glob.glob(results_pattern)
    
    if not result_files:
        print(f"No results in {results_dir}, using default resize_size={default}")
        return default
    
    best_f1 = 0
    best_resize = default
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            f1_score = result.get('final_val_f1', 0)
            resize_size = result['params']['resize_size']
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_resize = resize_size
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    print(f"Best resize size from script 00: {best_resize} (F1: {best_f1:.4f})")
    return best_resize



def get_height_crop_and_resize_transform(height=1280, resize=728):
    """
    Get transform with height center crop, then square resize.
    
    For pelage_base and random approaches:
    1. Center crop height to specified height (1280px), keep full width
    2. Resize to square (728x728)
    
    Args:
        height: Target crop height
        resize: Target square size
    
    Returns:
        torchvision.transforms.Compose with custom height crop + Resize + ToTensor + Normalize
    """
    from torchvision import transforms
    from PIL import Image
    
    print(f"Height crop transform: center crop height to {height}px → resize to {resize}x{resize}")
    
    return transforms.Compose([
        HeightCenterCrop(height),  # Custom transform for height-only center crop
        transforms.Resize((resize, resize), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_center_crop_transform(width=1480, height=1480):
    """
    Get transform with center crop on both width and height, no resizing.
    
    Args:
        width: Target crop width
        height: Target crop height
    
    Returns:
        torchvision.transforms.Compose with center crop + ToTensor + Normalize
    """
    from torchvision import transforms
    
    print(f"Center crop transform: crop to {width}x{height}px (no resize)")
    
    return transforms.Compose([
        CenterCrop2D(width, height),  # Custom transform for width+height center crop
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_center_crop_and_resize_transform(crop_size=1480, resize=256):
    """
    Get transform with center crop followed by resize.
    
    Args:
        crop_size: Target square crop size
        resize: Target square resize size
    
    Returns:
        torchvision.transforms.Compose with center crop + resize + ToTensor + Normalize
    """
    from torchvision import transforms
    from PIL import Image
    
    print(f"Center crop and resize transform: crop to {crop_size}x{crop_size}px → resize to {resize}x{resize}")
    
    return transforms.Compose([
        CenterCrop2D(crop_size, crop_size),  # Custom transform for square center crop
        transforms.Resize((resize, resize), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class HeightCenterCrop:
    """Custom transform to center crop only the height, keeping full width"""
    
    def __init__(self, height):
        self.height = height
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped
            
        Returns:
            PIL Image: Center-cropped image (height only)
        """
        img_width, img_height = img.size
        
        if img_height <= self.height:
            # No cropping needed
            return img
        
        # Calculate crop coordinates (center crop height only)
        top = (img_height - self.height) // 2
        bottom = top + self.height
        
        # Crop: (left, top, right, bottom)
        return img.crop((0, top, img_width, bottom))


class CenterCrop2D:
    """Custom transform to center crop both width and height"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped
            
        Returns:
            PIL Image: Center-cropped image (width and height)
        """
        img_width, img_height = img.size
        
        # Calculate crop coordinates for both dimensions
        left = max(0, (img_width - self.width) // 2)
        right = min(img_width, left + self.width)
        
        top = max(0, (img_height - self.height) // 2)
        bottom = min(img_height, top + self.height)
        
        # Crop: (left, top, right, bottom)
        return img.crop((left, top, right, bottom))


def get_proportional_crop_and_resize_transform(resize=256, top_frac=0.05, bottom_frac=0.05, left_frac=0.25, right_frac=0.25):
    """
    Get transform with proportional crop followed by resize.
    
    Args:
        resize: Target square resize size
        top_frac: Fraction to remove from top (default: 0.05)
        bottom_frac: Fraction to remove from bottom (default: 0.05)
        left_frac: Fraction to remove from left (default: 0.25)
        right_frac: Fraction to remove from right (default: 0.25)
    
    Returns:
        torchvision.transforms.Compose with proportional crop + resize + ToTensor + Normalize
    """
    from torchvision import transforms
    from PIL import Image
    
    print(f"Proportional crop and resize transform: remove {top_frac:.0%} top, {bottom_frac:.0%} bottom, {left_frac:.0%} left, {right_frac:.0%} right → resize to {resize}x{resize}")
    
    return transforms.Compose([
        ProportionalCrop(top_frac=top_frac, bottom_frac=bottom_frac, left_frac=left_frac, right_frac=right_frac),
        transforms.Resize((resize, resize), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class ProportionalCrop:
    """Custom transform to crop based on proportions of image dimensions"""
    
    def __init__(self, top_frac=0.05, bottom_frac=0.05, left_frac=0.25, right_frac=0.25):
        """
        Args:
            top_frac: Fraction to remove from top
            bottom_frac: Fraction to remove from bottom
            left_frac: Fraction to remove from left
            right_frac: Fraction to remove from right
        """
        self.top_frac = top_frac
        self.bottom_frac = bottom_frac
        self.left_frac = left_frac
        self.right_frac = right_frac
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped
            
        Returns:
            PIL Image: Proportionally cropped image
        """
        img_width, img_height = img.size
        
        # Calculate crop coordinates based on proportions
        left = int(img_width * self.left_frac)
        right = int(img_width * (1 - self.right_frac))
        top = int(img_height * self.top_frac)
        bottom = int(img_height * (1 - self.bottom_frac))
        
        # Ensure valid crop dimensions
        left = max(0, left)
        right = min(img_width, right)
        top = max(0, top)
        bottom = min(img_height, bottom)
        
        # Crop: (left, top, right, bottom)
        return img.crop((left, top, right, bottom))


def get_proportional_crop_and_resize_transform(resize=256, top_frac=0.05, bottom_frac=0.05, left_frac=0.25, right_frac=0.25):
    """
    Get transform with proportional crop followed by resize.
    
    Args:
        resize: Target square resize size
        top_frac: Fraction to remove from top (default: 0.05)
        bottom_frac: Fraction to remove from bottom (default: 0.05)
        left_frac: Fraction to remove from left (default: 0.25)
        right_frac: Fraction to remove from right (default: 0.25)
    
    Returns:
        torchvision.transforms.Compose with proportional crop + resize + ToTensor + Normalize
    """
    from torchvision import transforms
    from PIL import Image
    
    print(f"Proportional crop and resize transform: remove {top_frac:.0%} top, {bottom_frac:.0%} bottom, {left_frac:.0%} left, {right_frac:.0%} right → resize to {resize}x{resize}")
    
    return transforms.Compose([
        ProportionalCrop(top_frac=top_frac, bottom_frac=bottom_frac, left_frac=left_frac, right_frac=right_frac),
        transforms.Resize((resize, resize), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])