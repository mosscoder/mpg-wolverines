"""
Image preprocessing utilities for the wolverines workflow.
Handles standardized resizing and transforms after script 00 determines optimal size.
"""



def get_standard_transform(resize_size=224):
    """
    Get standard transform pipeline with resize and normalization.
    Uses ImageNet normalization constants.
    Fixed resize at 224x224 for DINOv3 efficiency.

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
