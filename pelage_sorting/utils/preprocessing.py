"""
Image preprocessing utilities for the wolverines workflow.
Handles standardized resizing and transforms after script 00 determines optimal size.
"""

def preprocess_dataset(dataset, resize_size):
    """
    Preprocess HuggingFace dataset by resizing images.
    
    Args:
        dataset: HuggingFace dataset with 'image' field
        resize_size: Target size for resize (width, height)
        
    Returns:
        Preprocessed dataset with resized images
    """
    def preprocess_batch(batch):
        """Preprocess images: resize only"""
        from PIL import Image
        images = [img.resize((resize_size, resize_size), Image.LANCZOS) for img in batch['image']]
        batch['image'] = images
        return batch
    
    return dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=32,
        num_proc=4,
        desc=f"Resizing images to {resize_size}x{resize_size}"
    )


def get_standard_transform():
    """
    Get standard normalization transform for preprocessed images.
    Uses ImageNet normalization constants.
    
    Returns:
        torchvision.transforms.Compose with ToTensor + Normalize
    """
    from torchvision import transforms
    return transforms.Compose([
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