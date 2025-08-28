"""
Feature visualization utilities for DINOv3 and individual ID models.
Functions for extracting, processing, and visualizing important features.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os


def extract_top_features(linear_weights, top_k=3):
    """Extract top K most important features globally across all classes
    
    Args:
        linear_weights: Linear layer weights tensor [num_classes, num_features]
        top_k: Number of top features to extract
        
    Returns:
        Tuple of (top_feature_indices, top_feature_values)
    """
    # Calculate global importance: max absolute weight across all classes
    global_importance = torch.max(torch.abs(linear_weights), dim=0)[0]  # Shape: [num_features]
    
    # Get top K feature indices
    top_feature_indices = torch.argsort(global_importance, descending=True)[:top_k]
    top_feature_values = global_importance[top_feature_indices]
    
    print(f"Top {top_k} most important features:")
    for i, (idx, val) in enumerate(zip(top_feature_indices, top_feature_values)):
        print(f"  {i+1}. Feature {idx.item()}: importance {val.item():.4f}")
    
    return top_feature_indices.cpu().numpy(), top_feature_values.cpu().numpy()


def extract_dinov3_patch_features(model, image_tensor, device, expected_patches=None):
    """Extract DINOv3 patch features for visualization
    
    Args:
        model: DINOv3 model with backbone attribute
        image_tensor: Input image tensor [1, 3, H, W]
        device: Device to run inference on
        expected_patches: Expected number of patches (for validation)
        
    Returns:
        Numpy array of patch features [num_patches, num_features]
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Get features from backbone
        backbone_output = model.backbone(image_tensor)
        all_features = backbone_output.last_hidden_state  # [1, num_tokens, num_features]
        
        print(f"DINOv3 output shape: {all_features.shape}")
        
        # Calculate expected patches if not provided
        if expected_patches is None:
            # For square images with 16x16 patches
            img_size = image_tensor.shape[-1]  # Assume square
            expected_patches = (img_size // 16) * (img_size // 16)
        
        num_tokens = all_features.shape[1]
        extra_tokens = num_tokens - expected_patches
        
        print(f"Expected patches: {expected_patches}, Got tokens: {num_tokens}, Extra tokens: {extra_tokens}")
        
        # Extract patch tokens only (skip CLS/register tokens at beginning)
        if extra_tokens > 0:
            patch_features = all_features[:, extra_tokens:, :]  # [1, num_patches, num_features]
        else:
            patch_features = all_features
        
        print(f"Patch features shape: {patch_features.shape}")
        
        return patch_features[0].cpu().numpy()  # [num_patches, num_features]


def create_spatial_feature_maps(patch_features, feature_indices, patch_grid_size, target_size=None, original_size=None):
    """Create spatial visualization of feature activations
    
    Args:
        patch_features: Patch features array [num_patches, num_features]  
        feature_indices: Indices of features to visualize
        patch_grid_size: Tuple of (height, width) for patch grid (e.g., (45, 45))
        target_size: Target size for upsampled visualizations (for backward compatibility)
        original_size: Original image size to maintain aspect ratio (height, width)
        
    Returns:
        List of upsampled feature maps
    """
    feature_maps = []
    
    # Use original size if provided, otherwise fall back to target_size
    if original_size is not None:
        upsample_size = original_size
    else:
        upsample_size = target_size or (728, 728)
    
    for feature_idx in feature_indices:
        # Get feature activations for all patches
        feature_activations = patch_features[:, feature_idx]  # [num_patches]
        
        # Reshape to spatial grid
        feature_map = feature_activations.reshape(patch_grid_size)  # [grid_h, grid_w]
        
        # Convert to tensor for upsampling
        feature_tensor = torch.from_numpy(feature_map).float().unsqueeze(0).unsqueeze(0)  # [1, 1, grid_h, grid_w]
        
        # Upsample to target size maintaining aspect ratio
        upsampled = F.interpolate(feature_tensor, size=upsample_size, mode='bilinear', align_corners=False)
        upsampled_map = upsampled[0, 0].numpy()  # [target_h, target_w]
        
        feature_maps.append(upsampled_map)
    
    return feature_maps


def create_rgb_composite(feature_maps):
    """Create RGB composite from 3 feature maps
    
    Args:
        feature_maps: List of 3 normalized feature maps [H, W]
        
    Returns:
        RGB composite array [H, W, 3]
    """
    if len(feature_maps) != 3:
        raise ValueError("Need exactly 3 feature maps for RGB composite")
    
    # Normalize each feature map independently
    normalized_features = []
    for feature_map in feature_maps:
        normalized = normalize_feature_map(feature_map, method='percentile')
        normalized_features.append(normalized)
    
    # Stack as RGB channels
    rgb_composite = np.stack(normalized_features, axis=-1)  # [H, W, 3]
    
    return rgb_composite


def normalize_feature_map(feature_map, method='percentile', percentile_range=(5, 95)):
    """Normalize feature map for visualization
    
    Args:
        feature_map: 2D numpy array of feature activations
        method: Normalization method ('percentile', 'minmax', 'zscore')
        percentile_range: Percentile range for clipping (for 'percentile' method)
        
    Returns:
        Normalized feature map in range [0, 1]
    """
    if method == 'percentile':
        p_low, p_high = np.percentile(feature_map, percentile_range)
        if p_high - p_low > 1e-7:
            normalized = np.clip((feature_map - p_low) / (p_high - p_low), 0, 1)
        else:
            normalized = np.zeros_like(feature_map)
    
    elif method == 'minmax':
        min_val, max_val = feature_map.min(), feature_map.max()
        if max_val - min_val > 1e-7:
            normalized = (feature_map - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(feature_map)
    
    elif method == 'zscore':
        mean_val, std_val = feature_map.mean(), feature_map.std()
        if std_val > 1e-7:
            zscore = (feature_map - mean_val) / std_val
            # Convert to 0-1 range using sigmoid
            normalized = 1 / (1 + np.exp(-zscore))
        else:
            normalized = np.full_like(feature_map, 0.5)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def prepare_image_for_display(image_tensor, denormalize=True):
    """Prepare tensor image for matplotlib display
    
    Args:
        image_tensor: Image tensor [3, H, W] or [1, 3, H, W]
        denormalize: Whether to denormalize from ImageNet stats
        
    Returns:
        Numpy array [H, W, 3] ready for imshow
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]  # Remove batch dimension
    
    # Convert to numpy
    if isinstance(image_tensor, torch.Tensor):
        display_array = image_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        display_array = np.transpose(image_tensor, (1, 2, 0))
    
    # Denormalize if requested (assume ImageNet normalization)
    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        display_array = display_array * std + mean
    
    # Clip to valid range
    display_array = np.clip(display_array, 0, 1)
    
    return display_array


def create_feature_visualization_grid(selected_images, individual_ids, top_feature_indices, 
                                    model, transform, device, output_path,
                                    patch_grid_size=(45, 45), target_size=(728, 728),
                                    figsize=(20, 4), colormap='viridis', maintain_aspect_ratio=True):
    """Create comprehensive feature visualization grid
    
    Args:
        selected_images: Dict mapping individual_id -> image info
        individual_ids: List of individual IDs (determines row order)
        top_feature_indices: Indices of top features to visualize
        model: DINOv3 model for feature extraction
        transform: Image transform function
        device: Device for inference
        output_path: Path to save visualization
        patch_grid_size: Grid size for patches (height, width)
        target_size: Target size for upsampled visualizations (fallback)
        figsize: Figure size per row
        colormap: Matplotlib colormap for feature visualizations
        maintain_aspect_ratio: Whether to preserve original image aspect ratio
    """
    n_individuals = len([ind for ind in individual_ids if ind in selected_images])
    n_cols = 5  # RGB + 3 features + RGB composite
    
    # Adjust figure size based on number of individuals
    fig_height = figsize[1] * n_individuals
    fig = plt.figure(figsize=(figsize[0], fig_height))
    gs = gridspec.GridSpec(n_individuals, n_cols, figure=fig, hspace=0.3, wspace=0.1)
    
    row = 0
    for ind_id in individual_ids:
        
        if ind_id not in selected_images:
            print(f"Skipping {ind_id} - no test image")
            continue
        
        image_info = selected_images[ind_id]
        pil_image = image_info['image']
        
        print(f"Processing {ind_id}...")
        
        # Get original image dimensions for aspect ratio preservation
        original_width, original_height = pil_image.size
        
        # Apply transform and extract features
        image_tensor = transform(pil_image).unsqueeze(0)  # [1, 3, H, W]
        patch_features = extract_dinov3_patch_features(model, image_tensor, device)
        
        # Determine upsampling size
        if maintain_aspect_ratio:
            # Use original aspect ratio
            upsample_size = (original_height, original_width)
        else:
            upsample_size = target_size
        
        # Create feature visualizations
        feature_maps = create_spatial_feature_maps(patch_features, top_feature_indices, 
                                                 patch_grid_size, target_size=target_size, 
                                                 original_size=upsample_size)
        
        # Prepare RGB display
        display_image = prepare_image_for_display(transform(pil_image))
        
        # Column 0: RGB Image
        ax_rgb = fig.add_subplot(gs[row, 0])
        ax_rgb.imshow(display_image)
        ax_rgb.set_title(f'{ind_id}\nRGB Image', fontsize=12, pad=10)
        ax_rgb.axis('off')
        
        # Columns 1-3: Feature maps
        for col, (feature_idx, feature_map) in enumerate(zip(top_feature_indices, feature_maps)):
            ax_feat = fig.add_subplot(gs[row, col + 1])
            
            # Normalize feature map
            normalized_map = normalize_feature_map(feature_map)
            
            # Display with colormap
            im = ax_feat.imshow(normalized_map, cmap=colormap, vmin=0, vmax=1, aspect='auto')
            ax_feat.set_title(f'Feature {feature_idx}\nActivation Map', fontsize=12, pad=10)
            ax_feat.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04)
        
        # Column 4: RGB Composite
        if len(feature_maps) == 3:
            ax_composite = fig.add_subplot(gs[row, 4])
            
            # Create RGB composite
            rgb_composite = create_rgb_composite(feature_maps)
            
            # Display composite
            ax_composite.imshow(rgb_composite, aspect='auto')
            ax_composite.set_title(f'RGB Composite\nF{top_feature_indices[0]}-F{top_feature_indices[1]}-F{top_feature_indices[2]}', 
                                 fontsize=12, pad=10)
            ax_composite.axis('off')
        
        row += 1
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Visualization saved to: {output_path}")


def create_feature_importance_summary(linear_weights, individual_ids, top_k=10):
    """Create summary of feature importance across all classes
    
    Args:
        linear_weights: Linear layer weights [num_classes, num_features]
        individual_ids: List of class names
        top_k: Number of top features to analyze
        
    Returns:
        Dict with feature importance analysis
    """
    # Global importance
    global_importance = torch.max(torch.abs(linear_weights), dim=0)[0]
    top_global_indices = torch.argsort(global_importance, descending=True)[:top_k]
    
    # Per-class importance  
    class_importance = {}
    for i, ind_id in enumerate(individual_ids):
        class_weights = torch.abs(linear_weights[i, :])
        top_class_indices = torch.argsort(class_weights, descending=True)[:top_k]
        class_importance[ind_id] = {
            'top_features': top_class_indices.cpu().numpy(),
            'top_weights': class_weights[top_class_indices].cpu().numpy()
        }
    
    summary = {
        'global_top_features': top_global_indices.cpu().numpy(),
        'global_importance_values': global_importance[top_global_indices].cpu().numpy(),
        'class_specific': class_importance,
        'weight_matrix_shape': linear_weights.shape,
        'individual_ids': individual_ids
    }
    
    return summary