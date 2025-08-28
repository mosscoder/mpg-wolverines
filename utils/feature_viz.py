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


def integrated_gradients_patches(model, image_tensor, class_id, device, steps=32, baseline='zeros'):
    """Compute integrated gradients for patch tokens to show importance
    
    Args:
        model: WolverinesModel with backbone and classifier
        image_tensor: Input image [1, 3, H, W]
        class_id: Target class to compute gradients for
        device: Device to run on
        steps: Number of integration steps
        baseline: Type of baseline ('zeros' or 'mean')
        
    Returns:
        ig: Integrated gradients per patch token [P, D]
        per_patch: Relevance score per patch [P]
    """
    image_tensor = image_tensor.to(device)
    
    # Get patch tokens from forward pass
    with torch.no_grad():
        out = model.backbone(image_tensor)
        tokens = out.last_hidden_state.squeeze(0)  # [num_tokens, D]
        
        # Determine number of special tokens (CLS + register tokens)
        num_patches = (image_tensor.shape[-1] // 16) * (image_tensor.shape[-2] // 16)
        num_tokens = tokens.shape[0]
        num_special = num_tokens - num_patches
        
        patches = tokens[num_special:].detach()  # [P, D] - skip CLS and register tokens
    
    # Create baseline
    if baseline == 'zeros':
        base = torch.zeros_like(patches)
    elif baseline == 'mean':
        base = patches.mean(dim=0, keepdim=True).expand_as(patches)
    else:
        base = torch.zeros_like(patches)
    
    # Accumulate gradients
    total = torch.zeros_like(patches)
    
    for alpha in torch.linspace(0, 1, steps, device=device):
        # Interpolate between baseline and actual patches
        x = (base + alpha * (patches - base)).detach().requires_grad_(True)
        
        # Create modified tokens with interpolated patches
        with torch.no_grad():
            out = model.backbone(image_tensor)
            toks = out.last_hidden_state.squeeze(0).detach()  # [num_tokens, D]
        
        # Replace patch tokens with interpolated values
        toks = toks.clone()
        toks[num_special:] = x
        
        # Get CLS token and compute logit
        cls = toks[0].unsqueeze(0)  # [1, D]
        logit = model.classifier(cls)[0, class_id]  # Scalar
        
        # Compute gradient
        (grad,) = torch.autograd.grad(logit, x, retain_graph=False)
        total += grad
    
    # Compute integrated gradients
    ig = (patches - base) * (total / steps)  # [P, D]
    per_patch = ig.sum(dim=1)  # [P] - relevance per patch
    
    return ig, per_patch


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


def create_spatial_feature_maps(patch_features, feature_indices, patch_grid_size, target_size=None, original_size=None, interpolate=True):
    """Create spatial visualization of feature activations
    
    Args:
        patch_features: Patch features array [num_patches, num_features]  
        feature_indices: Indices of features to visualize
        patch_grid_size: Tuple of (height, width) for patch grid (e.g., (45, 45))
        target_size: Target size for upsampled visualizations (for backward compatibility)
        original_size: Original image size to maintain aspect ratio (height, width)
        interpolate: Whether to interpolate patches to larger size (False = native resolution)
        
    Returns:
        List of feature maps (either upsampled or native resolution)
    """
    feature_maps = []
    
    for feature_idx in feature_indices:
        # Get feature activations for all patches
        feature_activations = patch_features[:, feature_idx]  # [num_patches]
        
        # Reshape to spatial grid
        feature_map = feature_activations.reshape(patch_grid_size)  # [grid_h, grid_w]
        
        if interpolate:
            # Use original size if provided, otherwise fall back to target_size
            if original_size is not None:
                upsample_size = original_size
            else:
                upsample_size = target_size or (728, 728)
            
            # Convert to tensor for upsampling
            feature_tensor = torch.from_numpy(feature_map).float().unsqueeze(0).unsqueeze(0)  # [1, 1, grid_h, grid_w]
            
            # Upsample to target size maintaining aspect ratio
            upsampled = F.interpolate(feature_tensor, size=upsample_size, mode='bilinear', align_corners=False)
            upsampled_map = upsampled[0, 0].numpy()  # [target_h, target_w]
            
            feature_maps.append(upsampled_map)
        else:
            # Keep native patch resolution for crisp visualization
            feature_maps.append(feature_map)
    
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


def create_integrated_gradients_grid(selected_images, individual_ids, model, transform, 
                                    device, output_path, patch_grid_size=(45, 45),
                                    figsize=(20, 4), colormap='RdBu_r', steps=32):
    """Create visualization grid using integrated gradients to show patch importance
    
    Args:
        selected_images: Dict mapping individual_id -> image info
        individual_ids: List of individual IDs
        model: Trained model with backbone and classifier
        transform: Image transform
        device: Device for computation
        output_path: Path to save visualization
        patch_grid_size: Grid size for patches
        figsize: Figure size per row
        colormap: Colormap for gradients (RdBu_r shows positive/negative)
        steps: Integration steps for IG
    """
    import matplotlib.patches as mpatches
    
    n_individuals = len([ind for ind in individual_ids if ind in selected_images])
    n_cols = 5  # Original, IG heatmap, Top positive patches, Top negative patches, Overlay
    
    fig_height = figsize[1] * n_individuals
    fig = plt.figure(figsize=(figsize[0], fig_height))
    gs = gridspec.GridSpec(n_individuals, n_cols, figure=fig, hspace=0.3, wspace=0.1)
    
    # Map individual IDs to class indices
    id_to_class = {ind_id: i for i, ind_id in enumerate(individual_ids)}
    
    row = 0
    for ind_id in individual_ids:
        if ind_id not in selected_images:
            print(f"Skipping {ind_id} - no test image")
            continue
        
        image_info = selected_images[ind_id]
        pil_image = image_info['image']
        class_id = id_to_class[ind_id]
        
        print(f"Processing {ind_id} (class {class_id})...")
        
        # Get original dimensions
        original_width, original_height = pil_image.size
        
        # Apply transform and compute integrated gradients
        image_tensor = transform(pil_image).unsqueeze(0)  # [1, 3, H, W]
        ig, per_patch = integrated_gradients_patches(model, image_tensor, class_id, device, steps=steps)
        
        # Reshape per_patch to spatial grid
        relevance_map = per_patch.cpu().numpy().reshape(patch_grid_size)
        
        # Prepare displays
        display_image = np.array(pil_image) / 255.0
        
        # Column 0: Original Image
        ax_orig = fig.add_subplot(gs[row, 0])
        ax_orig.imshow(display_image)
        ax_orig.set_title(f'{ind_id}\nOriginal Image', fontsize=12, pad=10)
        ax_orig.axis('off')
        
        # Column 1: IG Heatmap (native resolution)
        ax_ig = fig.add_subplot(gs[row, 1])
        vmax = max(abs(relevance_map.min()), abs(relevance_map.max()))
        im = ax_ig.imshow(relevance_map, cmap=colormap, vmin=-vmax, vmax=vmax,
                         aspect='auto', interpolation='nearest')
        ax_ig.set_title('Integrated Gradients\n(Patch Importance)', fontsize=12, pad=10)
        ax_ig.axis('off')
        plt.colorbar(im, ax=ax_ig, fraction=0.046, pad=0.04)
        
        # Column 2: Top positive patches
        ax_pos = fig.add_subplot(gs[row, 2])
        pos_mask = np.where(relevance_map > 0, relevance_map, 0)
        im_pos = ax_pos.imshow(pos_mask, cmap='Reds', vmin=0, vmax=vmax,
                              aspect='auto', interpolation='nearest')
        ax_pos.set_title('Positive\nContributions', fontsize=12, pad=10)
        ax_pos.axis('off')
        plt.colorbar(im_pos, ax=ax_pos, fraction=0.046, pad=0.04)
        
        # Column 3: Top negative patches
        ax_neg = fig.add_subplot(gs[row, 3])
        neg_mask = np.where(relevance_map < 0, -relevance_map, 0)
        im_neg = ax_neg.imshow(neg_mask, cmap='Blues', vmin=0, vmax=vmax,
                              aspect='auto', interpolation='nearest')
        ax_neg.set_title('Negative\nContributions', fontsize=12, pad=10)
        ax_neg.axis('off')
        plt.colorbar(im_neg, ax=ax_neg, fraction=0.046, pad=0.04)
        
        # Column 4: Overlay on original (upsampled heatmap)
        ax_overlay = fig.add_subplot(gs[row, 4])
        ax_overlay.imshow(display_image)
        
        # Upsample relevance map to match image size
        relevance_tensor = torch.from_numpy(relevance_map).float().unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(relevance_tensor, size=(original_height, original_width), 
                                mode='bilinear', align_corners=False)
        upsampled_relevance = upsampled[0, 0].numpy()
        
        # Overlay with transparency
        im_overlay = ax_overlay.imshow(upsampled_relevance, cmap=colormap, vmin=-vmax, vmax=vmax,
                                      alpha=0.5, interpolation='bilinear')
        ax_overlay.set_title('IG Overlay\non Original', fontsize=12, pad=10)
        ax_overlay.axis('off')
        
        row += 1
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Integrated gradients visualization saved to: {output_path}")


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
        
        # Create feature visualizations (without interpolation for crisp patches)
        feature_maps = create_spatial_feature_maps(patch_features, top_feature_indices, 
                                                 patch_grid_size, target_size=target_size, 
                                                 original_size=upsample_size, interpolate=False)
        
        # Prepare RGB display - show original image in native aspect ratio
        display_image = np.array(pil_image) / 255.0  # Convert to [0,1] range
        
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
            
            # Display with colormap (use nearest neighbor for crisp patches)
            im = ax_feat.imshow(normalized_map, cmap=colormap, vmin=0, vmax=1, 
                               aspect='auto', interpolation='nearest')
            ax_feat.set_title(f'Feature {feature_idx}\nActivation Map', fontsize=12, pad=10)
            ax_feat.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04)
        
        # Column 4: RGB Composite
        if len(feature_maps) == 3:
            ax_composite = fig.add_subplot(gs[row, 4])
            
            # Create RGB composite
            rgb_composite = create_rgb_composite(feature_maps)
            
            # Display composite (use nearest neighbor for crisp patches)
            ax_composite.imshow(rgb_composite, aspect='auto', interpolation='nearest')
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