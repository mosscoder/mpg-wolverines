#!/usr/bin/env python3
"""
Diagnostic script to inspect DINOv3 model architecture and find correct module names for LoRA.
Based on approach from user to identify all Linear and Conv2d layers.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

def list_leaf_linear_and_conv(mod):
    """List all leaf Linear and Conv2d modules in the model"""
    leafs = []
    for name, m in mod.named_modules():
        # Most LoRA adapters are placed on Linear layers, sometimes Conv2d.
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            leafs.append(name)
    return leafs

def analyze_attention_structure(model):
    """Analyze the structure of attention layers specifically"""
    print("\n" + "="*80)
    print("ATTENTION LAYER ANALYSIS:")
    print("="*80)
    
    # Try to find attention patterns
    attention_modules = []
    for name, module in model.named_modules():
        name_lower = name.lower()
        if 'attention' in name_lower or 'attn' in name_lower:
            attention_modules.append((name, type(module).__name__))
    
    if attention_modules:
        print("Found attention-related modules:")
        for name, module_type in attention_modules[:10]:  # Show first 10
            print(f"  {name}: {module_type}")
    else:
        print("No explicit 'attention' modules found")
    
    # Look for transformer blocks
    block_modules = []
    for name, module in model.named_modules():
        if 'block' in name.lower() or 'layer' in name.lower():
            if '.' not in name.replace('blocks.', '').replace('layers.', '').replace('layer.', ''):
                block_modules.append((name, type(module).__name__))
    
    if block_modules:
        print(f"\nFound {len(block_modules)} transformer blocks:")
        for name, module_type in block_modules[:5]:
            print(f"  {name}: {module_type}")
    
    # Look inside first block for structure
    try:
        # Common patterns
        possible_paths = [
            'blocks.0',
            'encoder.layer.0', 
            'layers.0'
        ]
        
        first_block = None
        first_block_name = None
        
        for path in possible_paths:
            try:
                parts = path.split('.')
                current = model
                for part in parts:
                    if part.isdigit():
                        current = current[int(part)]
                    else:
                        current = getattr(current, part)
                first_block = current
                first_block_name = path
                break
            except:
                continue
        
        if first_block:
            print(f"\nStructure of {first_block_name}:")
            for name, module in first_block.named_modules():
                if name:  # Skip the block itself
                    print(f"  {name}: {type(module).__name__}")
        else:
            print("\nCould not access first transformer block")
            
    except Exception as e:
        print(f"Error analyzing block structure: {e}")

def main():
    print("="*80)
    print("DINOv3 MODEL ARCHITECTURE INSPECTION")
    print("="*80)
    
    # Load our focal model
    print("Loading DINOv3-ViT-B/16 model...")
    model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    
    try:
        # Try with trust_remote_code first
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()
        print(f"✓ Model loaded with trust_remote_code=True")
    except:
        try:
            # Fallback without trust_remote_code
            model = AutoModel.from_pretrained(model_name).eval()
            print(f"✓ Model loaded without trust_remote_code")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return
    
    print(f"Model type: {type(model)}")
    print(f"Config type: {type(model.config)}")
    
    # Find all Linear and Conv2d candidates
    print("\n" + "="*80)
    print("ALL LINEAR AND CONV2D MODULES:")
    print("="*80)
    
    candidates = list_leaf_linear_and_conv(model)
    print(f"Found {len(candidates)} Linear/Conv2d modules:")
    
    for i, name in enumerate(sorted(set(candidates))):
        print(f"{i+1:3d}. {name}")
    
    # Analyze attention structure
    analyze_attention_structure(model)
    
    # Filter for likely LoRA targets
    print("\n" + "="*80)
    print("RECOMMENDED LoRA TARGET MODULES:")
    print("="*80)
    
    # Common LoRA target patterns
    target_patterns = ['qkv', 'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2', 'mlp', 'dense']
    
    recommended = []
    for pattern in target_patterns:
        matches = [name for name in candidates if pattern in name.lower()]
        if matches:
            recommended.extend(matches)
            print(f"\nPattern '{pattern}' matches:")
            for name in matches[:5]:  # Show first 5
                print(f"  {name}")
    
    if recommended:
        # Get unique base names (remove layer numbers for pattern)
        unique_patterns = set()
        for name in recommended:
            # Extract pattern by removing layer numbers
            import re
            pattern = re.sub(r'\.\d+\.', '.X.', name)
            unique_patterns.add(pattern)
        
        print(f"\nSUGGESTED LoRA CONFIG:")
        print("target_modules = [")
        for pattern in sorted(unique_patterns):
            # Convert back to module name without layer number
            base_name = pattern.split('.')[-1]
            print(f'    "{base_name}",')
        print("]")
    
    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("Run this script to identify the correct target_modules for LoRA")
    print("="*80)

if __name__ == "__main__":
    main()