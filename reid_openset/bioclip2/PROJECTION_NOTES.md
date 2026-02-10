# BioCLIP-2 Projection Fix: Raw ViT Features for Re-ID

## Problem

`encode_image()` in open_clip passes the ViT-L CLS token through a learned projection matrix (`visual.proj`, shape [1024, 768]) that maps visual features into CLIP's shared text-image embedding space. This projection is trained for image-text alignment -- it optimizes for matching images to captions, not for preserving fine-grained visual distinctions between individuals.

For re-ID, this is a lossy bottleneck. The 1024-d CLS token contains rich spatial and textural information (pelage patterns, marking geometry). The 768-d projected output discards exactly the kind of detail that distinguishes one wolverine from another, in favor of features useful for semantic text matching.

DINOv3 doesn't have this issue -- it outputs raw CLS token features directly with no such projection.

## Fix

```python
backbone.visual.proj = None
```

Setting `visual.proj = None` bypasses the CLIP projection. `encode_image()` then returns the raw 1024-d ViT-L CLS features after `ln_post` but before the text-alignment projection.

## open_clip Source References

- **Projection parameter initialized** at [transformer.py#L928](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py#L928):
  `self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))`

- **Projection applied in forward pass** at [transformer.py#L1008](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py#L1008):
  `pooled = pooled @ self.proj` (guarded by `if self.proj is not None`)

- **encode_image delegates to visual tower** at [model.py#L497-L499](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L497):
  `features = self.visual(image)`

The `if self.proj is not None` guard at line 1008 is what makes the `None` assignment a clean bypass.

## Impact

| | Before (projected) | After (raw) |
|---|---|---|
| Output dim | 768 | 1024 |
| Feature type | Text-aligned | Visual |
| Optimized for | Image-caption matching | Spatial/textural detail |
