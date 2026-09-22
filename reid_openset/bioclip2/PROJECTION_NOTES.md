# BioCLIP-2 projection bypass: the encoder's own visual output

## Reason

`encode_image()` in open_clip passes the ViT-L CLS token through a learned projection matrix (`visual.proj`, shape [1024, 768]) that maps visual features into CLIP's shared text-image embedding space. That projection was trained to align images with text.

DINOv3 and MegaDescriptor hand the projection head their own visual output, with no text-alignment layer. For consistency, we take BioCLIP-2's output before its text-alignment layer, so the projection head receives the encoder's own 1024-d visual representation, as it does for the other two encoders. We did not test whether the projected 768-d output would perform differently.

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
| Matches the other two encoders | No | Yes |
