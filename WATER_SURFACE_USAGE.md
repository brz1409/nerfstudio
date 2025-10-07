# Water Surface Configuration - Usage Guide

## Overview

Two-Media NeRF now supports **three explicit coordinate systems** for specifying water surface height. This resolves ambiguities when working with data from different sources (Agisoft Metashape, COLMAP, etc.).

## The Three Coordinate Systems

### 1. Metashape Coordinates (Recommended for Agisoft Metashape data)

**Use when**: Your data was processed with `ns-process-data metashape` from `cameras.xml`

**Parameter**: `--pipeline.model.water_surface_height_metashape HEIGHT`

**Coordinate system**: Original Agisoft Metashape world coordinates (OpenCV convention)
- Camera looks down -Z axis
- +X is right, +Y is down, +Z is forward
- Horizontal water surface at z = HEIGHT

**Example**:
```bash
ns-train two-media-vanilla-nerf --data data/my_metashape_project \
    --pipeline.model.water_surface_height_metashape 5.0
```

**Transformation applied**:
1. Metashape → Nerfstudio World (via `applied_transform` axis permutation)
2. Nerfstudio World → Model Space (via `dataparser_transform` + scale)

---

### 2. Nerfstudio World Coordinates (For COLMAP/Standard Processing)

**Use when**: Your data was processed with standard nerfstudio tools (COLMAP, Blender, etc.)

**Parameter**: `--pipeline.model.water_surface_height_world HEIGHT`

**Coordinate system**: Nerfstudio world coordinates (after initial processing, before normalization)
- OpenGL convention: +X right, +Y up, +Z backward
- Horizontal water surface at z = HEIGHT

**Example**:
```bash
ns-train two-media-vanilla-nerf --data data/my_colmap_project \
    --pipeline.model.water_surface_height_world 2.5
```

**Transformation applied**:
- Nerfstudio World → Model Space (via `dataparser_transform` + scale)

---

### 3. Model Coordinates (Advanced/Direct Control)

**Use when**: You know the exact water height in the normalized model coordinate system

**Parameter**: `--pipeline.model.water_surface_height_model HEIGHT`

**Coordinate system**: Final model space (normalized, after all transformations)
- Scene is typically normalized to unit box around origin
- Direct control, no transformations applied

**Example**:
```bash
ns-train two-media-vanilla-nerf --data data/my_project \
    --pipeline.model.water_surface_height_model 0.0
```

**Transformation applied**: None (used directly)

---

## Priority System

If multiple parameters are specified, the following priority is used:

1. **model** (highest priority) - Direct model coordinates
2. **world** - Nerfstudio world coordinates
3. **metashape** (lowest priority) - Metashape coordinates

**Example**: If both `water_surface_height_model` and `water_surface_height_metashape` are set, only `water_surface_height_model` is used.

---

## Common Scenarios

### Scenario 1: Agisoft Metashape Data (cameras.xml)

You processed underwater images with Agisoft Metashape and exported `cameras.xml`:

```bash
# Step 1: Process data
ns-process-data metashape --data path/to/metashape_project --xml cameras.xml

# Step 2: Train with Metashape coordinates
ns-train two-media-vanilla-nerf --data data/nerfstudio/my_scene \
    --pipeline.model.water_surface_height_metashape 3.5 \
    --pipeline.model.refractive_index_air 1.0 \
    --pipeline.model.refractive_index_water 1.333
```

**Water surface height**: Measured in the original Metashape coordinate system (usually z-axis)

---

### Scenario 2: COLMAP Processed Data

You used `ns-process-data images` with COLMAP:

```bash
# Step 1: Process data
ns-process-data images --data path/to/images

# Step 2: Train with Nerfstudio world coordinates
ns-train two-media-vanilla-nerf --data data/nerfstudio/my_scene \
    --pipeline.model.water_surface_height_world 0.0 \
    --pipeline.model.refractive_index_air 1.0 \
    --pipeline.model.refractive_index_water 1.333
```

**Water surface height**: Measured in COLMAP's reconstructed world coordinates

---

### Scenario 3: Fine-Tuning in Model Space

After initial training, you notice the water surface is slightly off:

```bash
# Directly specify in normalized model coordinates
ns-train two-media-vanilla-nerf --data data/nerfstudio/my_scene \
    --pipeline.model.water_surface_height_model -0.05 \
    --load-dir outputs/my_scene/nerfstudio_models
```

**Water surface height**: Fine-tuned position in the normalized [-1, 1] model space

---

## Diagnostic Output

When training starts, you'll see detailed logging about the water surface transformation:

```
[cyan]━━━ Water Surface (Metashape → Model) ━━━[/cyan]
  Source coordinate system: Metashape
  Height (metashape): z = 5.0000
  Height (world): z = 3.5000
  Height (model): z = 0.1234

  Applied Transform:
    [[0.0, 0.0, 1.0, 0.0],
     [1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]

  Dataparser Transform + Scale:
    Scale: 0.5
    Rotation + Translation: [4x4 matrix]

  Plane equation (model space):
    normal = [0.000, 0.000, 1.000]
    d = -0.1234
    → z = 0.1234

[green]✓ Water surface is horizontal[/green]
```

**Watch for warnings**:
- `⚠ WARNING: Water surface tilted X.X°` - Surface is not horizontal (check coordinate system)
- `✗ ERROR: Water surface is VERTICAL!` - Likely wrong coordinate system selected

---

## Troubleshooting

### Problem: Water surface not visible in viewer

**Possible causes**:
1. Wrong coordinate system parameter used
2. Water height outside scene bounds
3. Coordinate transformation issue

**Solution**:
1. Check diagnostic output at training start
2. Try different coordinate system parameters
3. Verify water height is reasonable for your data scale
4. Look for tilt warnings (should be horizontal)

---

### Problem: "Water surface is VERTICAL!" error

**Cause**: Metashape data treated as Nerfstudio world coordinates (or vice versa)

**Solution**:
- If using Metashape data (`cameras.xml`), use `water_surface_height_metashape`
- If using COLMAP/standard data, use `water_surface_height_world`

---

### Problem: Refraction looks incorrect

**Cause**: Water surface at wrong height

**Solution**:
1. Check actual water level in your captured scene
2. Measure height in the appropriate coordinate system
3. Start with rough estimate, refine based on viewer visualization
4. Use `water_surface_height_model` for direct fine-tuning

---

## Technical Details

### Metashape Coordinate Transform

Metashape uses OpenCV convention, which is permuted when imported to Nerfstudio:

```python
# Applied transform (Metashape → Nerfstudio)
applied_transform = [
    [0, 0, 1, 0],  # Nerfstudio +X = Metashape +Z
    [1, 0, 0, 0],  # Nerfstudio +Y = Metashape +X
    [0, 1, 0, 0],  # Nerfstudio +Z = Metashape +Y
    [0, 0, 0, 1]
]
```

**Effect**: A horizontal plane at `z=h` in Metashape becomes a **different** horizontal plane in Nerfstudio world coordinates.

### Dataparser Transform

The dataparser transform normalizes and centers the scene:

```python
# Nerfstudio World → Model
point_model = scale * (R @ point_world + t)
normal_model = R @ normal_world
```

This ensures the scene fits in a standard bounding box for efficient training.

---

## Best Practices

1. **Always specify coordinate system explicitly** - Don't rely on defaults or assumptions
2. **Check diagnostic output** - Verify transformations look correct at training start
3. **Start with known values** - Use measured water height from your capture metadata
4. **Validate in viewer** - Check water surface position before long training runs
5. **Use model coordinates for fine-tuning** - After initial training, adjust in normalized space

---

## Related Documentation

- [Water Surface Coordinate System Analysis](WATER_SURFACE_COORDINATE_SYSTEM_ANALYSIS.md) - Deep technical dive
- [Water Sampling Fix Summary](WATER_SAMPLING_FIX_SUMMARY.md) - Scene box clipping implementation
- [Nerfstudio Dataparsers](https://docs.nerf.studio/developer_guides/pipelines/dataparsers.html) - Coordinate system details
