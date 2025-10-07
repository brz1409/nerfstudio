# NeRFrac Usage Guide

## Quick Start

### 1. Basic Training

```bash
# Train with default settings (water scene)
ns-train nerfrac --data /path/to/your/underwater/scene

# The viewer will automatically open at http://localhost:7007
```

### 2. View Available Methods

```bash
# List all available methods (nerfrac should be listed)
ns-train --help

# See nerfrac-specific options
ns-train nerfrac --help
```

## Data Preparation

### Supported Data Formats

NeRFrac works with standard nerfstudio data formats:

#### Option 1: Process with COLMAP (Recommended)
```bash
# Process images from underwater scene
ns-process-data images \
    --data /path/to/underwater/images \
    --output-dir /path/to/processed/underwater

# Then train
ns-train nerfrac --data /path/to/processed/underwater
```

#### Option 2: Use existing LLFF/NeRF datasets
```bash
# If you already have LLFF format data
ns-train nerfrac \
    --data /path/to/llff/data \
    nerfstudio-data
```

#### Option 3: Process video
```bash
# Extract frames from underwater video
ns-process-data video \
    --data underwater_video.mp4 \
    --output-dir processed_underwater

ns-train nerfrac --data processed_underwater
```

### Data Requirements

- **Scene Type**: Underwater or through-glass scenes
- **Images**: 20-100+ images with varying viewpoints
- **Camera**: Any camera (phone, DSLR, action cam)
- **Format**: JPG, PNG
- **Resolution**: Any (will be downscaled if needed)

## Configuration Options

### Basic Parameters

```bash
# Adjust initial depth estimate (distance to water surface in NDC)
ns-train nerfrac --data DATA_PATH \
    --pipeline.model.init_depth -0.6

# Change refractive index (default: 1.333 for water)
ns-train nerfrac --data DATA_PATH \
    --pipeline.model.refractive_index 1.333

# For glass/acrylic: n ≈ 1.5
ns-train nerfrac --data DATA_PATH \
    --pipeline.model.refractive_index 1.5
```

### Advanced Parameters

```bash
# Full configuration example
ns-train nerfrac --data DATA_PATH \
    --pipeline.model.init_depth -0.55 \
    --pipeline.model.refractive_index 1.333 \
    --pipeline.model.multires_origin 8 \
    --pipeline.model.multires_refrac_dir 8 \
    --pipeline.model.num_coarse_samples 64 \
    --pipeline.model.num_importance_samples 64 \
    --pipeline.model.raw_noise_std 1.0 \
    --max-num-iterations 200000 \
    --pipeline.datamanager.train_num_rays_per_batch 500
```

### Camera Parameters (Optional)

If your data has specific camera parameters:

```bash
ns-train nerfrac --data DATA_PATH \
    --pipeline.model.image_height 1080 \
    --pipeline.model.image_width 1920 \
    --pipeline.model.focal_length 1000.0
```

## Training Configuration

### Default Training Settings

```python
max_iterations: 200,000
batch_size: 500 rays
learning_rate: 5e-4
scheduler: Exponential decay to 5e-5
```

### Adjust Training Duration

```bash
# Quick test (10k iterations)
ns-train nerfrac --data DATA_PATH \
    --max-num-iterations 10000

# Full training (200k iterations, ~4-8 hours on GPU)
ns-train nerfrac --data DATA_PATH \
    --max-num-iterations 200000

# Extended training for complex scenes
ns-train nerfrac --data DATA_PATH \
    --max-num-iterations 300000
```

### Memory Management

```bash
# Reduce batch size for limited GPU memory
ns-train nerfrac --data DATA_PATH \
    --pipeline.datamanager.train_num_rays_per_batch 256

# Reduce samples per ray
ns-train nerfrac --data DATA_PATH \
    --pipeline.model.num_coarse_samples 32 \
    --pipeline.model.num_importance_samples 32
```

## Viewing Training

### Real-time Viewer

```bash
# Training automatically opens viewer at http://localhost:7007
# Or manually specify port:
ns-train nerfrac --data DATA_PATH \
    --viewer.websocket-port 8080
```

### Viewer Features
- **Camera Controls**: Click and drag to orbit
- **Render Tab**: Adjust render quality, FOV
- **Scene Tab**: View scene bounds
- **Export Tab**: Export camera paths

### TensorBoard Monitoring

```bash
# In separate terminal:
tensorboard --logdir outputs/

# Then open http://localhost:6006
```

Metrics to monitor:
- `rgb_loss_coarse`: Coarse network RGB loss
- `rgb_loss_fine`: Fine network RGB loss
- `psnr`: Peak Signal-to-Noise Ratio
- Surface visualization outputs

## Rendering Outputs

### Render Novel Views

```bash
# First, create camera path in viewer (RENDER tab → Camera Path)
# Save as camera_path.json

# Then render video
ns-render camera-path \
    --load-config outputs/nerfrac/.../config.yml \
    --camera-path-filename camera_path.json \
    --output-path renders/underwater_flythrough.mp4
```

### Render Interpolation

```bash
# Render interpolated path between training views
ns-render interpolate \
    --load-config outputs/nerfrac/.../config.yml \
    --output-path renders/interpolation.mp4 \
    --interpolation-steps 30
```

### Render Dataset Views

```bash
# Render all training views
ns-render dataset \
    --load-config outputs/nerfrac/.../config.yml \
    --output-path renders/training_views/ \
    --split train

# Render test views (for evaluation)
ns-render dataset \
    --load-config outputs/nerfrac/.../config.yml \
    --output-path renders/test_views/ \
    --split test
```

## Evaluation

### Quantitative Metrics

```bash
# Evaluate on test set
ns-eval \
    --load-config outputs/nerfrac/.../config.yml \
    --output-path evaluation_results.json
```

Metrics computed:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity

### Visual Inspection

```bash
# Launch viewer on trained model
ns-viewer --load-config outputs/nerfrac/.../config.yml
```

## Example Workflows

### Workflow 1: Quick Test on New Data

```bash
# 1. Process data
ns-process-data images --data raw_images/ --output-dir processed/

# 2. Quick test training (10k iterations)
ns-train nerfrac --data processed/ --max-num-iterations 10000

# 3. Check results in viewer
# → If good, continue to full training
```

### Workflow 2: Full Production Pipeline

```bash
# 1. Process high-quality images
ns-process-data images \
    --data raw_underwater_images/ \
    --output-dir processed_underwater/ \
    --num-downscales 0  # Keep full resolution

# 2. Full training
ns-train nerfrac --data processed_underwater/ \
    --pipeline.model.init_depth -0.6 \
    --pipeline.model.refractive_index 1.333 \
    --max-num-iterations 200000 \
    --experiment-name underwater_full

# 3. Evaluate
ns-eval --load-config outputs/.../config.yml --output-path results.json

# 4. Create camera path in viewer and render
ns-render camera-path \
    --load-config outputs/.../config.yml \
    --camera-path-filename path.json \
    --output-path final_video.mp4 \
    --output-format video
```

### Workflow 3: Resume Training

```bash
# Resume from checkpoint
ns-train nerfrac --data DATA_PATH \
    --load-dir outputs/nerfrac/2025-10-07_XXXXXX/nerfstudio_models

# Continue training for more iterations
ns-train nerfrac --data DATA_PATH \
    --load-dir outputs/.../nerfstudio_models \
    --max-num-iterations 300000
```

## Troubleshooting

### Common Issues

#### 1. "Water surface not appearing correctly"

**Solution**: Adjust `init_depth`
```bash
# Try different values between -0.3 and -0.9
ns-train nerfrac --data DATA \
    --pipeline.model.init_depth -0.7
```

#### 2. "Blurry refracted regions"

**Solution**: Increase sampling
```bash
ns-train nerfrac --data DATA \
    --pipeline.model.num_coarse_samples 128 \
    --pipeline.model.num_importance_samples 128
```

#### 3. "Training is slow"

**Solution**: Reduce batch size or samples
```bash
ns-train nerfrac --data DATA \
    --pipeline.datamanager.train_num_rays_per_batch 256 \
    --pipeline.model.num_coarse_samples 32
```

#### 4. "Out of memory error"

**Solution**: Reduce memory usage
```bash
ns-train nerfrac --data DATA \
    --pipeline.datamanager.train_num_rays_per_batch 128 \
    --pipeline.model.eval_num_rays_per_chunk 128
```

#### 5. "Refractive index unknown"

**Common values**:
- Water: 1.333
- Acrylic/Plexiglass: 1.49-1.50
- Glass: 1.5-1.9 (depends on type)
- Oil: 1.45-1.47

Start with 1.333 for underwater scenes.

## Tips for Best Results

### Data Collection

1. **Capture sufficient views**: 50-100+ images recommended
2. **Vary viewpoints**: Cover the scene from multiple angles
3. **Avoid motion blur**: Use fast shutter speed underwater
4. **Good lighting**: Ensure scene is well-lit
5. **Overlap**: 60-80% overlap between consecutive images

### Training

1. **Start with default params**: Works well for most cases
2. **Monitor training**: Watch PSNR increase (should reach 25-35+)
3. **Check surface**: Visualize `surface_points` output
4. **Patience**: 200k iterations ≈ 4-8 hours on GPU

### Scene Requirements

**Best Results**:
- Static scene (no moving objects)
- Clear water/medium (not too turbid)
- Visible features underwater
- Stable water surface (or averaged over time)

**Challenging**:
- Highly turbid water
- Moving/wavy surface
- Very dark scenes
- Mirror-like reflections

## Export and Integration

### Export Point Cloud

```bash
ns-export pointcloud \
    --load-config outputs/.../config.yml \
    --output-dir exports/ \
    --num-points 1000000
```

### Export Mesh (if applicable)

```bash
ns-export poisson \
    --load-config outputs/.../config.yml \
    --output-dir exports/
```

## Advanced: Custom Dataparsers

If using custom data format:

```python
# In your config
ns-train nerfrac \
    --data DATA_PATH \
    --pipeline.datamanager.dataparser nerfstudio-data \
    --pipeline.datamanager.dataparser.scene-scale-factor 1.0
```

## Performance Benchmarks

Typical performance (on RTX 3090):

| Iterations | Time | PSNR | Notes |
|-----------|------|------|-------|
| 10k | ~30 min | 20-25 | Quick test |
| 50k | ~2 hours | 25-30 | Decent quality |
| 100k | ~4 hours | 28-32 | Good quality |
| 200k | ~8 hours | 30-35+ | Best quality |

## Next Steps

1. **Train your first model**: Start with `ns-train nerfrac --data YOUR_DATA`
2. **Explore in viewer**: Understand the scene reconstruction
3. **Tune parameters**: Adjust based on your scene
4. **Render outputs**: Create final visualizations
5. **Share results**: Export for presentations/papers

## Getting Help

- **Documentation**: See `NERFRAC_IMPLEMENTATION.md` for technical details
- **Nerfstudio Docs**: https://docs.nerf.studio
- **Issues**: Report at https://github.com/nerfstudio-project/nerfstudio/issues

## References

- Original Paper: "NeRFrac: Neural Radiance Fields through Refractive Surface" (ICCV 2023)
- Nerfstudio: https://docs.nerf.studio
- Original Code: https://github.com/Yifever20002/NeRFrac
