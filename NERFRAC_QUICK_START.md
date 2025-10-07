# NeRFrac Quick Start Guide

## Installation Check

```bash
# Verify nerfrac is available
ns-train --help | grep nerfrac
# Should show: nerfrac    NeRFrac (ICCV 2023): Neural Radiance Fields...
```

## 3-Step Quick Start

### Step 1: Prepare Data

```bash
# Process your underwater images
ns-process-data images \
    --data /path/to/underwater/images \
    --output-dir /path/to/processed
```

### Step 2: Train

```bash
# Basic training (uses defaults: water n=1.333, init_depth=-0.6)
ns-train nerfrac --data /path/to/processed

# Or with custom settings
ns-train nerfrac --data /path/to/processed \
    --pipeline.model.init_depth -0.55 \
    --pipeline.model.refractive_index 1.333 \
    --max-num-iterations 200000
```

### Step 3: View Results

Viewer automatically opens at: `http://localhost:7007`

## Common Commands

### Training Variants

```bash
# Quick test (10k iterations, ~30 min)
ns-train nerfrac --data DATA --max-num-iterations 10000

# Full quality (200k iterations, ~8 hours)
ns-train nerfrac --data DATA --max-num-iterations 200000

# Low memory mode
ns-train nerfrac --data DATA \
    --pipeline.datamanager.train_num_rays_per_batch 256
```

### Rendering

```bash
# Render camera path (create path.json in viewer first)
ns-render camera-path \
    --load-config outputs/nerfrac/.../config.yml \
    --camera-path-filename path.json \
    --output-path video.mp4

# Render test views
ns-render dataset \
    --load-config outputs/nerfrac/.../config.yml \
    --output-path test_renders/ \
    --split test
```

### Evaluation

```bash
# Compute metrics
ns-eval --load-config outputs/nerfrac/.../config.yml
```

## Key Parameters

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `init_depth` | -0.6 | Initial surface depth (NDC) | Surface at wrong location |
| `refractive_index` | 1.333 | Medium refractive index | Using glass (1.5) or other |
| `num_coarse_samples` | 64 | Samples per ray (coarse) | Blurry results → increase |
| `num_importance_samples` | 64 | Samples per ray (fine) | Blurry results → increase |
| `train_num_rays_per_batch` | 500 | Rays per training batch | OOM → decrease |
| `max_num_iterations` | 200000 | Total training iterations | Time constraint |

## Parameter Tuning Guide

### Water Surface Location Wrong?

Try different `init_depth` values:
```bash
# Surface too high → more negative
--pipeline.model.init_depth -0.8

# Surface too low → less negative
--pipeline.model.init_depth -0.4
```

### Refractive Index Reference

```bash
# Water
--pipeline.model.refractive_index 1.333

# Acrylic/Plexiglass
--pipeline.model.refractive_index 1.49

# Glass (crown glass)
--pipeline.model.refractive_index 1.52

# Glass (flint glass)
--pipeline.model.refractive_index 1.7
```

## Typical Workflow

```bash
# 1. Process data (5-10 min)
ns-process-data images --data raw/ --output-dir processed/

# 2. Quick test (30 min)
ns-train nerfrac --data processed/ --max-num-iterations 10000

# 3. Check in viewer → looks good?

# 4. Full training (8 hours)
ns-train nerfrac --data processed/ --max-num-iterations 200000

# 5. Evaluate
ns-eval --load-config outputs/.../config.yml

# 6. Render final video
ns-render camera-path \
    --load-config outputs/.../config.yml \
    --camera-path-filename path.json \
    --output-path final.mp4
```

## Output Files

After training, find outputs in:
```
outputs/nerfrac/YYYY-MM-DD_HHMMSS/
├── config.yml              # Configuration used
├── nerfstudio_models/      # Model checkpoints
│   ├── step-000010000.ckpt
│   ├── step-000020000.ckpt
│   └── ...
└── dataparser_transforms.json
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Surface not visible | Adjust `--pipeline.model.init_depth` |
| Out of memory | Reduce `--pipeline.datamanager.train_num_rays_per_batch` |
| Training too slow | Reduce samples or batch size |
| Blurry underwater | Increase `num_coarse_samples` and `num_importance_samples` |
| Wrong refraction | Check `refractive_index` value |

## Quick Test Dataset

Download a test underwater scene:
```bash
# Use any LLFF or nerfstudio dataset
# Or process your own underwater photos
ns-process-data images --data my_photos/ --output-dir processed/
ns-train nerfrac --data processed/
```

## Monitoring Training

Watch these metrics in viewer or TensorBoard:
- **PSNR**: Should increase to 25-35+ (higher = better)
- **rgb_loss_fine**: Should decrease to < 0.01
- **Viewer**: Render quality improves over time

Target: **PSNR > 30** after 200k iterations

## Getting Help

- Full guide: `NERFRAC_USAGE_GUIDE.md`
- Implementation details: `NERFRAC_IMPLEMENTATION.md`
- Nerfstudio docs: https://docs.nerf.studio

## Example: Complete Session

```bash
# Real example start to finish
cd ~/my_underwater_project

# 1. Prepare
ns-process-data images \
    --data raw_images/ \
    --output-dir processed_underwater/

# 2. Train with monitoring
ns-train nerfrac \
    --data processed_underwater/ \
    --experiment-name coral_reef \
    --viewer.websocket-port 7007

# 3. Wait for training (leave running, ~8 hours)
# Monitor at http://localhost:7007

# 4. After training completes
OUTPUT_PATH="outputs/coral_reef/2025-10-07_120000/nerfstudio_models"

# 5. Evaluate
ns-eval --load-config $OUTPUT_PATH/../config.yml

# 6. Create camera path in viewer, save as path.json

# 7. Render final video
ns-render camera-path \
    --load-config $OUTPUT_PATH/../config.yml \
    --camera-path-filename path.json \
    --output-path coral_reef_final.mp4

# Done! 🎉
```

---

**Ready to start?** Run: `ns-train nerfrac --data YOUR_DATA_PATH`
