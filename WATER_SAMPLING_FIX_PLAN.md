# Water Sampling Fix: Umfassender Plan

## Problem-Diagnose

### Ursprüngliche Annahme (FALSCH)
Der User vermutete, dass `remaining_dist = far - t_water` geometrisch inkorrekt ist und durch Projektion auf den refraktierten Strahl ersetzt werden sollte.

### Tatsächliches Problem (BEWIESEN durch Analyse)
**Das Problem ist NICHT die Projektion, sondern fehlende scene_box-Beachtung!**

Analyse-Ergebnisse (`analyze_scene_box_clipping.py`):
```
BUGGY (far - t_water):           62.5% Samples verschwendet
PROJECTED FIX:                   62.5% Samples verschwendet (kein Unterschied!)
SCENE_BOX CLIPPING:              0% verschwendet (100% Effizienz!)
```

**Root Cause:**
- `far = 10.0` (default) ist viel zu groß für typische scene_box `[-1.5, 1.5]`
- Wasser-Strahlen haben KEIN scene_box clipping
- Luft-Strahlen haben implizites clipping durch `t_air_far = min(t_water - eps, far)`
- Beide Implementierungen (buggy und projected) samplen 5-7 Einheiten außerhalb der Scene

## Lösung

### Primäre Lösung: Scene Box AABB Intersection

**Methode:**
```python
# Berechne Schnittpunkt des refraktierten Strahls mit scene_box AABB
t_near, t_far = intersect_aabb(water_origin, refracted_dirs, self.scene_box.aabb)
remaining_dist = t_far
```

**Vorteile:**
- ✅ 100% Sampling-Effizienz (keine verschwendeten Samples)
- ✅ Funktioniert automatisch mit jeder scene_box Größe
- ✅ Keine manuellen Parameter notwendig
- ✅ Konsistent mit nerfstudio's Architektur (scene_box ist bereits vorhanden)

**Nachteil:**
- Erfordert AABB intersection Implementierung (falls nicht vorhanden)

### Sekundäre Lösung: max_water_distance Parameter

**Methode:**
```python
# Clip gegen konfigurierbare Maximum-Distanz
max_dist = self.config.max_water_distance
remaining_dist = min(far - t_water, max_dist)
```

**Vorteile:**
- ✅ Einfach zu implementieren
- ✅ Benutzer kann kontrollieren

**Nachteile:**
- ⚠️ Immer noch ineffizient (70% Effizienz bei max_dist=3.0)
- ⚠️ Benötigt manuelle Tuning pro Szene
- ⚠️ Kann zu kurz oder zu lang sein

### Empfehlung: Hybrid-Ansatz

Kombiniere beide:
```python
# 1. Berechne scene_box intersection (optimal)
t_near, t_far = intersect_aabb(water_origin, refracted_dirs, self.scene_box.aabb)

# 2. Optional: Clip gegen max_water_distance als Safety
if hasattr(self.config, 'max_water_distance') and self.config.max_water_distance > 0:
    t_far = torch.minimum(t_far, torch.full_like(t_far, self.config.max_water_distance))

remaining_dist = t_far
```

## Implementierungs-Plan

### Phase 1: AABB Intersection Utility (CRITICAL)

**File:** `nerfstudio/utils/math.py` (prüfen ob bereits vorhanden) oder als Methode in `TwoMediaNeRFModel`

**Implementation:**
```python
def intersect_aabb(
    origins: Tensor,  # [N, 3]
    directions: Tensor,  # [N, 3]
    aabb: Tensor,  # [2, 3] - [min_xyz, max_xyz]
) -> Tuple[Tensor, Tensor]:
    """
    Compute ray-AABB intersection.

    Returns:
        t_near: [N] - distance to entry point (or 0 if origin inside)
        t_far: [N] - distance to exit point
    """
    # Compute t values for each axis
    inv_dir = 1.0 / (directions + 1e-6)  # Avoid division by zero
    t_min = (aabb[0] - origins) * inv_dir  # [N, 3]
    t_max = (aabb[1] - origins) * inv_dir  # [N, 3]

    # Handle negative directions (swap t_min and t_max)
    t1 = torch.minimum(t_min, t_max)
    t2 = torch.maximum(t_min, t_max)

    # Intersection is where all axes overlap
    t_near = torch.maximum(t1.max(dim=-1).values, torch.zeros_like(t1[:, 0]))
    t_far = t2.min(dim=-1).values

    # Invalid if t_far < t_near (no intersection)
    t_far = torch.where(t_far >= t_near, t_far, torch.zeros_like(t_far))

    return t_near, t_far
```

**Testing:**
```python
# Unit test in tests/utils/test_math.py
def test_intersect_aabb():
    # Case 1: Ray from outside entering box
    # Case 2: Ray from inside exiting box
    # Case 3: Ray missing box
    # Case 4: Ray parallel to box face
    pass
```

### Phase 2: Refactor Water Ray Sampling

**File:** `nerfstudio/models/two_media_vanilla_nerf.py`

**Location:** Line 374-389 (water_bundle creation)

**Changes:**

**Before:**
```python
if hits_water.any():
    entry_points = origins + directions * t_water.unsqueeze(-1)
    refracted_dirs = self._compute_refraction(directions)
    water_origins = entry_points + refracted_dirs * eps
    remaining_dist = torch.clamp(far - t_water, min=0.0)  # ❌ BUGGY

    water_bundle = RayBundle(
        origins=water_origins,
        directions=refracted_dirs,
        ...
        fars=remaining_dist.unsqueeze(-1),
    )
```

**After:**
```python
if hits_water.any():
    entry_points = origins + directions * t_water.unsqueeze(-1)
    refracted_dirs = self._compute_refraction(directions)
    water_origins = entry_points + refracted_dirs * eps

    # ✅ FIXED: Use scene_box intersection for accurate sampling bounds
    t_near_water, t_far_water = self._intersect_scene_box(water_origins, refracted_dirs)

    # Optional: Clip against max_water_distance config
    if self.config.max_water_distance > 0:
        t_far_water = torch.minimum(t_far_water,
                                     torch.full_like(t_far_water, self.config.max_water_distance))

    remaining_dist = t_far_water

    # Logging for debugging (first iteration only)
    if self.training and not hasattr(self, '_logged_water_sampling'):
        valid_mask = remaining_dist > 0
        if valid_mask.any():
            CONSOLE.log(
                f"[cyan]Water sampling stats: "
                f"mean_dist={remaining_dist[valid_mask].mean():.3f}, "
                f"max_dist={remaining_dist[valid_mask].max():.3f}, "
                f"rays_with_water={valid_mask.sum()}/{num_rays}[/cyan]"
            )
        self._logged_water_sampling = True

    water_bundle = RayBundle(
        origins=water_origins,
        directions=refracted_dirs,
        ...
        fars=remaining_dist.unsqueeze(-1),
    )
```

**Helper Method:**
```python
def _intersect_scene_box(self, origins: Tensor, directions: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute intersection of rays with scene bounding box.

    Args:
        origins: [N, 3] ray origins
        directions: [N, 3] ray directions (normalized)

    Returns:
        t_near: [N] distance to box entry (0 if inside)
        t_far: [N] distance to box exit
    """
    # Import or use existing nerfstudio utility if available
    from nerfstudio.utils.math import intersect_aabb  # or implement locally

    aabb = self.scene_box.aabb.to(origins.device)
    t_near, t_far = intersect_aabb(origins, directions, aabb)

    return t_near, t_far
```

### Phase 3: Config Update

**File:** `nerfstudio/models/two_media_vanilla_nerf.py`

**Add to TwoMediaVanillaModelConfig:**
```python
@dataclass
class TwoMediaVanillaModelConfig(ModelConfig):
    # ... existing params ...

    max_water_distance: float = -1.0
    """Maximum sampling distance in water medium.
    If > 0, clips scene_box intersection to this value.
    If <= 0, uses only scene_box intersection (recommended).
    Typical values: 3.0-5.0 for conservative clipping."""
```

### Phase 4: Validation & Logging

**Add diagnostic output in `get_metrics_dict`:**
```python
def get_metrics_dict(self, outputs, batch):
    metrics = super().get_metrics_dict(outputs, batch)

    # Add water sampling stats
    if "water_sample_count" in outputs:
        metrics["water_samples_per_ray"] = outputs["water_sample_count"].mean()

    if "water_max_depth" in outputs:
        metrics["water_max_depth"] = outputs["water_max_depth"].mean()

    return metrics
```

**Track in get_outputs:**
```python
# After water sampling
if hits_water.any():
    # ... existing code ...

    # Store stats for metrics
    outputs["water_sample_count"] = torch.sum(hits_water)
    outputs["water_max_depth"] = remaining_dist[hits_water].max() if hits_water.any() else 0.0
```

### Phase 5: Testing

**Test Script:** `test_water_sampling.py`

```python
"""Test water sampling with scene_box clipping."""
import torch
from nerfstudio.models.two_media_vanilla_nerf import TwoMediaNeRFModel, TwoMediaVanillaModelConfig
from nerfstudio.data.scene_box import SceneBox

# Create model with small scene
config = TwoMediaVanillaModelConfig(
    water_surface_height_model=0.0,
    max_water_distance=-1.0,  # Use only scene_box
)

scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]))

# Test: Do water samples stay within scene_box?
# ...
```

**Integration Test:**
```bash
# Short training run
ns-train two-media-vanilla-nerf --data data/test_scene \
    --max-num-iterations 1000 \
    --pipeline.model.max_water_distance -1.0

# Check logs for:
# - "Water sampling stats" message
# - water_samples_per_ray metric
# - PSNR improvement vs old version
```

## Erwartete Verbesserungen

### Vor (Buggy):
- 62.5% Samples außerhalb scene_box verschwendet
- Niedrige Wasser-Gewichte (weights_water ≈ 0)
- Verwaschenes Rendering
- Ineffiziente GPU-Nutzung

### Nach (Scene Box Clipping):
- 100% Samples innerhalb scene_box
- Höhere Wasser-Gewichte (weights_water > 0)
- Schärferes Rendering mit Unterwasser-Details
- Bessere GPU-Effizienz

### Metrics:
- **PSNR:** Erwarte +2-5 dB Verbesserung
- **Training Speed:** ~20% schneller (weniger verschwendete Samples)
- **Accumulation:** Höhere Werte (weniger Background Blending)
- **Memory:** Leicht reduziert (effizienteres Sampling)

## Risiken & Fallbacks

### Risiko 1: AABB intersection Bug
**Symptom:** Rendering komplett schwarz oder alle Strahlen invalid

**Debug:**
```python
# Add assertion in _intersect_scene_box
assert (t_far >= t_near).all(), "Invalid AABB intersection!"
assert (t_far > 0).any(), "No valid water rays!"
```

**Fallback:** Nutze `max_water_distance` Parameter statt AABB

### Risiko 2: Scene box zu klein
**Symptom:** Wasser-Samples zu kurz, Details fehlen

**Debug:**
```python
CONSOLE.log(f"Scene box: {self.scene_box.aabb}")
CONSOLE.log(f"Water sampling mean: {remaining_dist.mean()}")
```

**Fix:** User kann `max_water_distance` größer setzen als scene_box

### Risiko 3: Koordinatensystem-Probleme
**Symptom:** Wasser-Samples an falscher Position

**Debug:**
```python
# Visualize water ray endpoints in viewer
water_endpoints = water_origins + refracted_dirs * remaining_dist.unsqueeze(-1)
# Check if they're within scene_box
within = self.scene_box.within(water_endpoints)
CONSOLE.log(f"Water endpoints within scene: {within.float().mean():.1%}")
```

## Zeitplan

1. **Phase 1-2** (AABB + Refactor): 1-2 Stunden
2. **Phase 3** (Config): 15 Minuten
3. **Phase 4** (Logging): 30 Minuten
4. **Phase 5** (Testing): 1-2 Stunden

**Total:** 3-5 Stunden Development + Testing

## Fazit

Die ursprünglich vorgeschlagene "Projektion-Fix" löst das Problem **NICHT**, da beide Varianten gleich ineffizient sind (62.5% waste).

Die **wahre Lösung** ist scene_box-aware sampling via AABB intersection, was:
- ✅ 100% Effizienz erreicht
- ✅ Automatisch mit jeder Scene funktioniert
- ✅ Keine manuelle Tuning erfordert
- ✅ Mit nerfstudio's Architektur konsistent ist

**Empfehlung:** Implementiere AABB intersection als primäre Methode, mit optionalem `max_water_distance` als Safety-Parameter.
