# Wasseroberflächen-Koordinatensystem: Tiefenanalyse & Lösung

## 🔴 **Das Problem**

**Symptom:** Wasseroberfläche wird falsch transformiert, in Plotly manchmal gar nicht dargestellt

**Root Cause:** Komplexe Koordinatensystem-Transformationen zwischen Agisoft Metashape und Nerfstudio

---

## 📐 **Koordinatensystem-Hierarchie**

Es gibt **DREI** verschiedene Koordinatensysteme:

```
1. METASHAPE WORLD (from cameras.xml)
   ↓ [applied_transform]
2. NERFSTUDIO WORLD (after dataparser)
   ↓ [dataparser_transform + dataparser_scale]
3. NERFSTUDIO MODEL (after scene normalization)
```

### **1. Metashape World Coordinates**

**Konvention (OpenCV):**
- Kamera schaut nach **-Z**
- +X ist **rechts**
- +Y ist **oben**
- +Z ist **vorne** (aus der Kamera heraus)

**Wasseroberfläche in Metashape:**
- Horizontal Ebene: `z = water_height`
- Normal: `[0, 0, 1]` (nach oben)
- Beispiel: `water_height = 0.0` → Ebene bei z=0

---

### **2. Nerfstudio World Coordinates**

**Transformation:** `metashape_to_json()` in `metashape_utils.py:189-191`

```python
# Line 189: Permutation der Zeilen
transform = transform[[2, 0, 1, 3], :]  # [Z, X, Y, W] statt [X, Y, Z, W]

# Line 191: Spiegelung von Y und Z
transform[:, 1:3] *= -1
```

**Applied Transform (Line 196-198):**
```python
applied_transform = np.eye(4)[:3, :]
applied_transform = applied_transform[np.array([2, 0, 1]), :]

# Resultat:
[[0, 0, 1],   # Metashape Z → Nerfstudio X
 [1, 0, 0],   # Metashape X → Nerfstudio Y
 [0, 1, 0]]   # Metashape Y → Nerfstudio Z
```

**Was bedeutet das für die Wasseroberfläche?**

Metashape: `z = h` → Ebene normal `[0, 0, 1]`, Punkt `[0, 0, h]`

Nach Applied Transform:
```
Point: [x, y, z]_metashape → [z, x, y]_nerfstudio
       [0, 0, h]           → [h, 0, 0]

Normal: applied_transform @ [0, 0, 1]
      = [[0, 0, 1],    [0]     [1]
         [1, 0, 0],  @  [0]  =  [0]
         [0, 1, 0]]     [1]     [0]
```

**Resultat:**
- Metashape horizontale Ebene (z = h) → Nerfstudio **VERTIKALE** Ebene (x = h)!
- Normal [0, 0, 1] → [1, 0, 0] (zeigt jetzt in X-Richtung!)

**❌ Das ist das PROBLEM!**

---

### **3. Nerfstudio Model Coordinates**

**Transformation:** `dataparser_transform` + `dataparser_scale` (von DataParser berechnet)

Zweck:
- Zentriert die Szene
- Skaliert sie in eine normalisierte Box
- Rotiert für bessere Orientierung

**Diese Transform wird von TwoMediaNeRF in `_setup_water_interface()` verwendet**

---

## 🔍 **Aktueller Code-Analyse**

### **Problem 1: Inkonsistente Referenz-Koordinatensysteme**

```python
# Line 196 in two_media_vanilla_nerf.py:
point_world = torch.tensor([0.0, 0.0, self.config.water_surface_height_world])
```

**Was ist `water_surface_height_world`?**

| Interpretation | Koordinatensystem | Korrekt? |
|----------------|-------------------|----------|
| Metashape z-Koordinate | Metashape World | ❌ Nein - wird nicht korrekt transformiert |
| Nerfstudio world z | Nerfstudio World | ❌ Unklar - applied_transform fehlt |
| Nerfstudio model z | Model Space | ❌ Nein - Transformation wird doppelt angewendet |

**Das Problem:** User gibt Metashape z-Wert ein, aber Code behandelt es als wäre es bereits in Nerfstudio World!

---

### **Problem 2: Falsche Normal-Transformation**

```python
# Line 202-203:
R = transform[:3, :3].float()  # dataparser_transform Rotation
normal_model = R @ normal_world
```

**`normal_world` ist falsch definiert:**
```python
# Line 195:
normal_world = torch.tensor([0.0, 0.0, 1.0])  # Annahme: z zeigt nach oben
```

Aber nach `applied_transform` ist die Wasseroberflächen-Normal **[1, 0, 0]**, nicht [0, 0, 1]!

---

### **Problem 3: Point-Transformation**

```python
# Line 205-206:
point_model = scale * (R @ point_world + t)
```

`point_world = [0, 0, water_height]` ist **Metashape coordinates**, wird aber behandelt als wäre es schon Nerfstudio World!

**Korrekt wäre:**
1. Metashape → Nerfstudio World (via applied_transform)
2. Nerfstudio World → Model (via dataparser_transform)

---

## ✅ **Die Lösung: 3-Stufige Strategie**

### **Strategie 1: Explizite Koordinatensystem-Angabe (EMPFOHLEN)**

**Idee:** User gibt **explizit** an in welchem Koordinatensystem die Wasseroberfläche definiert ist.

**Config:**
```python
@dataclass
class TwoMediaVanillaModelConfig(ModelConfig):
    # Option A: Metashape World Coordinates
    water_surface_height_metashape: Optional[float] = None
    """Water surface z-coordinate in Agisoft Metashape coordinate system.
    This will be automatically transformed through applied_transform and dataparser_transform."""

    # Option B: Nerfstudio World Coordinates (after applied_transform)
    water_surface_height_world: Optional[float] = None
    """Water surface z-coordinate in Nerfstudio world coordinates (after applied_transform).
    This will be transformed through dataparser_transform only."""

    # Option C: Model Coordinates (direct, no transformation)
    water_surface_height_model: Optional[float] = None
    """Water surface z-coordinate directly in model coordinates.
    No transformation will be applied. Use this if you already know the model-space height."""
```

**Implementierung:**
```python
def _setup_water_interface(self) -> None:
    """Compute water surface plane in model coordinates."""

    # Priorität: model > world > metashape

    if self.config.water_surface_height_model is not None:
        # OPTION C: Direct model coordinates (no transform)
        self._setup_water_direct_model(self.config.water_surface_height_model)
        return

    if self.config.water_surface_height_world is not None:
        # OPTION B: Nerfstudio world → model
        self._setup_water_from_nerfstudio_world(self.config.water_surface_height_world)
        return

    if self.config.water_surface_height_metashape is not None:
        # OPTION A: Metashape → nerfstudio world → model
        self._setup_water_from_metashape(self.config.water_surface_height_metashape)
        return

    raise ValueError("No water surface height specified! Use one of: "
                     "water_surface_height_model, water_surface_height_world, "
                     "or water_surface_height_metashape")

def _setup_water_direct_model(self, height: float):
    """Water surface directly in model coordinates (no transformation)."""
    self.register_buffer("water_plane_normal", torch.tensor([[0.0, 0.0, 1.0]]))
    d = -height  # z = height → 0*x + 0*y + 1*z + d = 0 → d = -height
    self.register_buffer("water_plane_d", torch.tensor([d]))
    CONSOLE.log(f"[cyan]Water surface at z={height:.4f} (model coords, direct)[/cyan]")

def _setup_water_from_nerfstudio_world(self, height: float):
    """Transform from nerfstudio world to model coordinates."""
    dataparser_transform = self.kwargs.get("dataparser_transform")
    dataparser_scale = float(self.kwargs.get("dataparser_scale", 1.0))

    if dataparser_transform is None:
        # No transformation, nerfstudio world = model
        self._setup_water_direct_model(height)
        return

    # Nerfstudio world: horizontal plane at z = height
    normal_world = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    point_world = torch.tensor([0.0, 0.0, height], dtype=torch.float32)

    # Transform to model coordinates
    R = dataparser_transform[:3, :3].float()
    t = dataparser_transform[:3, 3].float()

    normal_model = R @ normal_world
    normal_model = F.normalize(normal_model, dim=0)

    point_model = dataparser_scale * (R @ point_world + t)

    d = -torch.dot(normal_model, point_model)

    self.register_buffer("water_plane_normal", normal_model.unsqueeze(0))
    self.register_buffer("water_plane_d", torch.tensor([d]))

    self._log_water_surface_position(normal_model, d)

def _setup_water_from_metashape(self, height: float):
    """Transform from Metashape coordinates through applied_transform and dataparser_transform."""

    # Step 1: Get applied_transform from dataparser
    applied_transform = self.kwargs.get("applied_transform")

    if applied_transform is None:
        # Fallback: Standard Metashape applied_transform
        # From metashape_utils.py:196-198
        applied_transform = torch.eye(4)[:3, :]
        applied_transform = applied_transform[torch.tensor([2, 0, 1]), :]
        CONSOLE.log("[yellow]No applied_transform found, using standard Metashape convention[/yellow]")
    else:
        applied_transform = torch.tensor(applied_transform, dtype=torch.float32)

    # Step 2: Metashape coordinates
    # Horizontal plane: z = height, normal [0, 0, 1]
    point_metashape = torch.tensor([0.0, 0.0, height], dtype=torch.float32)
    normal_metashape = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    # Step 3: Transform to nerfstudio world
    # Point: p_world = applied_transform @ [p; 1]
    point_metashape_homo = torch.cat([point_metashape, torch.ones(1)])
    point_world = (applied_transform @ point_metashape_homo)[:3]

    # Normal: n_world = applied_transform[:3, :3] @ n_metashape
    normal_world = applied_transform[:3, :3] @ normal_metashape
    normal_world = F.normalize(normal_world, dim=0)

    # Step 4: Transform to model (same as _setup_water_from_nerfstudio_world)
    dataparser_transform = self.kwargs.get("dataparser_transform")
    dataparser_scale = float(self.kwargs.get("dataparser_scale", 1.0))

    if dataparser_transform is None:
        # No further transformation
        normal_model = normal_world
        point_model = point_world
    else:
        R = dataparser_transform[:3, :3].float()
        t = dataparser_transform[:3, 3].float()

        normal_model = R @ normal_world
        normal_model = F.normalize(normal_model, dim=0)

        point_model = dataparser_scale * (R @ point_world + t)

    d = -torch.dot(normal_model, point_model)

    self.register_buffer("water_plane_normal", normal_model.unsqueeze(0))
    self.register_buffer("water_plane_d", torch.tensor([d]))

    CONSOLE.log(f"[cyan]Water surface transformed from Metashape z={height:.4f}[/cyan]")
    self._log_water_surface_position(normal_model, d)

def _log_water_surface_position(self, normal_model: torch.Tensor, d: float):
    """Log the resulting water surface position and orientation."""
    if abs(normal_model[2]) > 1e-6:
        water_z_model = -d / normal_model[2]
        cosine = abs(normal_model[2]).clamp(-1.0, 1.0)
        angle_deg = math.degrees(torch.acos(cosine).item())

        if angle_deg > 5:
            CONSOLE.log(f"[yellow]WARNING: Water surface not horizontal![/yellow]")
            CONSOLE.log(f"  Model coords: z={water_z_model:.4f}")
            CONSOLE.log(f"  Normal: {normal_model.tolist()}")
            CONSOLE.log(f"  Angle from horizontal: {angle_deg:.1f}°")
            CONSOLE.log(f"[yellow]This may indicate coordinate system issues![/yellow]")
        else:
            CONSOLE.log(f"  Model coords: z={water_z_model:.4f}, angle={angle_deg:.2f}°")
    else:
        CONSOLE.log(f"[red]ERROR: Water surface is VERTICAL in model space![/red]")
        CONSOLE.log(f"  Normal: {normal_model.tolist()}")
        CONSOLE.log(f"[red]Check coordinate system transformations![/red]")
```

---

### **Strategie 2: Automatische Metashape-Erkennung**

**Idee:** Erkenne automatisch ob Daten von Metashape kommen und wende korrekte Transformation an.

```python
def _setup_water_interface(self) -> None:
    """Auto-detect coordinate system and apply correct transformation."""

    # Check if we have Metashape data
    applied_transform = self.kwargs.get("applied_transform")
    is_metashape = applied_transform is not None and \
                   self._is_metashape_transform(applied_transform)

    height = self.config.water_surface_height_world  # or whatever the user provides

    if is_metashape:
        CONSOLE.log("[cyan]Metashape data detected, applying Metashape→Nerfstudio transformation[/cyan]")
        self._setup_water_from_metashape(height)
    else:
        CONSOLE.log("[cyan]Standard data, using direct transformation[/cyan]")
        self._setup_water_from_nerfstudio_world(height)

def _is_metashape_transform(self, transform) -> bool:
    """Check if transform matches Metashape's applied_transform pattern."""
    expected = torch.tensor([[0, 0, 1],
                             [1, 0, 0],
                             [0, 1, 0]], dtype=torch.float32)

    transform_tensor = torch.tensor(transform, dtype=torch.float32)
    if transform_tensor.shape == (3, 4):
        transform_3x3 = transform_tensor[:3, :3]
    else:
        transform_3x3 = transform_tensor

    return torch.allclose(transform_3x3, expected, atol=1e-5)
```

---

### **Strategie 3: Visualisierungs-Tool**

**Idee:** Tool um Wasseroberfläche zu visualisieren und manuell anzupassen.

```python
def visualize_water_surface(
    dataparser_outputs,
    water_height_metashape: float,
    output_path: Path
):
    """Visualize water surface in different coordinate systems."""

    import plotly.graph_objects as go

    # Camera positions
    camera_to_worlds = dataparser_outputs.cameras.camera_to_worlds
    cam_positions = camera_to_worlds[:, :3, 3].numpy()

    # Create meshgrid for water surface
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)

    # Water surface in different coordinate systems
    surfaces = []

    # 1. Metashape coords
    Z_metashape = np.ones_like(X) * water_height_metashape
    surfaces.append(("Metashape", X, Y, Z_metashape))

    # 2. After applied_transform
    if dataparser_outputs.applied_transform is not None:
        # Transform grid points
        # ... apply transformation
        surfaces.append(("Nerfstudio World", X_world, Y_world, Z_world))

    # 3. After dataparser_transform
    if dataparser_outputs.dataparser_transform is not None:
        # ... apply transformation
        surfaces.append(("Model Space", X_model, Y_model, Z_model))

    # Create figure
    fig = go.Figure()

    # Add cameras
    fig.add_trace(go.Scatter3d(
        x=cam_positions[:, 0],
        y=cam_positions[:, 1],
        z=cam_positions[:, 2],
        mode='markers',
        name='Cameras',
        marker=dict(size=3, color='red')
    ))

    # Add water surfaces
    for name, X, Y, Z in surfaces:
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            name=name,
            opacity=0.5
        ))

    fig.write_html(output_path)
    CONSOLE.log(f"Saved visualization to {output_path}")
```

---

## 🎯 **Empfohlene Implementierung**

**Phase 1: Sofort (Fix das Problem)**
1. ✅ Implementiere `_setup_water_from_metashape()` mit korrekter applied_transform
2. ✅ Füge `water_surface_height_metashape` Config-Parameter hinzu
3. ✅ Deprecate `water_surface_height_world` (unklar welches "world")

**Phase 2: Kurzfristig (Usability)**
1. ✅ Auto-Erkennung von Metashape Daten
2. ✅ Besseres Logging (zeige alle 3 Koordinatensysteme)
3. ✅ Validation (warne bei verdächtigen Werten)

**Phase 3: Mittelfristig (Tools)**
1. ✅ Visualisierungs-Tool
2. ✅ Interactive water surface adjustment
3. ✅ Dokumentation mit Beispielen

---

## 📋 **Testing-Checklist**

- [ ] Metashape Daten mit bekannter Wasseroberfläche
- [ ] Visualisierung zeigt Ebene korrekt
- [ ] Water rays treffen Wasseroberfläche
- [ ] Refraktion funktioniert
- [ ] Logging zeigt alle Transformationen
- [ ] Vergleich mit Ground Truth

---

## 🔬 **Debugging-Commands**

```python
# In Python after loading model:
model.water_plane_normal  # Should be close to [0, 0, 1] for horizontal
model.water_plane_d       # z-position = -d

# Check transforms:
dataparser_outputs.applied_transform
dataparser_outputs.dataparser_transform
dataparser_outputs.dataparser_scale

# Manual test:
metashape_point = torch.tensor([0., 0., 5.0])  # z=5 in Metashape
# Apply transforms manually and check if it matches model.water_plane_d
```
