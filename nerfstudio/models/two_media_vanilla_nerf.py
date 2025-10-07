# Copyright 2024 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Two-medium extension of the vanilla NeRF model.

Simplified approach using stratified sampling instead of occupancy grids.
This avoids memory issues and correctly handles air/water separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from nerfstudio.utils.math import intersect_aabb
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class TwoMediaVanillaModelConfig(ModelConfig):
    """Two-media Vanilla NeRF with stratified sampling (no occupancy grid)."""

    _target: Type = field(default_factory=lambda: TwoMediaNeRFModel)

    # Interface parameters
    water_surface_height_metashape: Optional[float] = None
    """Water surface z-coordinate in Agisoft Metashape coordinate system.
    Use this when working with Metashape data (cameras.xml). The height will be
    automatically transformed through applied_transform and dataparser_transform.
    Example: If water is at z=5.0 in Metashape, set this to 5.0."""

    water_surface_height_world: Optional[float] = None
    """Water surface z-coordinate in Nerfstudio world coordinates (after applied_transform, before dataparser_transform).
    Use this for data processed with nerfstudio's standard pipeline.
    Note: For Metashape data, use water_surface_height_metashape instead."""

    water_surface_height_model: Optional[float] = None
    """Water surface z-coordinate directly in model coordinates (after all transformations).
    Use this if you know the exact height in the normalized model space.
    This bypasses all coordinate transformations."""

    air_refractive_index: float = 1.0
    """Refractive index of air."""
    water_refractive_index: float = 1.333
    """Refractive index of water."""
    interface_epsilon: float = 1e-4
    """Small offset around interface to avoid numerical issues."""

    # Sampling parameters (matching vanilla_nerf classic)
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation per ray segment."""
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation per ray segment."""
    eval_num_rays_per_chunk: int = 256
    """Rays per chunk during evaluation (low for memory safety)."""

    # Rendering
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Background color."""

    # Optional features aligned with classic vanilla NeRF
    enable_temporal_distortion: bool = False
    """Specifies whether to include temporal distortion."""
    temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"kind": TemporalDistortionKind.DNERF})
    """Parameters for temporal distortion instantiation."""
    use_gradient_scaling: bool = False
    """Apply distance-based gradient scaling."""


class TwoMediaNeRFModel(Model):
    """Two-medium NeRF using stratified sampling for robust memory-safe training."""

    config: TwoMediaVanillaModelConfig

    def __init__(
        self,
        config: TwoMediaVanillaModelConfig,
        **kwargs,
    ) -> None:
        self.air_field_coarse: Optional[NeRFField] = None
        self.air_field_fine: Optional[NeRFField] = None
        self.water_field_coarse: Optional[NeRFField] = None
        self.water_field_fine: Optional[NeRFField] = None
        self.temporal_distortion = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self) -> None:
        """Initialize fields and samplers."""
        super().populate_modules()

        # Encodings
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        # Four separate fields: coarse and fine for both air and water (like 2× vanilla_nerf)
        self.air_field_coarse = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.air_field_fine = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.water_field_coarse = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.water_field_fine = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)

        # Samplers (classic NeRF approach)
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # Renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # Loss
        self.rgb_loss = MSELoss()

        # Metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from torchmetrics.functional import structural_similarity_index_measure

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # Temporal distortion (optional)
        self.temporal_distortion = None
        if getattr(self.config, "enable_temporal_distortion", False):
            params = dict(self.config.temporal_distortion_params)
            kind = params.pop("kind")
            self.temporal_distortion = kind.to_temporal_distortion(params)

        # Setup water interface geometry
        self._setup_water_interface()

        CONSOLE.log(
            f"TwoMediaNeRF initialized: 4 separate MLPs (air/water × coarse/fine), "
            f"{self.config.num_coarse_samples} coarse + {self.config.num_importance_samples} fine samples per segment"
        )

    def _setup_water_interface(self) -> None:
        """Compute water surface plane in model coordinates.

        Supports three coordinate systems (priority: model > world > metashape):
        1. water_surface_height_model: Direct model coordinates (no transformation)
        2. water_surface_height_world: Nerfstudio world → model (via dataparser_transform)
        3. water_surface_height_metashape: Metashape → world → model (via applied_transform + dataparser_transform)
        """

        # Priority 1: Direct model coordinates (no transformation needed)
        if self.config.water_surface_height_model is not None:
            self._setup_water_direct_model(self.config.water_surface_height_model)
            return

        # Priority 2: Nerfstudio world coordinates (apply dataparser_transform)
        if self.config.water_surface_height_world is not None:
            self._setup_water_from_nerfstudio_world(self.config.water_surface_height_world)
            return

        # Priority 3: Metashape coordinates (apply applied_transform + dataparser_transform)
        if self.config.water_surface_height_metashape is not None:
            self._setup_water_from_metashape(self.config.water_surface_height_metashape)
            return

        # No water surface specified - error
        raise ValueError(
            "No water surface height specified! You must set ONE of:\n"
            "  --pipeline.model.water_surface_height_metashape (for Metashape/Agisoft data)\n"
            "  --pipeline.model.water_surface_height_world (for nerfstudio-processed data)\n"
            "  --pipeline.model.water_surface_height_model (if you know model-space coordinates)"
        )

    def _setup_water_direct_model(self, height: float) -> None:
        """Setup water surface directly in model coordinates (no transformation).

        Args:
            height: z-coordinate of horizontal water surface in model space
        """
        # Horizontal plane: z = height → normal [0,0,1], plane equation: z + d = 0
        self.register_buffer("water_plane_normal", torch.tensor([[0.0, 0.0, 1.0]]), persistent=False)
        d = -height
        self.register_buffer("water_plane_d", torch.tensor([d]), persistent=False)
        self.register_buffer("water_plane_offset", torch.tensor([d]), persistent=False)

        CONSOLE.log(f"[cyan]━━━ Water Surface (Direct Model Coords) ━━━[/cyan]")
        CONSOLE.log(f"  Height (model): z = {height:.4f}")
        CONSOLE.log(f"  Normal: [0, 0, 1] (horizontal)")
        CONSOLE.log(f"  Plane equation: z + {d:.4f} = 0")

    def _setup_water_from_nerfstudio_world(self, height: float) -> None:
        """Transform water surface from Nerfstudio world to model coordinates.

        Args:
            height: z-coordinate of water surface in Nerfstudio world coordinates
        """
        dataparser_transform = self.kwargs.get("dataparser_transform")
        dataparser_scale = float(self.kwargs.get("dataparser_scale", 1.0))

        if dataparser_transform is None:
            # No transformation - world = model
            CONSOLE.log("[yellow]No dataparser_transform found, world = model coordinates[/yellow]")
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

        # Plane equation: n·x + d = 0, where d = -n·p
        d = -torch.dot(normal_model, point_model)

        self.register_buffer("water_plane_normal", normal_model.unsqueeze(0), persistent=False)
        self.register_buffer("water_plane_d", torch.tensor([d]), persistent=False)
        self.register_buffer("water_plane_offset", torch.tensor([d]), persistent=False)

        self._log_water_surface_result(
            "Nerfstudio World",
            height,
            normal_model,
            d,
            dataparser_scale
        )

    def _setup_water_from_metashape(self, height: float) -> None:
        """Transform water surface from Metashape to model coordinates.

        Applies two transformations:
        1. Metashape → Nerfstudio World (via applied_transform)
        2. Nerfstudio World → Model (via dataparser_transform + scale)

        Args:
            height: z-coordinate of water surface in Metashape coordinate system
        """
        # Step 1: Get applied_transform
        applied_transform_data = self.kwargs.get("applied_transform")

        if applied_transform_data is None:
            # Use standard Metashape convention (from metashape_utils.py:196-198)
            CONSOLE.log("[yellow]No applied_transform found, using standard Metashape→Nerfstudio convention[/yellow]")
            applied_transform = torch.eye(4, dtype=torch.float32)
            applied_transform[:3, :] = applied_transform[:3, :][[2, 0, 1], :]
        else:
            applied_transform = torch.tensor(applied_transform_data, dtype=torch.float32)
            # Handle [3, 4] or [4, 4] format
            if applied_transform.shape == (3, 4):
                temp = torch.eye(4, dtype=torch.float32)
                temp[:3, :] = applied_transform
                applied_transform = temp

        # Step 2: Metashape coordinates
        # Horizontal plane: z = height, normal [0, 0, 1]
        point_metashape = torch.tensor([0.0, 0.0, height], dtype=torch.float32)
        normal_metashape = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

        # Step 3: Transform to Nerfstudio world
        # Point: p_world = applied_transform @ [p; 1]
        point_metashape_homo = torch.cat([point_metashape, torch.ones(1)])
        point_world = (applied_transform @ point_metashape_homo)[:3]

        # Normal: n_world = applied_transform[:3, :3] @ n_metashape
        normal_world = applied_transform[:3, :3] @ normal_metashape
        normal_world = F.normalize(normal_world, dim=0)

        # Step 4: Transform to model
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

        # Plane equation: n·x + d = 0
        d = -torch.dot(normal_model, point_model)

        self.register_buffer("water_plane_normal", normal_model.unsqueeze(0), persistent=False)
        self.register_buffer("water_plane_d", torch.tensor([d]), persistent=False)
        self.register_buffer("water_plane_offset", torch.tensor([d]), persistent=False)

        # Detailed logging for Metashape
        CONSOLE.log(f"[cyan]━━━ Water Surface (Metashape → Model) ━━━[/cyan]")
        CONSOLE.log(f"  Input (Metashape): z = {height:.4f}")
        CONSOLE.log(f"  After applied_transform (World): {point_world.tolist()}")
        CONSOLE.log(f"  Normal after applied_transform: {normal_world.tolist()}")
        self._log_water_surface_result(
            "Model (final)",
            None,
            normal_model,
            d,
            dataparser_scale,
            show_header=False
        )

    def _log_water_surface_result(
        self,
        coord_system: str,
        input_height: Optional[float],
        normal_model: torch.Tensor,
        d: float,
        scale: float,
        show_header: bool = True
    ) -> None:
        """Log the resulting water surface position and check for issues.

        Args:
            coord_system: Name of input coordinate system
            input_height: Input height (if applicable)
            normal_model: Computed normal in model space
            d: Plane equation constant
            scale: Dataparser scale factor
        """
        if show_header:
            CONSOLE.log(f"[cyan]━━━ Water Surface ({coord_system} → Model) ━━━[/cyan]")
            if input_height is not None:
                CONSOLE.log(f"  Input ({coord_system}): z = {input_height:.4f}")

        # Check if normal has significant z-component (indicates horizontal-ish plane)
        if abs(normal_model[2]) > 1e-6:
            # Solve for z when x=0, y=0: n_z * z + d = 0 → z = -d/n_z
            water_z_model = -d / normal_model[2]

            # Check angle from horizontal
            cosine = abs(normal_model[2]).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cosine).item()
            angle_deg = math.degrees(angle_rad)

            CONSOLE.log(f"  Output (Model): z ≈ {water_z_model:.4f}")
            CONSOLE.log(f"  Normal: [{normal_model[0]:.4f}, {normal_model[1]:.4f}, {normal_model[2]:.4f}]")
            CONSOLE.log(f"  Angle from horizontal: {angle_deg:.2f}°")

            if angle_deg > 5:
                CONSOLE.log(f"[yellow]  ⚠ WARNING: Water surface is tilted {angle_deg:.1f}° from horizontal![/yellow]")
                CONSOLE.log(f"[yellow]  This may indicate coordinate system issues.[/yellow]")
                CONSOLE.log(f"[yellow]  Consider using water_surface_height_model for exact horizontal plane.[/yellow]")
            else:
                CONSOLE.log(f"[green]  ✓ Water surface is horizontal (angle < 5°)[/green]")
        else:
            # Normal is nearly horizontal → plane is nearly vertical
            CONSOLE.log(f"[red]  ✗ ERROR: Water surface is VERTICAL in model space![/red]")
            CONSOLE.log(f"  Normal: {normal_model.tolist()}")
            CONSOLE.log(f"[red]  This indicates a coordinate system transformation error![/red]")
            CONSOLE.log(f"[red]  Check your coordinate system parameters and transforms.[/red]")

        CONSOLE.log(f"  Plane equation: {normal_model[0]:.4f}*x + {normal_model[1]:.4f}*y + {normal_model[2]:.4f}*z + {d:.4f} = 0")
        CONSOLE.log(f"  Dataparser scale: {scale:.4f}")

    def _signed_distance_to_water(self, positions: Tensor) -> Tensor:
        """Compute signed distance to water surface. Positive = above water (air)."""
        # Plane equation: n·x + d = 0
        # Signed distance: (n·x + d) / |n| (normal is already normalized)
        return (positions @ self.water_plane_normal.T).squeeze(-1) + self.water_plane_d

    def _compute_refraction(self, incident_dirs: Tensor) -> Tensor:
        """Compute refracted ray directions at water surface (air → water)."""
        normal = F.normalize(self.water_plane_normal, dim=-1)
        incident = F.normalize(incident_dirs, dim=-1)

        # Ensure normal points from air to water (towards incident ray)
        cos_i = (incident * normal).sum(dim=-1, keepdim=True)
        normal = torch.where(cos_i > 0, -normal, normal)
        cos_i = -(incident * normal).sum(dim=-1, keepdim=True).clamp(min=0.0)

        # Snell's law
        eta = self.config.air_refractive_index / self.config.water_refractive_index
        sin2_t = eta**2 * (1.0 - cos_i**2)

        # Total internal reflection check
        if (sin2_t > 1.0).any():
            CONSOLE.log("[yellow]Warning: Total internal reflection detected (shouldn't happen air→water)[/yellow]")
            sin2_t = sin2_t.clamp(max=1.0)

        cos_t = torch.sqrt(1.0 - sin2_t)
        refracted = eta * incident + (eta * cos_i - cos_t) * normal

        return F.normalize(refracted, dim=-1)

    def _intersect_scene_box(self, origins: Tensor, directions: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute intersection of rays with scene bounding box.

        This ensures water rays only sample within the valid scene geometry,
        matching the behavior of vanilla NeRF where Camera automatically clips
        rays to the scene_box via AABB intersection.

        Args:
            origins: [N, 3] ray origins
            directions: [N, 3] ray directions (normalized)

        Returns:
            t_near: [N] distance to box entry (0 if inside box)
            t_far: [N] distance to box exit
        """
        device = origins.device

        # scene_box.aabb is [2, 3], but intersect_aabb expects [6] (flattened)
        # Format: [x_min, y_min, z_min, x_max, y_max, z_max]
        aabb_flat = self.scene_box.aabb.flatten().to(device)

        # Use nerfstudio's standard AABB intersection (same as Camera uses)
        t_near, t_far = intersect_aabb(origins, directions, aabb_flat)

        return t_near, t_far

    def _apply_temporal_distortion(self, ray_samples: Optional[RaySamples]) -> None:
        """Apply temporal distortion offsets in-place if enabled."""
        if self.temporal_distortion is None or ray_samples is None:
            return

        offsets = None
        if ray_samples.times is not None:
            offsets = self.temporal_distortion(ray_samples.frustums.get_positions(), ray_samples.times)
        ray_samples.frustums.set_offsets(offsets)

    def _render_segment(
        self,
        field: NeRFField,
        ray_samples: RaySamples,
    ) -> Tuple[Tensor, Tensor]:
        """
        Render a ray segment through a field.

        Returns:
            weights: (num_rays, num_samples, 1) rendering weights
            rgb: (num_rays, num_samples, 3) RGB colors
        """
        field_outputs = field(ray_samples)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        densities = field_outputs[FieldHeadNames.DENSITY]  # (num_rays, num_samples, 1)
        rgb = field_outputs[FieldHeadNames.RGB]  # (num_rays, num_samples, 3)

        weights = ray_samples.get_weights(densities)

        return weights, rgb

    def _expected_depth(
        self,
        weights: Tensor,
        accumulation: Tensor,
        ray_samples_air: RaySamples,
        ray_samples_water: Optional[RaySamples],
        t_water: Tensor,
    ) -> Tensor:
        """Compute expected depth combining air and water segments."""

        t_mids_air = (ray_samples_air.frustums.starts + ray_samples_air.frustums.ends) / 2.0

        if ray_samples_water is not None:
            t_mids_water = (ray_samples_water.frustums.starts + ray_samples_water.frustums.ends) / 2.0
            t_mids_water = t_mids_water + t_water.view(-1, 1, 1)
        else:
            t_mids_water = torch.zeros(
                (weights.shape[0], 0, 1), device=weights.device, dtype=weights.dtype
            )

        t_mids_all = torch.cat([t_mids_air, t_mids_water], dim=-2)
        depth = (weights * t_mids_all).sum(dim=-2) / (accumulation + 1e-10)
        return depth

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Tensor]:
        """Forward pass with hierarchical sampling through two media."""

        ray_bundle = ray_bundle.flatten()
        num_rays = len(ray_bundle)
        device = ray_bundle.origins.device

        origins = ray_bundle.origins
        directions = F.normalize(ray_bundle.directions, dim=-1)

        near = ray_bundle.nears[..., 0] if ray_bundle.nears is not None else torch.zeros(num_rays, device=device)
        far = ray_bundle.fars[..., 0] if ray_bundle.fars is not None else torch.full((num_rays,), 10.0, device=device)

        eps = self.config.interface_epsilon
        signed_dist = self._signed_distance_to_water(origins)
        denom = (directions @ self.water_plane_normal.T).squeeze(-1)

        t_water = torch.where(
            torch.abs(denom) > 1e-6,
            -signed_dist / denom,
            far + 1.0,
        )

        starts_above = signed_dist > eps
        hits_water = (t_water > near) & (t_water < far) & starts_above

        # Coarse sampling (air)
        t_air_far = torch.where(hits_water, t_water - eps, far)
        t_air_far = torch.clamp(t_air_far, min=near + eps)
        air_fars = torch.maximum(t_air_far, near + eps)

        air_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=ray_bundle.pixel_area,
            camera_indices=ray_bundle.camera_indices,
            nears=near.unsqueeze(-1),
            fars=air_fars.unsqueeze(-1),
            metadata=ray_bundle.metadata,
            times=ray_bundle.times,
        )
        ray_samples_air_coarse = self.sampler_uniform(
            air_bundle, num_samples=self.config.num_coarse_samples
        )
        self._apply_temporal_distortion(ray_samples_air_coarse)
        weights_air_coarse, rgb_air_coarse = self._render_segment(
            self.air_field_coarse, ray_samples_air_coarse
        )

        # Coarse sampling (water)
        water_bundle: Optional[RayBundle] = None
        ray_samples_water_coarse: Optional[RaySamples] = None
        weights_water_coarse = torch.zeros((num_rays, 0, 1), device=device)
        rgb_water_coarse = torch.zeros((num_rays, 0, 3), device=device)

        if hits_water.any():
            entry_points = origins + directions * t_water.unsqueeze(-1)
            refracted_dirs = self._compute_refraction(directions)
            water_origins = entry_points + refracted_dirs * eps

            # FIX: Use scene_box AABB intersection instead of (far - t_water)
            # This matches vanilla NeRF behavior where Camera automatically clips rays to scene_box.
            # Previously: remaining_dist = far - t_water caused 60%+ samples outside scene!
            t_near_water, t_far_water = self._intersect_scene_box(water_origins, refracted_dirs)
            remaining_dist = t_far_water

            # Log sampling statistics (once per training session)
            if self.training and not hasattr(self, '_logged_water_sampling'):
                valid_mask = remaining_dist > 0
                if valid_mask.any():
                    old_method = torch.clamp(far - t_water, min=0.0)
                    CONSOLE.log(
                        f"[cyan]Water sampling (scene_box AABB intersection):[/cyan]\n"
                        f"  Rays hitting water: {valid_mask.sum()}/{num_rays}\n"
                        f"  Mean sampling distance: {remaining_dist[valid_mask].mean():.4f}\n"
                        f"  Max sampling distance: {remaining_dist[valid_mask].max():.4f}\n"
                        f"  Old method (far - t_water) would give: {old_method[valid_mask].mean():.4f} mean\n"
                        f"  Scene box: {self.scene_box.aabb.tolist()}"
                    )
                self._logged_water_sampling = True

            water_bundle = RayBundle(
                origins=water_origins,
                directions=refracted_dirs,
                pixel_area=ray_bundle.pixel_area,
                camera_indices=ray_bundle.camera_indices,
                nears=torch.zeros((num_rays, 1), device=device),
                fars=remaining_dist.unsqueeze(-1),
                metadata=ray_bundle.metadata,
                times=ray_bundle.times,
            )

            ray_samples_water_coarse = self.sampler_uniform(
                water_bundle, num_samples=self.config.num_coarse_samples
            )
            self._apply_temporal_distortion(ray_samples_water_coarse)
            weights_water_coarse, rgb_water_coarse = self._render_segment(
                self.water_field_coarse, ray_samples_water_coarse
            )

            air_transmittance = 1.0 - weights_air_coarse.sum(dim=-2, keepdim=True)
            weights_water_coarse = weights_water_coarse * air_transmittance

        # Fine sampling (air)
        ray_samples_air_fine = self.sampler_pdf(
            air_bundle, ray_samples_air_coarse, weights_air_coarse.detach()
        )
        self._apply_temporal_distortion(ray_samples_air_fine)
        weights_air_fine, rgb_air_fine = self._render_segment(
            self.air_field_fine, ray_samples_air_fine
        )

        # Fine sampling (water)
        weights_water_fine = torch.zeros((num_rays, 0, 1), device=device)
        rgb_water_fine = torch.zeros((num_rays, 0, 3), device=device)
        ray_samples_water_fine: Optional[RaySamples] = None

        if hits_water.any() and water_bundle is not None and ray_samples_water_coarse is not None:
            ray_samples_water_fine = self.sampler_pdf(
                water_bundle, ray_samples_water_coarse, weights_water_coarse.detach()
            )
            self._apply_temporal_distortion(ray_samples_water_fine)
            weights_water_fine, rgb_water_fine = self._render_segment(
                self.water_field_fine, ray_samples_water_fine
            )

            air_transmittance_fine = 1.0 - weights_air_fine.sum(dim=-2, keepdim=True)
            weights_water_fine = weights_water_fine * air_transmittance_fine

        # Aggregate coarse outputs
        weights_coarse_all = torch.cat([weights_air_coarse, weights_water_coarse], dim=-2)
        rgbs_coarse = torch.cat([rgb_air_coarse, rgb_water_coarse], dim=-2)
        rgb_coarse = self.renderer_rgb(rgb=rgbs_coarse, weights=weights_coarse_all)
        accumulation_coarse = self.renderer_accumulation(weights_coarse_all)
        depth_coarse = self._expected_depth(
            weights_coarse_all,
            accumulation_coarse,
            ray_samples_air_coarse,
            ray_samples_water_coarse,
            t_water,
        )

        # Aggregate fine outputs
        weights_fine_all = torch.cat([weights_air_fine, weights_water_fine], dim=-2)
        rgbs_fine = torch.cat([rgb_air_fine, rgb_water_fine], dim=-2)
        rgb_fine = self.renderer_rgb(rgb=rgbs_fine, weights=weights_fine_all)
        accumulation_fine = self.renderer_accumulation(weights_fine_all)
        depth_fine = self._expected_depth(
            weights_fine_all,
            accumulation_fine,
            ray_samples_air_fine,
            ray_samples_water_fine,
            t_water,
        )

        return {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "rgb": rgb_fine,
            "accumulation": accumulation_fine,
            "depth": depth_fine,
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.air_field_coarse is None or self.air_field_fine is None or self.water_field_coarse is None or self.water_field_fine is None:
            raise ValueError("populate_modules() must be called before get_param_groups")
        param_groups["air_field"] = list(self.air_field_coarse.parameters()) + list(self.air_field_fine.parameters())
        param_groups["water_field"] = list(self.water_field_coarse.parameters()) + list(self.water_field_fine.parameters())
        if self.temporal_distortion is not None:
            param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
        return param_groups

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)
        coarse_pred, coarse_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
        )
        fine_pred, fine_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )

        rgb_loss_coarse = self.rgb_loss(coarse_image, coarse_pred)
        rgb_loss_fine = self.rgb_loss(fine_image, fine_pred)

        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_metrics_dict(self, outputs, batch):
        return {}

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        image = batch["image"].to(outputs["rgb_coarse"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        assert self.config.collider_params is not None
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image_t = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse_t = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine_t = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image_t, rgb_coarse_t)
        fine_psnr = self.psnr(image_t, rgb_fine_t)
        fine_ssim = self.ssim(image_t, rgb_fine_t)
        fine_lpips = self.lpips(image_t, rgb_fine_t)
        assert isinstance(fine_ssim, torch.Tensor)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr),
            "fine_psnr": float(fine_psnr),
            "fine_ssim": float(fine_ssim),
            "fine_lpips": float(fine_lpips),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
