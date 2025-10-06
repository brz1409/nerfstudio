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

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class TwoMediaVanillaModelConfig(ModelConfig):
    """Two-media Vanilla NeRF with stratified sampling (no occupancy grid)."""

    _target: Type = field(default_factory=lambda: TwoMediaNeRFModel)

    # Interface parameters
    water_surface_height_world: float = 0.0
    """Height of water surface in world coordinates (before dataparser transform)."""
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


class TwoMediaNeRFModel(Model):
    """Two-medium NeRF using stratified sampling for robust memory-safe training."""

    config: TwoMediaVanillaModelConfig

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
        self.renderer_depth = DepthRenderer(method="expected")

        # Loss
        self.rgb_loss = MSELoss()

        # Metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from torchmetrics.functional import structural_similarity_index_measure

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # Setup water interface geometry
        self._setup_water_interface()

        CONSOLE.log(
            f"TwoMediaNeRF initialized: 4 separate MLPs (air/water × coarse/fine), "
            f"{self.config.num_coarse_samples} coarse + {self.config.num_importance_samples} fine samples per segment"
        )

    def _setup_water_interface(self) -> None:
        """Compute water surface plane in model coordinates."""
        # Get dataparser transform
        transform = self.kwargs.get("dataparser_transform")
        scale = float(self.kwargs.get("dataparser_scale", 1.0))

        if transform is None:
            # Identity transform
            self.register_buffer("water_plane_normal", torch.tensor([[0.0, 0.0, 1.0]]), persistent=False)
            water_z = -self.config.water_surface_height_world
            self.register_buffer("water_plane_d", torch.tensor([water_z]), persistent=False)
            CONSOLE.log(f"Water surface at z={water_z:.4f} (no transform)")
            return

        # Transform normal and point
        # World: plane is z = h → normal [0,0,1], point [0,0,h]
        # Plane equation: n·(x - p) = 0 → n·x - n·p = 0 → n·x + d = 0 where d = -n·p

        normal_world = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        point_world = torch.tensor([0.0, 0.0, self.config.water_surface_height_world], dtype=torch.float32)

        # Apply rotation to normal (rotation only, no translation)
        R = transform[:3, :3].float()
        t = transform[:3, 3].float()

        normal_model = R @ normal_world
        normal_model = F.normalize(normal_model, dim=0)

        # Transform point: p_model = scale * (R @ p_world + t)
        point_model = scale * (R @ point_world + t)

        # Plane equation: n·x + d = 0, where d = -n·p
        d = -torch.dot(normal_model, point_model)

        self.register_buffer("water_plane_normal", normal_model.unsqueeze(0), persistent=False)
        self.register_buffer("water_plane_d", torch.tensor([d]), persistent=False)

        # Log resulting position
        if abs(normal_model[2]) > 1e-6:
            # Solve for z when x=0, y=0: n_z * z + d = 0 → z = -d/n_z
            water_z_model = -d / normal_model[2]
            CONSOLE.log(f"Water surface at z={water_z_model.item():.4f} in model space")
        else:
            CONSOLE.log("[yellow]Water surface not aligned with z-axis after transform[/yellow]")

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

    def _create_ray_samples(
        self,
        ray_bundle: RayBundle,
        t_starts: Tensor,
        t_ends: Tensor,
    ) -> RaySamples:
        """Create RaySamples from t values along rays."""
        origins = ray_bundle.origins  # (num_rays, 3)
        directions = ray_bundle.directions  # (num_rays, 3)

        # Create frustums
        from nerfstudio.cameras.rays import Frustums
        frustums = Frustums(
            origins=origins.unsqueeze(-2).expand(*t_starts.shape, 3),
            directions=directions.unsqueeze(-2).expand(*t_starts.shape, 3),
            starts=t_starts.unsqueeze(-1),
            ends=t_ends.unsqueeze(-1),
            pixel_area=ray_bundle.pixel_area.unsqueeze(-2).expand(*t_starts.shape, 1) if ray_bundle.pixel_area is not None else None,
        )

        return RaySamples(
            frustums=frustums,
            camera_indices=ray_bundle.camera_indices.unsqueeze(-2).expand(*t_starts.shape, 1) if ray_bundle.camera_indices is not None else None,
        )

    def _render_segment(
        self,
        field: NeRFField,
        ray_bundle: RayBundle,
        t_starts: Tensor,
        t_ends: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Render a ray segment through a field.

        Returns:
            weights: (num_rays, num_samples) rendering weights
            rgb: (num_rays, num_samples, 3) RGB colors
            densities: (num_rays, num_samples) density values
        """
        ray_samples = self._create_ray_samples(ray_bundle, t_starts, t_ends)
        field_outputs = field(ray_samples)

        densities = field_outputs[FieldHeadNames.DENSITY].squeeze(-1)  # (num_rays, num_samples)
        rgb = field_outputs[FieldHeadNames.RGB]  # (num_rays, num_samples, 3)

        # Volume rendering weights
        deltas = t_ends - t_starts
        alphas = 1.0 - torch.exp(-densities * deltas)

        num_rays = t_starts.shape[0]
        transmittance = torch.cumprod(
            torch.cat([torch.ones((num_rays, 1), device=t_starts.device), 1.0 - alphas + 1e-10], dim=-1),
            dim=-1,
        )[:, :-1]
        weights = alphas * transmittance

        return weights, rgb, densities

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Tensor]:
        """Forward pass with hierarchical sampling through two media."""

        # Flatten ray bundle
        ray_bundle = ray_bundle.flatten()
        num_rays = len(ray_bundle)
        device = ray_bundle.origins.device

        origins = ray_bundle.origins
        directions = F.normalize(ray_bundle.directions, dim=-1)

        # Near/far planes
        near = ray_bundle.nears[..., 0] if ray_bundle.nears is not None else torch.zeros(num_rays, device=device)
        far = ray_bundle.fars[..., 0] if ray_bundle.fars is not None else torch.full((num_rays,), 10.0, device=device)

        # Find water surface intersection
        eps = self.config.interface_epsilon
        signed_dist = self._signed_distance_to_water(origins)
        denom = (directions @ self.water_plane_normal.T).squeeze(-1)

        t_water = torch.where(
            torch.abs(denom) > 1e-6,
            -signed_dist / denom,
            far + 1.0  # No intersection
        )

        starts_above = signed_dist > eps
        hits_water = (t_water > near) & (t_water < far) & starts_above

        # ===== COARSE PASS =====
        # Air segment
        t_air_far = torch.where(hits_water, t_water - eps, far)
        t_air_far = torch.clamp(t_air_far, min=near + eps)

        # Uniform sampling within the air segment bounds.
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
        t_starts_air = ray_samples_air_coarse.frustums.starts[..., 0]
        t_ends_air = ray_samples_air_coarse.frustums.ends[..., 0]

        weights_air_coarse, rgb_air_coarse, _ = self._render_segment(
            self.air_field_coarse, ray_bundle, t_starts_air, t_ends_air
        )

        # Water segment (only for rays that hit water)
        if hits_water.any():
            # Create refracted bundle
            entry_points = origins + directions * t_water.unsqueeze(-1)
            refracted_dirs = self._compute_refraction(directions)
            water_origins = entry_points + refracted_dirs * eps
            remaining_dist = torch.clamp(far - t_water, min=0.0)

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
            t_starts_water = ray_samples_water_coarse.frustums.starts[..., 0]
            t_ends_water = ray_samples_water_coarse.frustums.ends[..., 0]
            weights_water_coarse, rgb_water_coarse, _ = self._render_segment(
                self.water_field_coarse, water_bundle, t_starts_water, t_ends_water
            )

            # Account for air transmittance
            air_transmittance = 1.0 - weights_air_coarse.sum(dim=-1, keepdim=True)
            weights_water_coarse = weights_water_coarse * air_transmittance
        else:
            weights_water_coarse = torch.zeros((num_rays, 0), device=device)
            rgb_water_coarse = torch.zeros((num_rays, 0, 3), device=device)
            t_starts_water = torch.zeros((num_rays, 0), device=device)
            t_ends_water = torch.zeros((num_rays, 0), device=device)

        # ===== FINE PASS =====
        # Importance sampling in air
        ray_samples_pdf_air = self.sampler_pdf(ray_bundle, self._create_ray_samples(ray_bundle, t_starts_air, t_ends_air), weights_air_coarse.detach())
        t_starts_air_fine = ray_samples_pdf_air.frustums.starts[..., 0]
        t_ends_air_fine = ray_samples_pdf_air.frustums.ends[..., 0]

        # Combine coarse + fine for air
        t_starts_air_all = torch.cat([t_starts_air, t_starts_air_fine], dim=-1)
        t_ends_air_all = torch.cat([t_ends_air, t_ends_air_fine], dim=-1)
        # Sort by start time
        t_mids_air = (t_starts_air_all + t_ends_air_all) / 2.0
        sorted_indices_air = torch.argsort(t_mids_air, dim=-1)
        t_starts_air_all = torch.gather(t_starts_air_all, -1, sorted_indices_air)
        t_ends_air_all = torch.gather(t_ends_air_all, -1, sorted_indices_air)

        weights_air_fine, rgb_air_fine, _ = self._render_segment(
            self.air_field_fine, ray_bundle, t_starts_air_all, t_ends_air_all
        )

        # Importance sampling in water
        if hits_water.any():
            ray_samples_pdf_water = self.sampler_pdf(water_bundle, self._create_ray_samples(water_bundle, t_starts_water, t_ends_water), weights_water_coarse.detach())
            t_starts_water_fine = ray_samples_pdf_water.frustums.starts[..., 0]
            t_ends_water_fine = ray_samples_pdf_water.frustums.ends[..., 0]

            t_starts_water_all = torch.cat([t_starts_water, t_starts_water_fine], dim=-1)
            t_ends_water_all = torch.cat([t_ends_water, t_ends_water_fine], dim=-1)
            t_mids_water = (t_starts_water_all + t_ends_water_all) / 2.0
            sorted_indices_water = torch.argsort(t_mids_water, dim=-1)
            t_starts_water_all = torch.gather(t_starts_water_all, -1, sorted_indices_water)
            t_ends_water_all = torch.gather(t_ends_water_all, -1, sorted_indices_water)

            weights_water_fine, rgb_water_fine, _ = self._render_segment(
                self.water_field_fine, water_bundle, t_starts_water_all, t_ends_water_all
            )

            # Account for air transmittance
            air_transmittance_fine = 1.0 - weights_air_fine.sum(dim=-1, keepdim=True)
            weights_water_fine = weights_water_fine * air_transmittance_fine
        else:
            weights_water_fine = torch.zeros((num_rays, 0), device=device)
            rgb_water_fine = torch.zeros((num_rays, 0, 3), device=device)
            t_starts_water_all = torch.zeros((num_rays, 0), device=device)
            t_ends_water_all = torch.zeros((num_rays, 0), device=device)

        # ===== FINAL RENDERING =====
        weights = torch.cat([weights_air_fine, weights_water_fine], dim=-1)
        rgbs = torch.cat([rgb_air_fine, rgb_water_fine], dim=-2)

        rgb = (weights.unsqueeze(-1) * rgbs).sum(dim=-2)
        accumulation = weights.sum(dim=-1, keepdim=True)

        # Depth
        t_mids_all = torch.cat([
            (t_starts_air_all + t_ends_air_all) / 2.0,
            (t_starts_water_all + t_ends_water_all) / 2.0 + t_water.unsqueeze(-1) if hits_water.any() else torch.zeros((num_rays, 0), device=device),
        ], dim=-1)
        depth = (weights * t_mids_all).sum(dim=-1, keepdim=True) / (accumulation + 1e-10)

        # Background
        if self.config.background_color == "white":
            rgb = rgb + (1.0 - accumulation)

        return {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.air_field_coarse is None or self.air_field_fine is None or self.water_field_coarse is None or self.water_field_fine is None:
            raise ValueError("populate_modules() must be called before get_param_groups")
        # Keep separate optimizers for air and water branches to match TrainerConfig
        param_groups["air_field"] = list(self.air_field_coarse.parameters()) + list(self.air_field_fine.parameters())
        param_groups["water_field"] = list(self.water_field_coarse.parameters()) + list(self.water_field_fine.parameters())
        return param_groups

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb_loss = self.rgb_loss(image, outputs["rgb"])
        return {"rgb_loss": rgb_loss}

    def get_metrics_dict(self, outputs, batch):
        return {}

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Metrics
        image_t = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_t = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image_t, rgb_t)
        ssim = self.ssim(image_t, rgb_t)
        lpips = self.lpips(image_t, rgb_t)

        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}
        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth}

        return metrics_dict, images_dict
