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
from collections.abc import Mapping
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


LOG_PREFIX = "[TwoMediaNeRF]"


@dataclass
class TwoMediaVanillaModelConfig(ModelConfig):
    """Two-media Vanilla NeRF with stratified sampling (no occupancy grid)."""

    _target: Type = field(default_factory=lambda: TwoMediaNeRFModel)

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

        self._log_model_configuration()

    def _log_model_configuration(self) -> None:
        """Emit a concise summary of the most relevant model parameters."""
        cfg = self.config
        config_parts = [
            f"air_ior={cfg.air_refractive_index:.3f}",
            f"water_ior={cfg.water_refractive_index:.3f}",
            f"interface_eps={cfg.interface_epsilon:.1e}",
            f"coarse_samples={cfg.num_coarse_samples}",
            f"fine_samples={cfg.num_importance_samples}",
            f"eval_chunk={cfg.eval_num_rays_per_chunk}",
            f"background={cfg.background_color}",
            f"grad_scaling={'on' if cfg.use_gradient_scaling else 'off'}",
            f"temporal_distortion={'on' if self.temporal_distortion is not None else 'off'}",
        ]
        CONSOLE.log(f"{LOG_PREFIX} Config | " + ", ".join(config_parts))

        scene_box = getattr(self, "scene_box", None)
        if scene_box is not None:
            CONSOLE.log(f"{LOG_PREFIX} Scene box | {scene_box.aabb.flatten().tolist()}")

    def _setup_water_interface(self) -> None:
        """Compute water surface plane in model coordinates."""

        metadata = self.kwargs.get("metadata")
        if isinstance(metadata, Mapping):
            water_meta = metadata.get("water_surface")
            if isinstance(water_meta, Mapping) and self._setup_water_from_metadata(water_meta):
                return

        raise ValueError(
            "TwoMediaNeRFModel requires water surface metadata from the dataparser. "
            "Ensure the dataparser provides `metadata['water_surface']['plane_model']`."
        )

    def _setup_water_from_metadata(self, water_meta: Mapping[str, Any]) -> bool:
        """Use precomputed plane parameters from dataparser metadata if available."""
        plane_model = water_meta.get("plane_model")
        if plane_model is None:
            return False

        normal = plane_model.get("normal")
        d_value = plane_model.get("d")
        if normal is None or d_value is None:
            return False

        normal_tensor = torch.tensor(normal, dtype=torch.float32)
        norm = torch.norm(normal_tensor)
        if norm == 0:
            CONSOLE.log(f"{LOG_PREFIX} warning: water surface metadata normal has zero length; ignoring entry.")
            return False

        normal_tensor = normal_tensor / norm
        d = float(d_value) / float(norm)

        d_tensor = torch.tensor([d], dtype=torch.float32)
        self.register_buffer("water_plane_normal", normal_tensor.unsqueeze(0), persistent=False)
        self.register_buffer("water_plane_d", d_tensor, persistent=False)
        self.register_buffer("water_plane_offset", d_tensor.clone(), persistent=False)

        source = water_meta.get("source", "Metadata")
        self._log_water_surface_result(
            f"{source} (Model)",
            None,
            self.water_plane_normal.squeeze(0),
            d,
            scale=1.0,
        )
        return True

    def _log_water_surface_result(
        self,
        coord_system: str,
        input_height: Optional[float],
        normal_model: torch.Tensor,
        d: float,
        scale: float = 1.0,
        show_header: bool = True
    ) -> None:
        """Log the resulting water surface position and check for issues.

        Args:
            coord_system: Name of input coordinate system
            input_height: Input height (if applicable)
            normal_model: Computed normal in model space
            d: Plane equation constant
            scale: Reference scale factor (if available)
        """
        if show_header:
            header_parts = [f"source={coord_system}", f"scale={scale:.4f}"]
            if input_height is not None:
                header_parts.append(f"input_height={input_height:.4f}")
            CONSOLE.log(f"{LOG_PREFIX} Water surface | " + ", ".join(header_parts))

        normal_fmt = "[" + ", ".join(f"{float(v):.4f}" for v in normal_model) + "]"

        if abs(normal_model[2]) > 1e-6:
            water_z_model = -d / float(normal_model[2])
            cosine = float(abs(normal_model[2]).clamp(-1.0, 1.0))
            angle_deg = math.degrees(math.acos(cosine))

            summary = [
                f"model_z={water_z_model:.4f}",
                f"tilt_deg={angle_deg:.2f}",
                f"normal={normal_fmt}",
                f"plane_const={d:.4f}",
            ]
            CONSOLE.log(f"{LOG_PREFIX} Water surface | " + ", ".join(summary))

            if angle_deg > 5.0:
                CONSOLE.log(
                    f"{LOG_PREFIX} warning: water surface tilt {angle_deg:.1f}° exceeds 5°; verify markers or transforms."
                )
        else:
            CONSOLE.log(
                f"{LOG_PREFIX} error: water surface normal is near-horizontal → plane vertical, normal={normal_fmt}, d={d:.4f}."
            )

    def _signed_distance_to_water(self, positions: Tensor) -> Tensor:
        """Compute signed distance to water surface. Positive = above water (air)."""
        # Plane equation: n·x + d = 0
        # Signed distance: (n·x + d) / |n| (normal is already normalized)
        return (positions @ self.water_plane_normal.T).squeeze(-1) + self.water_plane_d

    def _compute_refraction(self, incident_dirs: Tensor, n1: float, n2: float) -> Tensor:
        """Compute refracted ray directions at interface for a given IOR pair.

        Args:
            incident_dirs: [N,3] incident directions (normalized)
            n1: refractive index of incident medium
            n2: refractive index of transmitted medium

        Returns:
            Refracted unit directions. In TIR cases clamps to critical angle (warns) and returns best-effort.
        """
        normal = F.normalize(self.water_plane_normal, dim=-1)
        incident = F.normalize(incident_dirs, dim=-1)

        # Orient normal opposite to incident to ensure cos_i >= 0
        cos_i_raw = (incident * normal).sum(dim=-1, keepdim=True)
        normal = torch.where(cos_i_raw > 0, -normal, normal)
        cos_i = -(incident * normal).sum(dim=-1, keepdim=True).clamp(min=0.0)

        # Snell's law
        eta = float(n1) / float(n2)
        sin2_t = eta**2 * (1.0 - cos_i**2)

        # Total internal reflection check
        tir_mask = sin2_t > 1.0
        if tir_mask.any():
            num_tir = int(tir_mask.sum().item())
            CONSOLE.log(
                f"{LOG_PREFIX} warning: total internal reflection detected on {num_tir} rays for n1={n1:.3f}→n2={n2:.3f}."
            )
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

    def _expected_depth_multi(
        self,
        weights: Tensor,
        accumulation: Tensor,
        parts: List[Tuple[Optional[RaySamples], Optional[Tensor]]],
    ) -> Tensor:
        """Expected depth from multiple segments.

        Args:
            weights: Combined global weights across all segments in the same order as parts
            accumulation: Combined accumulation corresponding to weights
            parts: List of (ray_samples, offset) pairs.
                   For each part, t_mids = (starts+ends)/2 + offset (offset can be None → 0). If ray_samples is None
                   or has zero samples, contributes empty mids.
        Returns:
            Depth tensor [N,1]
        """
        device = weights.device
        dtype = weights.dtype
        num_rays = weights.shape[0]
        t_mids_all: List[Tensor] = []
        for rs, offset in parts:
            if rs is None or rs.frustums.starts.shape[-2] == 0:
                t_mids_all.append(torch.zeros((num_rays, 0, 1), device=device, dtype=dtype))
                continue
            mids = (rs.frustums.starts + rs.frustums.ends) / 2.0
            if offset is not None:
                mids = mids + offset.view(-1, 1, 1)
            t_mids_all.append(mids)
        t_mids_cat = torch.cat(t_mids_all, dim=-2)
        depth = (weights * t_mids_cat).sum(dim=-2) / (accumulation + 1e-10)
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

        # Interface param along original ray
        t_int = torch.where(
            torch.abs(denom) > 1e-6,
            -signed_dist / denom,
            far + 1.0,
        )

        starts_above = signed_dist > eps
        starts_below = signed_dist < -eps
        hits_from_above = (t_int > near) & (t_int < far) & starts_above
        hits_from_below = (t_int > near) & (t_int < far) & starts_below

        # Coarse sampling: First segment (air if starts above, otherwise water)
        # AIR (first) for rays starting above surface; disable for others
        t_air1_far_unclamped = torch.where(hits_from_above, t_int - eps, far)
        t_air1_far = torch.where(
            starts_above, torch.clamp(t_air1_far_unclamped, min=near + eps), torch.zeros_like(far)
        )
        air1_fars = t_air1_far

        air1_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=ray_bundle.pixel_area,
            camera_indices=ray_bundle.camera_indices,
            nears=torch.where(starts_above, near, torch.zeros_like(near)).unsqueeze(-1),
            fars=air1_fars.unsqueeze(-1),
            metadata=ray_bundle.metadata,
            times=ray_bundle.times,
        )
        ray_samples_air1_coarse = self.sampler_uniform(
            air1_bundle, num_samples=self.config.num_coarse_samples
        )
        self._apply_temporal_distortion(ray_samples_air1_coarse)
        weights_air1_coarse, rgb_air1_coarse = self._render_segment(
            self.air_field_coarse, ray_samples_air1_coarse
        )

        # WATER (second) for rays starting above and hitting interface
        water2_bundle: Optional[RayBundle] = None
        ray_samples_water2_coarse: Optional[RaySamples] = None
        weights_water2_coarse = torch.zeros((num_rays, 0, 1), device=device)
        weights_water2_coarse_pdf = weights_water2_coarse
        rgb_water2_coarse = torch.zeros((num_rays, 0, 3), device=device)

        if hits_from_above.any():
            entry_points = origins + directions * t_int.unsqueeze(-1)
            refracted_dirs = self._compute_refraction(directions, self.config.air_refractive_index, self.config.water_refractive_index)
            water_origins = entry_points + refracted_dirs * eps

            t_near_w2, t_far_w2 = self._intersect_scene_box(water_origins, refracted_dirs)
            # Mask per ray: if no hit, far=0
            t_far_w2 = torch.where(hits_from_above, t_far_w2, torch.zeros_like(t_far_w2))

            water2_bundle = RayBundle(
                origins=water_origins,
                directions=refracted_dirs,
                pixel_area=ray_bundle.pixel_area,
                camera_indices=ray_bundle.camera_indices,
                nears=torch.where(hits_from_above, t_near_w2, torch.zeros_like(t_near_w2)).unsqueeze(-1),
                fars=t_far_w2.unsqueeze(-1),
                metadata=ray_bundle.metadata,
                times=ray_bundle.times,
            )

            ray_samples_water2_coarse = self.sampler_uniform(
                water2_bundle, num_samples=self.config.num_coarse_samples
            )
            self._apply_temporal_distortion(ray_samples_water2_coarse)
            weights_water2_coarse, rgb_water2_coarse = self._render_segment(
                self.water_field_coarse, ray_samples_water2_coarse
            )

            weights_water2_coarse_pdf = weights_water2_coarse

            air1_transmittance = torch.clamp(
                1.0 - weights_air1_coarse.sum(dim=-2, keepdim=True), min=0.0, max=1.0
            )

            if self.training and not hasattr(self, "_logged_water_sampling"):
                valid_mask = (t_far_w2 > 0)
                rays_hit = int(valid_mask.sum().item())
                stats_parts = [f"rays_hit={rays_hit}/{num_rays}"]
                if rays_hit > 0:
                    remaining_vals = t_far_w2[valid_mask]
                    stats_parts.append(f"mean_far={float(remaining_vals.mean().item()):.3f}")
                    stats_parts.append(f"max_far={float(remaining_vals.max().item()):.3f}")
                    old_method = torch.clamp(far - t_int, min=0.0)
                    old_vals = old_method[valid_mask]
                    stats_parts.append(f"old_far_mean={float(old_vals.mean().item()):.3f}")
                air_trans_values = air1_transmittance[hits_from_above].reshape(-1)
                if air_trans_values.numel() > 0:
                    trans_mean = float(air_trans_values.mean().item())
                    trans_min = float(air_trans_values.min().item())
                    stats_parts.append(f"air_trans_mean={trans_mean:.3f}")
                    stats_parts.append(f"air_trans_min={trans_min:.3e}")
                    if trans_min < 1e-3:
                        CONSOLE.log(
                            f"{LOG_PREFIX} notice: minimum air transmittance {trans_min:.3e}; water branch may receive few samples."
                        )
                scene_box = getattr(self, "scene_box", None)
                if scene_box is not None:
                    stats_parts.append(f"scene_box={scene_box.aabb.flatten().tolist()}")
                CONSOLE.log(f"{LOG_PREFIX} Water sampling | " + ", ".join(stats_parts))
                self._logged_water_sampling = True

            # Globalize with air transmittance and mask by hits
            weights_water2_coarse = weights_water2_coarse_pdf * air1_transmittance
            weights_water2_coarse = weights_water2_coarse * hits_from_above.view(-1, 1, 1)

        # WATER (first) for rays starting below surface; disable for others
        t_water1_far_unclamped = torch.where(hits_from_below, t_int - eps, far)
        t_water1_far = torch.where(
            starts_below, torch.clamp(t_water1_far_unclamped, min=near + eps), torch.zeros_like(far)
        )
        water1_fars = t_water1_far

        water1_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=ray_bundle.pixel_area,
            camera_indices=ray_bundle.camera_indices,
            nears=torch.where(starts_below, near, torch.zeros_like(near)).unsqueeze(-1),
            fars=water1_fars.unsqueeze(-1),
            metadata=ray_bundle.metadata,
            times=ray_bundle.times,
        )
        ray_samples_water1_coarse = self.sampler_uniform(
            water1_bundle, num_samples=self.config.num_coarse_samples
        )
        self._apply_temporal_distortion(ray_samples_water1_coarse)
        weights_water1_coarse, rgb_water1_coarse = self._render_segment(
            self.water_field_coarse, ray_samples_water1_coarse
        )

        # AIR (second) for rays starting below and hitting interface
        air2_bundle: Optional[RayBundle] = None
        ray_samples_air2_coarse: Optional[RaySamples] = None
        weights_air2_coarse = torch.zeros((num_rays, 0, 1), device=device)
        weights_air2_coarse_pdf = weights_air2_coarse
        rgb_air2_coarse = torch.zeros((num_rays, 0, 3), device=device)

        if hits_from_below.any():
            entry_points_b = origins + directions * t_int.unsqueeze(-1)
            refracted_dirs_b = self._compute_refraction(directions, self.config.water_refractive_index, self.config.air_refractive_index)
            air_origins2 = entry_points_b + refracted_dirs_b * eps

            t_near_a2, t_far_a2 = self._intersect_scene_box(air_origins2, refracted_dirs_b)
            t_far_a2 = torch.where(hits_from_below, t_far_a2, torch.zeros_like(t_far_a2))

            air2_bundle = RayBundle(
                origins=air_origins2,
                directions=refracted_dirs_b,
                pixel_area=ray_bundle.pixel_area,
                camera_indices=ray_bundle.camera_indices,
                nears=torch.where(hits_from_below, t_near_a2, torch.zeros_like(t_near_a2)).unsqueeze(-1),
                fars=t_far_a2.unsqueeze(-1),
                metadata=ray_bundle.metadata,
                times=ray_bundle.times,
            )

            ray_samples_air2_coarse = self.sampler_uniform(
                air2_bundle, num_samples=self.config.num_coarse_samples
            )
            self._apply_temporal_distortion(ray_samples_air2_coarse)
            weights_air2_coarse, rgb_air2_coarse = self._render_segment(
                self.air_field_coarse, ray_samples_air2_coarse
            )

            weights_air2_coarse_pdf = weights_air2_coarse

            water1_transmittance = torch.clamp(
                1.0 - weights_water1_coarse.sum(dim=-2, keepdim=True), min=0.0, max=1.0
            )
            weights_air2_coarse = weights_air2_coarse_pdf * water1_transmittance
            weights_air2_coarse = weights_air2_coarse * hits_from_below.view(-1, 1, 1)

        # Fine sampling (air)
        ray_samples_air1_fine = self.sampler_pdf(
            air1_bundle, ray_samples_air1_coarse, weights_air1_coarse.detach()
        )
        self._apply_temporal_distortion(ray_samples_air1_fine)
        weights_air1_fine, rgb_air1_fine = self._render_segment(
            self.air_field_fine, ray_samples_air1_fine
        )

        # Fine sampling (water)
        # Fine sampling (water second)
        weights_water2_fine = torch.zeros((num_rays, 0, 1), device=device)
        rgb_water2_fine = torch.zeros((num_rays, 0, 3), device=device)
        ray_samples_water2_fine: Optional[RaySamples] = None

        if hits_from_above.any() and water2_bundle is not None and ray_samples_water2_coarse is not None:
            ray_samples_water2_fine = self.sampler_pdf(
                water2_bundle, ray_samples_water2_coarse, weights_water2_coarse_pdf.detach()
            )
            self._apply_temporal_distortion(ray_samples_water2_fine)
            weights_water2_fine, rgb_water2_fine = self._render_segment(
                self.water_field_fine, ray_samples_water2_fine
            )

            air1_transmittance_fine = torch.clamp(
                1.0 - weights_air1_fine.sum(dim=-2, keepdim=True), min=0.0, max=1.0
            )

            if self.training and not hasattr(self, "_logged_water_fine_sampling"):
                fine_trans_values = air1_transmittance_fine[hits_from_above].reshape(-1)
                if fine_trans_values.numel() > 0:
                    trans_mean = float(fine_trans_values.mean().item())
                    trans_min = float(fine_trans_values.min().item())
                    stats = [
                        f"mean_trans={trans_mean:.3f}",
                        f"min_trans={trans_min:.3e}",
                        f"samples={fine_trans_values.numel()}",
                    ]
                    CONSOLE.log(f"{LOG_PREFIX} Water fine gating | " + ", ".join(stats))
                    if trans_min < 1e-3:
                        CONSOLE.log(
                            f"{LOG_PREFIX} notice: fine-stage air transmittance floor {trans_min:.3e}; confirm water branch supervision."
                        )
                self._logged_water_fine_sampling = True

            weights_water2_fine = weights_water2_fine * air1_transmittance_fine
            weights_water2_fine = weights_water2_fine * hits_from_above.view(-1, 1, 1)

        # Fine sampling (water first)
        ray_samples_water1_fine = self.sampler_pdf(
            water1_bundle, ray_samples_water1_coarse, weights_water1_coarse.detach()
        )
        self._apply_temporal_distortion(ray_samples_water1_fine)
        weights_water1_fine, rgb_water1_fine = self._render_segment(
            self.water_field_fine, ray_samples_water1_fine
        )

        # Fine sampling (air second)
        weights_air2_fine = torch.zeros((num_rays, 0, 1), device=device)
        rgb_air2_fine = torch.zeros((num_rays, 0, 3), device=device)
        ray_samples_air2_fine: Optional[RaySamples] = None

        if hits_from_below.any() and air2_bundle is not None and ray_samples_air2_coarse is not None:
            ray_samples_air2_fine = self.sampler_pdf(
                air2_bundle, ray_samples_air2_coarse, weights_air2_coarse_pdf.detach()
            )
            self._apply_temporal_distortion(ray_samples_air2_fine)
            weights_air2_fine, rgb_air2_fine = self._render_segment(
                self.air_field_fine, ray_samples_air2_fine
            )

            water1_transmittance_fine = torch.clamp(
                1.0 - weights_water1_fine.sum(dim=-2, keepdim=True), min=0.0, max=1.0
            )
            weights_air2_fine = weights_air2_fine * water1_transmittance_fine
            weights_air2_fine = weights_air2_fine * hits_from_below.view(-1, 1, 1)

        # Aggregate coarse outputs
        weights_coarse_all = torch.cat(
            [
                weights_air1_coarse,
                weights_water2_coarse,
                weights_water1_coarse,
                weights_air2_coarse,
            ],
            dim=-2,
        )
        rgbs_coarse = torch.cat(
            [
                rgb_air1_coarse,
                rgb_water2_coarse,
                rgb_water1_coarse,
                rgb_air2_coarse,
            ],
            dim=-2,
        )
        rgb_coarse = self.renderer_rgb(rgb=rgbs_coarse, weights=weights_coarse_all)
        accumulation_coarse = self.renderer_accumulation(weights_coarse_all)
        depth_coarse = self._expected_depth_multi(
            weights_coarse_all,
            accumulation_coarse,
            [
                (ray_samples_air1_coarse, None),
                (ray_samples_water2_coarse, t_int),
                (ray_samples_water1_coarse, None),
                (ray_samples_air2_coarse, t_int),
            ],
        )

        # Aggregate fine outputs
        weights_fine_all = torch.cat(
            [
                weights_air1_fine,
                weights_water2_fine,
                weights_water1_fine,
                weights_air2_fine,
            ],
            dim=-2,
        )
        rgbs_fine = torch.cat(
            [
                rgb_air1_fine,
                rgb_water2_fine,
                rgb_water1_fine,
                rgb_air2_fine,
            ],
            dim=-2,
        )
        rgb_fine = self.renderer_rgb(rgb=rgbs_fine, weights=weights_fine_all)
        accumulation_fine = self.renderer_accumulation(weights_fine_all)
        depth_fine = self._expected_depth_multi(
            weights_fine_all,
            accumulation_fine,
            [
                (ray_samples_air1_fine, None),
                (ray_samples_water2_fine, t_int),
                (ray_samples_water1_fine, None),
                (ray_samples_air2_fine, t_int),
            ],
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
