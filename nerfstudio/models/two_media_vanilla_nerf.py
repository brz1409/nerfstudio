# Copyright 2024 the Regents of the University of California,
# Nerfstudio Team and contributors. All rights reserved.
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

"""Two-medium Vanilla NeRF (air/water) with refraction at a planar interface.

This model mirrors nerfstudio.models.vanilla_nerf.NeRFModel almost 1:1:
- Same encodings, fields (here duplicated for air/water and coarse/fine),
  samplers (Uniform + PDF), renderers (RGB/Accum/Depth) and losses/metrics.
- Only differences:
  * We split each camera ray into up to two segments separated by a water plane.
  * We refract the ray direction at the interface using Snell's law.
  * We *combine* segment contributions by multiplying the second segment's
    weights / RGB / accumulation by the remaining transmittance of the first
    segment, so the overall behavior matches vanilla when no interface is hit.
  * Depth is computed as the *expected depth along the original camera ray*
    by projecting each segment's sample midpoints back to the camera ray.

The water plane parameters are expected in dataparser metadata:
    metadata["water_surface"]["plane_model"] = {"normal": [nx,ny,nz], "d": d}
Optionally:
    metadata["water_surface"]["recommendations"]["far_plane"] -> float

If metadata is missing, the model reduces to vanilla NeRF with the *air* fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch import nn
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
from nerfstudio.utils.math import intersect_aabb, safe_normalize


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

@dataclass
class TwoMediaVanillaModelConfig(ModelConfig):
    """Two-media Vanilla NeRF with stratified sampling (no occupancy grid)."""

    _target: Type = field(default_factory=lambda: TwoMediaVanillaModel)

    # Refractive indices
    air_refractive_index: float = 1.0
    water_refractive_index: float = 1.333
    interface_epsilon: float = 1e-4  # offset around interface to avoid self-intersection

    # Sampling (kept identical to vanilla defaults)
    num_coarse_samples: int = 64
    num_importance_samples: int = 128

    # Optional temporal distortion
    enable_temporal_distortion: bool = False
    temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"kind": TemporalDistortionKind.DNERF})

    # Training tricks
    use_gradient_scaling: bool = False

    # Background
    background_color: str = "white"  # Literal["random", "last_sample", "black", "white"]

    # Collider (same as base)
    # collider_params inherited; we optionally override far_plane from metadata in populate_modules()


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

class TwoMediaVanillaModel(Model):
    """Vanilla NeRF split into two media with a planar refractive interface."""

    config: TwoMediaVanillaModelConfig

    def __init__(self, config: TwoMediaVanillaModelConfig, **kwargs) -> None:
        # fields initialized in populate_modules
        self.air_field_coarse: Optional[NeRFField] = None
        self.air_field_fine: Optional[NeRFField] = None
        self.water_field_coarse: Optional[NeRFField] = None
        self.water_field_fine: Optional[NeRFField] = None
        self.temporal_distortion = None

        super().__init__(config=config, **kwargs)
        self._tir_warned: bool = False  # single warning for total internal reflection

    # ----------------------------- helpers: water plane -----------------------------

    def _load_water_plane_from_metadata(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (normal[1,3], d[1]) from dataparser metadata if available, else None."""
        md = self.kwargs.get("metadata") if hasattr(self, "kwargs") else None
        if not isinstance(md, dict):
            return None
        ws = md.get("water_surface")
        if not isinstance(ws, dict):
            return None
        plane = ws.get("plane_model")
        if not isinstance(plane, dict):
            return None
        n = torch.tensor(plane["normal"], dtype=torch.float32)
        d = torch.tensor([float(plane["d"])], dtype=torch.float32)
        n = n / (n.norm() + 1e-10)
        # enforce upward z if desired (dataparser already does, but keep consistent)
        if n[2] < 0:
            n = -n
            d = -d
        return n[None, :], d  # [1,3], [1]

    # ----------------------------- populate modules ---------------------------------

    def populate_modules(self) -> None:
        """Initialize fields/samplers/renderers and water plane buffers."""
        # Try to get water plane *before* collider creation so we can tune far_plane
        plane = self._load_water_plane_from_metadata()
        recommendations_far: Optional[float] = None
        md = self.kwargs.get("metadata") if hasattr(self, "kwargs") else None
        if isinstance(md, dict):
            rec = md.get("water_surface", {}).get("recommendations", {})
            if isinstance(rec, dict) and "far_plane" in rec:
                try:
                    recommendations_far = float(rec["far_plane"])
                except Exception:
                    recommendations_far = None

        # If we have a recommended far plane, patch collider params *before* super call
        if recommendations_far is not None and self.config.collider_params is not None:
            # keep near plane, set far plane to recommendation
            cp = dict(self.config.collider_params)
            cp["far_plane"] = float(recommendations_far)
            self.config.collider_params = to_immutable_dict(cp)

        # create collider etc.
        super().populate_modules()

        # Encodings
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        # Two media × coarse/fine
        self.air_field_coarse = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.air_field_fine = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.water_field_coarse = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.water_field_fine = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)

        # Samplers (parity with vanilla: set counts in Ctor, call without num_samples)
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # Renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        # We'll still build a DepthRenderer for convenience, but final depth comes from projection
        self.renderer_depth = DepthRenderer(method="expected")

        # Losses & metrics
        self.rgb_loss = MSELoss()
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # Temporal distortion (optional)
        if getattr(self.config, "enable_temporal_distortion", False):
            params = dict(self.config.temporal_distortion_params)
            kind = params.pop("kind")
            self.temporal_distortion = kind.to_temporal_distortion(params)

        # Water plane buffers (persisted)
        if plane is not None:
            n, d = plane
            self.register_buffer("water_plane_normal", n, persistent=True)  # [1,3]
            self.register_buffer("water_plane_d", d, persistent=True)       # [1]
        else:
            # Fallback: no water plane → behave like vanilla (always "air")
            self.register_buffer("water_plane_normal", torch.tensor([[0.0, 0.0, 1.0]]), persistent=True)
            self.register_buffer("water_plane_d", torch.tensor([1e6]), persistent=True)  # push plane far away

        # Cached aabb for segment-2 clipping
        # SceneBox has aabb tensor [2,3]. Pack to [6] for intersect_aabb util.
        aabb = torch.cat([self.scene_box.aabb[0], self.scene_box.aabb[1]], dim=0).to(torch.float32)
        self.register_buffer("scene_aabb_flat", aabb, persistent=True)

    # ----------------------------- optimizer groups ---------------------------------

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        if any(x is None for x in [self.air_field_coarse, self.air_field_fine, self.water_field_coarse, self.water_field_fine]):
            raise ValueError("populate_modules() must be called before get_param_groups")
        groups = {}
        groups["fields"] = (
            list(self.air_field_coarse.parameters())
            + list(self.air_field_fine.parameters())
            + list(self.water_field_coarse.parameters())
            + list(self.water_field_fine.parameters())
        )
        if self.temporal_distortion is not None:
            groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
        return groups

    # ----------------------------- math: plane & refraction -------------------------

    @torch.no_grad()
    def _signed_distance_to_water(self, x: torch.Tensor) -> torch.Tensor:
        """Compute signed distance n·x + d; positive = 'above' plane (air side)."""
        return (x @ self.water_plane_normal[0]) + self.water_plane_d[0]

    @staticmethod
    def _snell_refract(v: torch.Tensor, n: torch.Tensor, n1: float, n2: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refract direction v across interface with normal n (both normalized).
        Returns (refracted_dir, tir_mask) where tir_mask=True indicates total internal reflection.
        The normal should point from medium-2 into medium-1 for the dot products to follow
        the standard convention; we handle orientation internally.
        """
        v = safe_normalize(v)
        n = safe_normalize(n)
        cosi = torch.sum(v * n, dim=-1, keepdim=True).clamp(-1.0, 1.0)  # cos(theta_i) with sign
        # We want normal pointing *against* incoming ray for the refraction formula.
        n_face = torch.where(cosi > 0, -n, n)  # flip if needed
        cosi = torch.sum(v * n_face, dim=-1, keepdim=True)
        eta = n1 / n2
        k = 1.0 - eta * eta * (1.0 - cosi * cosi)
        tir = k < 0.0
        # Only compute where not tir
        t = eta * v - (eta * cosi + torch.sqrt(torch.clamp(k, min=0.0))) * n_face
        t = torch.where(tir, v, t)  # dummy output for tir
        t = safe_normalize(t)
        return t, tir.squeeze(-1)

    # ----------------------------- raybundle utilities ------------------------------

    def _make_raybundle(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        nears: torch.Tensor,
        fars: torch.Tensor,
        base: RayBundle,
        pixel_area: Optional[torch.Tensor] = None,
    ) -> RayBundle:
        """Clone a RayBundle with new geometric params."""
        return RayBundle(
            origins=origins,
            directions=directions,
            nears=nears[..., None],
            fars=fars[..., None],
            pixel_area=pixel_area if pixel_area is not None else base.pixel_area,
            camera_indices=base.camera_indices,
            times=base.times,
            metadata=base.metadata,
        )

    # --------------------------------------------------------------------------------
    # Core rendering (identical structure to vanilla, but split into segments)
    # --------------------------------------------------------------------------------

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        if any(x is None for x in [self.air_field_coarse, self.air_field_fine, self.water_field_coarse, self.water_field_fine]):
            raise ValueError("populate_modules() must be called before get_outputs")

        device = ray_bundle.origins.device
        dtype = ray_bundle.origins.dtype

        origins = ray_bundle.origins.view(-1, 3)
        directions = safe_normalize(ray_bundle.directions.view(-1, 3))
        num_rays = origins.shape[0]

        # pull near/far: fallbacks if collider absent
        near_default = torch.zeros(num_rays, device=device, dtype=dtype)
        far_default = torch.full((num_rays,), 1e4, device=device, dtype=dtype)
        near = (ray_bundle.nears[..., 0] if ray_bundle.nears is not None else near_default).view(-1)
        far = (ray_bundle.fars[..., 0] if ray_bundle.fars is not None else far_default).view(-1)

        # Signed distance at origin -> start medium
        signed = self._signed_distance_to_water(origins)  # >0: air, <0: water
        starts_above = signed > 0
        starts_below = signed < 0

        # Ray-plane intersection parameter along *original* ray
        denom = torch.sum(directions * self.water_plane_normal, dim=-1)  # [N]
        # Use large value for parallel rays (no intersection)
        t_int = torch.where(denom.abs() > 1e-8, -signed / denom, torch.full_like(far, 1e9))
        # Intersection is valid if within (near, far)
        hits = (t_int > (near + 1e-6)) & (t_int < (far - 1e-6))

        eps = self.config.interface_epsilon
        # Prepare per-ray segment near/far for the *first* segment along original ray
        seg1_near = near.clone()
        seg1_far = torch.where(hits, t_int - eps, far)  # clip to interface when hit

        # Segment-2 setup: build origins & directions by refraction at interface
        p_int = origins + t_int.unsqueeze(-1) * directions
        # Decide which way we're going through the interface
        n = self.water_plane_normal.expand_as(origins)
        going_air_to_water = starts_above & hits
        going_water_to_air = starts_below & hits

        # Compute refracted directions for both cases
        dir_refr_a2w, tir_a2w = self._snell_refract(directions, n, self.config.air_refractive_index, self.config.water_refractive_index)
        dir_refr_w2a, tir_w2a = self._snell_refract(directions, n, self.config.water_refractive_index, self.config.air_refractive_index)

        # Choose refracted dir per ray (dummy when not used)
        dir_refr = torch.where(going_air_to_water.unsqueeze(-1), dir_refr_a2w, directions)
        dir_refr = torch.where(going_water_to_air.unsqueeze(-1), dir_refr_w2a, dir_refr)

        tir_mask = torch.zeros_like(hits)
        tir_mask = torch.where(going_air_to_water, tir_a2w, tir_mask)
        tir_mask = torch.where(going_water_to_air, tir_w2a, tir_mask)

        # If TIR occurs, we disable segment-2 for that ray
        seg2_enabled = hits & (~tir_mask)

        if torch.any(tir_mask) and not self._tir_warned:
            # Single gentle warning; no spam
            print("[TwoMediaVanillaModel] Total internal reflection encountered for some rays; disabling segment-2 on those rays.")
            self._tir_warned = True

        # Segment-2 near/far in its *own* parameterization: near=0; far via AABB intersection
        # origins for seg2: intersection point + eps along refracted dir
        seg2_origins = p_int + eps * dir_refr
        seg2_dirs = dir_refr

        # Compute [tmin, tmax] to scene aabb along seg2 ray. Returns distances >= 0.
        tmin2, tmax2 = intersect_aabb(seg2_origins, seg2_dirs, self.scene_aabb_flat)
        # We want a usable finite far; if invalid_value returned, disable seg2
        valid_aabb = tmax2.isfinite() & (tmax2 < 1e9)
        seg2_enabled = seg2_enabled & valid_aabb
        seg2_near = torch.zeros_like(near)
        seg2_far = torch.where(seg2_enabled, tmax2, torch.zeros_like(tmax2))

        # Build raybundles per segment / medium
        # Segment-1 uses original origins/directions; Segment-2 uses refracted
        # Note: we do *not* attempt to adjust pixel_area for segment-2. (Set to None to avoid misuse.)
        rb_seg1 = self._make_raybundle(origins, directions, seg1_near, seg1_far, base=ray_bundle)
        rb_seg2 = self._make_raybundle(seg2_origins, seg2_dirs, seg2_near, seg2_far, base=ray_bundle, pixel_area=None)

        # Masks for media selection
        seg1_is_air = starts_above
        seg1_is_water = starts_below
        seg2_is_air = going_water_to_air & seg2_enabled
        seg2_is_water = going_air_to_water & seg2_enabled

        # ---------------------------------- Coarse pass ----------------------------------
        # Segment-1 coarse
        rs1 = self.sampler_uniform(rb_seg1)  # [N, S1]
        if self.temporal_distortion is not None and rs1.times is not None:
            offsets = self.temporal_distortion(rs1.frustums.get_positions(), rs1.times)
            rs1.frustums.set_offsets(offsets)
        # Evaluate both media fields but mask via densities
        f1_air = self.air_field_coarse.forward(rs1)
        f1_water = self.water_field_coarse.forward(rs1)
        # Merge densities/colors based on seg1 medium
        sigma1 = torch.where(seg1_is_air.unsqueeze(-1), f1_air[FieldHeadNames.DENSITY], f1_water[FieldHeadNames.DENSITY])
        rgb1 = torch.where(seg1_is_air.unsqueeze(-1).unsqueeze(-1), f1_air[FieldHeadNames.RGB], f1_water[FieldHeadNames.RGB])
        w1 = rs1.get_weights(sigma1)
        rgb_coarse_1 = self.renderer_rgb(rgb=rgb1, weights=w1)
        acc_coarse_1 = self.renderer_accumulation(w1)

        # Segment-2 coarse (only for enabled rays); sample anyway and zero later
        rs2 = self.sampler_uniform(rb_seg2)
        if self.temporal_distortion is not None and rs2.times is not None:
            offsets = self.temporal_distortion(rs2.frustums.get_positions(), rs2.times)
            rs2.frustums.set_offsets(offsets)
        f2_air = self.air_field_coarse.forward(rs2)
        f2_water = self.water_field_coarse.forward(rs2)
        sigma2 = torch.where(seg2_is_air.unsqueeze(-1), f2_air[FieldHeadNames.DENSITY], f2_water[FieldHeadNames.DENSITY])
        rgb2 = torch.where(seg2_is_air.unsqueeze(-1).unsqueeze(-1), f2_air[FieldHeadNames.RGB], f2_water[FieldHeadNames.RGB])
        w2 = rs2.get_weights(sigma2)
        # Scale second segment by remaining transmittance of segment-1
        T1 = (1.0 - acc_coarse_1).clamp(min=0.0, max=1.0)[..., None]
        rgb_coarse_2 = self.renderer_rgb(rgb=rgb2, weights=w2) * T1
        acc_coarse_2 = self.renderer_accumulation(w2) * T1[..., 0]

        rgb_coarse = rgb_coarse_1 + rgb_coarse_2 * seg2_enabled.unsqueeze(-1)
        accumulation_coarse = (acc_coarse_1 + acc_coarse_2 * seg2_enabled.float()).clamp(max=1.0)

        # Depth coarse via projection onto original camera ray
        depth_coarse = self._expected_depth_projected(w1, rs1, w2 * T1, rs2, origins, directions, seg2_enabled)

        # ---------------------------------- Fine pass ------------------------------------
        rs1_fine = self.sampler_pdf(ray_bundle=rb_seg1, ray_samples_uniform=rs1, weights=w1)
        if self.temporal_distortion is not None and rs1_fine.times is not None:
            offsets = self.temporal_distortion(rs1_fine.frustums.get_positions(), rs1_fine.times)
            rs1_fine.frustums.set_offsets(offsets)
        f1_air_f = self.air_field_fine.forward(rs1_fine)
        f1_water_f = self.water_field_fine.forward(rs1_fine)
        sigma1_f = torch.where(seg1_is_air.unsqueeze(-1), f1_air_f[FieldHeadNames.DENSITY], f1_water_f[FieldHeadNames.DENSITY])
        rgb1_f = torch.where(seg1_is_air.unsqueeze(-1).unsqueeze(-1), f1_air_f[FieldHeadNames.RGB], f1_water_f[FieldHeadNames.RGB])
        w1_f = rs1_fine.get_weights(sigma1_f)
        rgb_fine_1 = self.renderer_rgb(rgb=rgb1_f, weights=w1_f)
        acc_fine_1 = self.renderer_accumulation(w1_f)

        rs2_fine = self.sampler_pdf(ray_bundle=rb_seg2, ray_samples_uniform=rs2, weights=w2)
        if self.temporal_distortion is not None and rs2_fine.times is not None:
            offsets = self.temporal_distortion(rs2_fine.frustums.get_positions(), rs2_fine.times)
            rs2_fine.frustums.set_offsets(offsets)
        f2_air_f = self.air_field_fine.forward(rs2_fine)
        f2_water_f = self.water_field_fine.forward(rs2_fine)
        sigma2_f = torch.where(seg2_is_air.unsqueeze(-1), f2_air_f[FieldHeadNames.DENSITY], f2_water_f[FieldHeadNames.DENSITY])
        rgb2_f = torch.where(seg2_is_air.unsqueeze(-1).unsqueeze(-1), f2_air_f[FieldHeadNames.RGB], f2_water_f[FieldHeadNames.RGB])
        w2_f = rs2_fine.get_weights(sigma2_f)
        T1_f = (1.0 - acc_fine_1).clamp(min=0.0, max=1.0)[..., None]
        rgb_fine_2 = self.renderer_rgb(rgb=rgb2_f, weights=w2_f) * T1_f
        acc_fine_2 = self.renderer_accumulation(w2_f) * T1_f[..., 0]

        rgb_fine = rgb_fine_1 + rgb_fine_2 * seg2_enabled.unsqueeze(-1)
        accumulation_fine = (acc_fine_1 + acc_fine_2 * seg2_enabled.float()).clamp(max=1.0)

        depth_fine = self._expected_depth_projected(w1_f, rs1_fine, w2_f * T1_f, rs2_fine, origins, directions, seg2_enabled)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs

    # ----------------------------- depth (projected) --------------------------------

    @staticmethod
    def _expected_midpoint_positions(ray_samples: RaySamples) -> torch.Tensor:
        """Return 3D midpoints of each sample: origin + dir * 0.5*(start+end)."""
        mids = 0.5 * (ray_samples.frustums.starts + ray_samples.frustums.ends)  # [N,S,1]
        pos = ray_samples.frustums.origins + ray_samples.frustums.directions * mids  # [N,S,3]
        return pos

    def _expected_depth_projected(
        self,
        weights1: torch.Tensor,
        rs1: RaySamples,
        weights2_scaled: torch.Tensor,
        rs2: RaySamples,
        cam_origins: torch.Tensor,
        cam_dirs_unit: torch.Tensor,
        seg2_enabled: torch.Tensor,
    ) -> torch.Tensor:
        """Compute expected depth along the *original* camera ray by projecting
        the 3D midpoints from seg-1 and seg-2 onto that ray.
        """
        # [N,S1,3] and [N,S2,3]
        pos1 = self._expected_midpoint_positions(rs1)
        pos2 = self._expected_midpoint_positions(rs2)

        # projection scalar t = dot(p - o, d_unit)
        t1 = torch.sum((pos1 - cam_origins[:, None, :]) * cam_dirs_unit[:, None, :], dim=-1, keepdim=True)  # [N,S1,1]
        t2 = torch.sum((pos2 - cam_origins[:, None, :]) * cam_dirs_unit[:, None, :], dim=-1, keepdim=True)  # [N,S2,1]

        # combine weights (note: weights2 already multiplied by remaining transmittance)
        # pad disabled seg2 with zeros
        weights2 = weights2_scaled * seg2_enabled[:, None, None].float()

        # expected depth = sum(w * t) / sum(w)
        w_sum = (weights1.sum(dim=-2, keepdim=True) + weights2.sum(dim=-2, keepdim=True)).clamp(min=1e-10)
        depth = (weights1 * t1).sum(dim=-2, keepdim=True) + (weights2 * t2).sum(dim=-2, keepdim=True)
        depth = depth / w_sum  # [N,1,1]
        return depth.squeeze(-1)  # [N,1]

    # ----------------------------- loss & metrics -----------------------------------

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
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

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
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

        # metrics
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
