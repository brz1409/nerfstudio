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

This model trains two independent neural fields for air and water while
sharing a single nerfacc occupancy grid and sampler. Rays entering the
water medium are refracted according to Snell's law at a planar interface,
which is specified in world coordinates and transformed into the model's
coordinate frame before training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import nerfacc
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from nerfstudio.utils import colormaps, misc
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import writer as global_writer
from nerfstudio.utils.writer import EventName as WriterEventName


@dataclass
class TwoMediaVanillaModelConfig(VanillaModelConfig):
    """Vanilla NeRF configuration extended with a two-media interface."""

    _target: Type = field(default_factory=lambda: TwoMediaNeRFModel)
    water_surface_height_world: float = 0.0
    """Height of the water surface in the original world coordinate system."""
    air_refractive_index: float = 1.0
    """Refractive index of the air medium."""
    water_refractive_index: float = 1.333
    """Refractive index of the water medium."""
    interface_epsilon: float = 1e-4
    """Small offset applied around the interface to avoid numerical overlap."""


class TwoMediaNeRFModel(Model):
    """Two-medium Vanilla NeRF model that refracts rays at a planar interface."""

    config: TwoMediaVanillaModelConfig

    def __init__(
        self,
        config: TwoMediaVanillaModelConfig,
        **kwargs: Any,
    ) -> None:
        self.air_field: Optional[NeRFField] = None
        self.water_field: Optional[NeRFField] = None
        self.temporal_distortion = None

        self._dataparser_transform: Optional[Tensor] = kwargs.get("dataparser_transform")
        self._dataparser_scale: float = float(kwargs.get("dataparser_scale", 1.0))

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self) -> None:
        """Initialise fields, samplers, losses and renderers."""
        super().populate_modules()

        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.air_field = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )
        self.water_field = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)
        if self.config.render_step_size is None:
            # Auto step size: ~1000 samples in the base level grid.
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 1000

        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self._evaluate_density,
        )

        render_step = self.config.render_step_size
        render_step_str = f"{render_step:.6f}" if render_step is not None else "auto"
        CONSOLE.log(
            "TwoMediaNeRF sampling uses a shared nerfacc occupancy grid "
            f"(render_step_size={render_step_str}, levels={self.config.grid_levels}, "
            f"resolution={self.config.grid_resolution})."
        )
        self._nerfacc_sampling_logged = False

        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        self.rgb_loss = MSELoss()

        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        if getattr(self.config, "enable_temporal_distortion", False):
            params = self.config.temporal_distortion_params
            kind = params.pop("kind")
            self.temporal_distortion = kind.to_temporal_distortion(params)

        self._air_ior = float(self.config.air_refractive_index)
        self._water_ior = float(self.config.water_refractive_index)
        self._setup_interface_geometry()

    def _setup_interface_geometry(self) -> None:
        """Compute the interface plane in model coordinates and store as buffers."""

        if self._dataparser_transform is None:
            transform = torch.eye(3, 4, dtype=torch.float32)
        else:
            transform = self._dataparser_transform.to(dtype=torch.float32)

        normal_world = torch.tensor([0.0, 0.0, 1.0], dtype=transform.dtype)
        rotation = transform[:, :3]
        translation = transform[:, 3]

        # Plane: n^T x + d = 0; world plane is z = h -> d_world = -h.
        d_world = -float(self.config.water_surface_height_world)

        normal = rotation @ normal_world
        d_model = self._dataparser_scale * (d_world - torch.dot(normal, translation))

        normal_norm = torch.norm(normal)
        if normal_norm > 0:
            normal = normal / normal_norm
            d_model = d_model / normal_norm
        else:
            normal = normal_world

        self.register_buffer("water_plane_normal", normal.reshape(1, 3), persistent=False)
        self.register_buffer("water_plane_offset", torch.tensor([d_model], dtype=normal.dtype), persistent=False)

        if torch.abs(normal[2]) > 1e-6:
            height_model = (-d_model / normal[2]).item()
            CONSOLE.log(
                f"TwoMediaNeRF water surface located at z={height_model:.4f} in model coordinates.")
        else:
            CONSOLE.log(
                "TwoMediaNeRF water surface normal not aligned with model z-axis after transform.", style="yellow"
            )

    def _signed_distance_to_interface(self, positions: Tensor) -> Tensor:
        normal = self.water_plane_normal.to(device=positions.device, dtype=positions.dtype)
        offset = self.water_plane_offset.to(device=positions.device, dtype=positions.dtype)
        return torch.matmul(positions, normal.transpose(0, 1)).squeeze(-1) + offset

    def _evaluate_density(self, positions: Tensor, times: Optional[Tensor] = None) -> Tensor:
        if self.air_field is None or self.water_field is None:
            raise ValueError("Fields must be initialised before density evaluation.")

        distances = self._signed_distance_to_interface(positions)
        air_mask = distances >= 0.0

        density = torch.zeros_like(distances)[..., None]
        if air_mask.any():
            density[air_mask] = self.air_field.density_fn(positions[air_mask], times=times[air_mask] if times is not None else None)
        water_mask = ~air_mask
        if water_mask.any():
            density[water_mask] = self.water_field.density_fn(
                positions[water_mask], times=times[water_mask] if times is not None else None
            )
        return density

    def get_training_callbacks(self, training_callback_attributes):
        def update_occupancy_grid(step: int):
            assert self.config.render_step_size is not None
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self._evaluate_density(x)
                * (self.config.render_step_size if self.config.render_step_size is not None else 1.0),
            )

        def log_nerfacc_metrics(step: int):
            try:
                if hasattr(self, "_nerfacc_last_samples_sum"):
                    global_writer.put_scalar(name=WriterEventName.NERFACC_ACTIVE, scalar=1.0, step=step)
                    global_writer.put_scalar(
                        name=WriterEventName.NERFACC_SAMPLES_PER_BATCH,
                        scalar=float(self._nerfacc_last_samples_sum),
                        step=step,
                    )
                if step == 0:
                    rstep = self.config.render_step_size if self.config.render_step_size is not None else -1.0
                    global_writer.put_scalar(name=WriterEventName.NERFACC_RENDER_STEP_SIZE, scalar=float(rstep), step=0)
            except Exception:  # pragma: no cover - logging should never break training
                pass

        from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=log_nerfacc_metrics,
            ),
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        if self.air_field is None or self.water_field is None:
            raise ValueError("populate_modules() must be called before get_param_groups")
        param_groups: Dict[str, List[Parameter]] = {
            "air_field": list(self.air_field.parameters()),
            "water_field": list(self.water_field.parameters()),
        }
        if self.temporal_distortion is not None:
            param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
        return param_groups

    def _compute_refraction(self, incident_dirs: Tensor) -> Tensor:
        normal = self.water_plane_normal.to(device=incident_dirs.device, dtype=incident_dirs.dtype)
        normal = normal.expand_as(incident_dirs)
        incident = F.normalize(incident_dirs, dim=-1)

        cos_incident = torch.sum(incident * normal, dim=-1, keepdim=True)
        normal = torch.where(cos_incident > 0.0, -normal, normal)
        cos_incident = torch.clamp(torch.sum(incident * normal, dim=-1, keepdim=True), max=0.0)

        eta = self._air_ior / self._water_ior
        eta_tensor = torch.full_like(cos_incident, eta)
        sin_t_sq = eta_tensor**2 * (1.0 - cos_incident**2)
        total_internal = sin_t_sq > 1.0
        cos_t = torch.sqrt(torch.clamp(1.0 - sin_t_sq, min=0.0))
        refracted = eta_tensor * incident + (eta_tensor * (-cos_incident) - cos_t) * normal
        refracted = F.normalize(refracted, dim=-1)

        if total_internal.any():  # fallback to reflection for completeness
            reflected = incident - 2.0 * cos_incident * normal
            refracted = torch.where(total_internal.expand_as(refracted), reflected, refracted)

        return refracted

    def _accumulate_depth(
        self,
        weights: Tensor,
        depths: Tensor,
        ray_indices: Tensor,
        num_rays: int,
    ) -> Tensor:
        eps = 1e-8
        depth_values = depths[..., None]
        depth_numerator = nerfacc.accumulate_along_rays(
            weights[..., 0], values=depth_values, ray_indices=ray_indices, n_rays=num_rays
        )
        acc = nerfacc.accumulate_along_rays(weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays)
        depth = depth_numerator / (acc.unsqueeze(-1) + eps)
        depth = torch.nan_to_num(depth)
        return depth

    def get_outputs(self, ray_bundle: RayBundle):
        if self.air_field is None or self.water_field is None:
            raise ValueError("populate_modules() must be called before get_outputs")

        if not self._nerfacc_sampling_logged:
            CONSOLE.log("TwoMediaNeRF forward pass confirmed nerfacc-based sampling path is active.")
            self._nerfacc_sampling_logged = True

        flat_bundle = ray_bundle.flatten()
        num_rays = len(flat_bundle)

        device = flat_bundle.origins.device
        dtype = flat_bundle.origins.dtype

        near_plane = 0.0
        far_plane = 1e10
        if self.config.enable_collider and self.config.collider_params is not None:
            near_plane = float(self.config.collider_params["near_plane"])
            far_plane = float(self.config.collider_params["far_plane"])

        base_near = (
            flat_bundle.nears[..., 0] if flat_bundle.nears is not None else torch.full((num_rays,), near_plane, device=device, dtype=dtype)
        )
        base_far = (
            flat_bundle.fars[..., 0] if flat_bundle.fars is not None else torch.full((num_rays,), far_plane, device=device, dtype=dtype)
        )

        origins = flat_bundle.origins
        directions = F.normalize(flat_bundle.directions, dim=-1)

        surface_eps = float(self.config.interface_epsilon)

        normal = self.water_plane_normal.to(device=device, dtype=dtype)
        signed_distance = self._signed_distance_to_interface(origins)
        denom = torch.matmul(directions, normal.transpose(0, 1)).squeeze(-1)

        with torch.no_grad():
            safe_denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
            t_intersection = -signed_distance / safe_denom

            starts_underwater = signed_distance < -surface_eps
            valid_hit = (
                (signed_distance >= -surface_eps)
                & (denom < -surface_eps)
                & (t_intersection > base_near + surface_eps)
                & (t_intersection < base_far - surface_eps)
            )

        air_near = base_near
        air_far = torch.where(valid_hit, torch.minimum(t_intersection - surface_eps, base_far), base_far)
        air_far = torch.where(air_far <= air_near + surface_eps, air_near, air_far)
        air_far = torch.where(starts_underwater, air_near, air_far)

        air_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=flat_bundle.pixel_area,
            camera_indices=flat_bundle.camera_indices,
            nears=air_near.unsqueeze(-1),
            fars=air_far.unsqueeze(-1),
            metadata=flat_bundle.metadata,
            times=flat_bundle.times,
        )

        render_step = self.config.render_step_size if self.config.render_step_size is not None else 1e-3
        with torch.no_grad():
            ray_samples_air, ray_indices_air = self.sampler(
                ray_bundle=air_bundle,
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        if self.temporal_distortion is not None and ray_samples_air.times is not None:
            offsets_air = self.temporal_distortion(ray_samples_air.frustums.get_positions(), ray_samples_air.times)
            ray_samples_air.frustums.set_offsets(offsets_air)

        field_air = self.air_field.forward(ray_samples_air)
        if self.config.use_gradient_scaling:
            field_air = scale_gradients_by_distance_squared(field_air, ray_samples_air)

        packed_info_air = nerfacc.pack_info(ray_indices_air, num_rays)
        weights_air = nerfacc.render_weight_from_density(
            t_starts=ray_samples_air.frustums.starts[..., 0],
            t_ends=ray_samples_air.frustums.ends[..., 0],
            sigmas=field_air[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info_air,
        )[0][..., None]

        hit_indices = torch.nonzero(valid_hit, as_tuple=False).squeeze(-1)
        has_water = hit_indices.numel() > 0

        weights_water = torch.zeros((0, 1), device=device, dtype=dtype)
        ray_indices_water = torch.zeros((0,), dtype=torch.long, device=device)
        rgb_water = torch.zeros((0, 3), device=device, dtype=dtype)
        depth_mid_water = torch.zeros((0,), device=device, dtype=dtype)

        if has_water:
            entry_depths = t_intersection[hit_indices]
            remaining_far = torch.clamp(base_far[hit_indices] - entry_depths, min=0.0)
            valid_remaining = remaining_far > surface_eps

            if valid_remaining.any():
                keep_indices = hit_indices[valid_remaining]
                entry_depths = entry_depths[valid_remaining]
                remaining_far = remaining_far[valid_remaining]

                entry_points = origins[keep_indices] + directions[keep_indices] * entry_depths.unsqueeze(-1)
                refracted_dirs = self._compute_refraction(directions[keep_indices])
                water_origins = entry_points + refracted_dirs * surface_eps

                metadata_subset = {k: v[keep_indices] for k, v in flat_bundle.metadata.items()}
                water_bundle = RayBundle(
                    origins=water_origins,
                    directions=refracted_dirs,
                    pixel_area=flat_bundle.pixel_area[keep_indices],
                    camera_indices=None if flat_bundle.camera_indices is None else flat_bundle.camera_indices[keep_indices],
                    nears=torch.zeros_like(entry_depths).unsqueeze(-1),
                    fars=torch.clamp(remaining_far - surface_eps, min=0.0).unsqueeze(-1),
                    metadata=metadata_subset,
                    times=None if flat_bundle.times is None else flat_bundle.times[keep_indices],
                )

                with torch.no_grad():
                    ray_samples_water, ray_indices_subset = self.sampler(
                        ray_bundle=water_bundle,
                        near_plane=0.0,
                        far_plane=far_plane,
                        render_step_size=render_step,
                        alpha_thre=self.config.alpha_thre,
                        cone_angle=self.config.cone_angle,
                    )

                if self.temporal_distortion is not None and ray_samples_water.times is not None:
                    offsets_water = self.temporal_distortion(
                        ray_samples_water.frustums.get_positions(), ray_samples_water.times
                    )
                    ray_samples_water.frustums.set_offsets(offsets_water)

                field_water = self.water_field.forward(ray_samples_water)
                if self.config.use_gradient_scaling:
                    field_water = scale_gradients_by_distance_squared(field_water, ray_samples_water)

                global_ray_indices = keep_indices[ray_indices_subset]
                packed_info_water = nerfacc.pack_info(global_ray_indices, num_rays)

                weights_water_raw = nerfacc.render_weight_from_density(
                    t_starts=ray_samples_water.frustums.starts[..., 0],
                    t_ends=ray_samples_water.frustums.ends[..., 0],
                    sigmas=field_water[FieldHeadNames.DENSITY][..., 0],
                    packed_info=packed_info_water,
                )[0][..., None]

                transmittance_air = torch.ones((num_rays,), device=device, dtype=dtype)
                if weights_air.numel() > 0:
                    accum_air = nerfacc.accumulate_along_rays(
                        weights_air[..., 0], values=None, ray_indices=ray_indices_air, n_rays=num_rays
                    )
                    transmittance_air = torch.clamp(1.0 - accum_air, min=0.0, max=1.0)

                weights_water = weights_water_raw * transmittance_air[global_ray_indices][:, None]
                ray_indices_water = global_ray_indices
                rgb_water = field_water[FieldHeadNames.RGB]

                depth_mid_water = (
                    ((ray_samples_water.frustums.starts + ray_samples_water.frustums.ends) / 2.0)[..., 0]
                    + entry_depths[ray_indices_subset]
                )

        rgb_air = field_air[FieldHeadNames.RGB]
        depth_mid_air = ((ray_samples_air.frustums.starts + ray_samples_air.frustums.ends) / 2.0)[..., 0]

        weights_all = weights_air
        rgb_all = rgb_air
        ray_indices_all = ray_indices_air
        depth_all = depth_mid_air

        if weights_water.numel() > 0:
            weights_all = torch.cat([weights_all, weights_water], dim=0)
            rgb_all = torch.cat([rgb_all, rgb_water], dim=0)
            ray_indices_all = torch.cat([ray_indices_all, ray_indices_water], dim=0)
            depth_all = torch.cat([depth_all, depth_mid_water], dim=0)

        if weights_all.numel() > 0 and ray_indices_all.numel() > 0:
            sort_perm = torch.argsort(ray_indices_all)
            weights_all = weights_all[sort_perm]
            rgb_all = rgb_all[sort_perm]
            ray_indices_all = ray_indices_all[sort_perm]
            depth_all = depth_all[sort_perm]

        packed_info_all = nerfacc.pack_info(ray_indices_all, num_rays)
        try:
            self._nerfacc_last_samples_sum = float(packed_info_all[:, 1].sum())
        except Exception:
            self._nerfacc_last_samples_sum = None

        rgb = self.renderer_rgb(
            rgb=rgb_all,
            weights=weights_all,
            ray_indices=ray_indices_all,
            num_rays=num_rays,
        )

        accumulation = self.renderer_accumulation(weights=weights_all, ray_indices=ray_indices_all, num_rays=num_rays)

        depth = self._accumulate_depth(weights_all, depth_all, ray_indices_all, num_rays)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info_all[:, 1],
        }
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict: Dict[str, torch.Tensor] = {}
        if "num_samples_per_ray" in outputs:
            metrics_dict["nerfacc_active"] = torch.tensor(1.0, device=self.device)
            metrics_dict["nerfacc_num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        device = outputs["rgb"].device
        image = batch["image"].to(device)
        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"], pred_accumulation=outputs["accumulation"], gt_image=image
        )
        rgb_loss = self.rgb_loss(image, pred_rgb)
        return {"rgb_loss": rgb_loss}

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
