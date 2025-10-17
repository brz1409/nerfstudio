# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
Implementation of vanilla nerf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import nerfacc
import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import writer as global_writer
from nerfstudio.utils.writer import EventName as WriterEventName
# metrics
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


@dataclass
class VanillaModelConfig(ModelConfig):
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: NeRFModel)
    # The original Vanilla NeRF sampler settings are unused when nerfacc is enabled.
    # Kept for backward compatibility with configs but not used.
    num_coarse_samples: int = 64
    """(Unused with nerfacc) Number of samples in coarse field evaluation."""
    num_importance_samples: int = 128
    """(Unused with nerfacc) Number of samples in fine field evaluation."""

    enable_temporal_distortion: bool = False
    """Specifies whether or not to include ray warping based on time."""
    temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"kind": TemporalDistortionKind.DNERF})
    """Parameters to instantiate temporal distortion with"""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Whether to randomize the background color."""

    # nerfacc / instant-ngp style sampling parameters
    grid_resolution: Union[int, List[int]] = 128
    """Resolution of the occupancy grid used for sampling."""
    grid_levels: int = 4
    """Levels of the occupancy grid used for sampling."""
    alpha_thre: float = 0.01
    """Opacity threshold for skipping samples during ray marching."""
    cone_angle: float = 0.0
    """Cone angle for ray marching (0.0 for uniform; ~1/256 for real scenes)."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering; auto-computed if None."""


class NeRFModel(Model):
    """Vanilla NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: VanillaModelConfig

    def __init__(
        self,
        config: VanillaModelConfig,
        **kwargs,
    ) -> None:
        self.field = None
        self.temporal_distortion = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        # nerfacc occupancy grid + volumetric sampler (instant-ngp style)
        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)
        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 1000

        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # Use the field to provide densities for occlusion skipping during training
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )
        render_step = self.config.render_step_size
        render_step_str = f"{render_step:.6f}" if render_step is not None else "auto"
        CONSOLE.log(
            "VanillaNeRF sampling uses nerfacc occupancy grid "
            f"(render_step_size={render_step_str}, levels={self.config.grid_levels}, "
            f"resolution={self.config.grid_resolution})."
        )
        self._nerfacc_sampling_logged = False

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        # Expected depth supports packed ray samples from nerfacc.
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()



        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        if getattr(self.config, "enable_temporal_distortion", False):
            params = self.config.temporal_distortion_params
            kind = params.pop("kind")
            self.temporal_distortion = kind.to_temporal_distortion(params)

    def get_training_callbacks(self, training_callback_attributes):
        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field.density_fn(x)
                * (self.config.render_step_size if self.config.render_step_size is not None else 1.0),
            )

        def log_nerfacc_metrics(step: int):
            # Log explicit nerfacc metrics to dashboards (WandB/TensorBoard/Local).
            try:
                if hasattr(self, "_nerfacc_last_samples_sum"):
                    global_writer.put_scalar(name=WriterEventName.NERFACC_ACTIVE, scalar=1.0, step=step)
                    global_writer.put_scalar(
                        name=WriterEventName.NERFACC_SAMPLES_PER_BATCH,
                        scalar=float(self._nerfacc_last_samples_sum),
                        step=step,
                    )
                # Also expose render step size as a constant for convenience
                if step == 0:
                    rstep = self.config.render_step_size if self.config.render_step_size is not None else -1.0
                    global_writer.put_scalar(name=WriterEventName.NERFACC_RENDER_STEP_SIZE, scalar=float(rstep), step=0)
            except Exception:
                # Best-effort logging: never break training due to logging
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
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        if self.temporal_distortion is not None:
            param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # Volumetric sampling with nerfacc occupancy grid (packed representation)
        if not self._nerfacc_sampling_logged:
            CONSOLE.log("VanillaNeRF forward pass confirmed nerfacc-based sampling path is active.")
            self._nerfacc_sampling_logged = True

        num_rays = len(ray_bundle)
        near_plane = 0.0
        far_plane = 1e10
        if self.config.enable_collider and self.config.collider_params is not None:
            near_plane = float(self.config.collider_params["near_plane"])
            far_plane = float(self.config.collider_params["far_plane"])

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=self.config.render_step_size if self.config.render_step_size is not None else 1e-3,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        # temporal distortion if enabled
        if self.temporal_distortion is not None and ray_samples.times is not None:
            offsets = self.temporal_distortion(ray_samples.frustums.get_positions(), ray_samples.times)
            ray_samples.frustums.set_offsets(offsets)

        # Evaluate field on samples
        field_outputs = self.field.forward(ray_samples)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        # Compute weights from densities using nerfacc for packed rays
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0][..., None]
        # Store latest nerfacc sample count for logging callbacks
        try:
            self._nerfacc_last_samples_sum = float(packed_info[:, 1].sum())
        except Exception:
            self._nerfacc_last_samples_sum = None

        # Render RGB/Depth/Accumulation (packed)
        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
        }
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict: Dict[str, torch.Tensor] = {}
        # Always expose a positive flag + number of samples to confirm nerfacc sampling.
        if "num_samples_per_ray" in outputs:
            metrics_dict["nerfacc_active"] = torch.tensor(1.0, device=self.device)
            metrics_dict["nerfacc_num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
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

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
