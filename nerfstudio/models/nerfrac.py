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
NeRFrac: Neural Radiance Fields through Refractive Surface

Based on "NeRFrac: Neural Radiance Fields through Refractive Surface" (ICCV 2023)
by Yifan Zhan, Shohei Nobuhara, Ko Nishino, Yinqiang Zheng

Learns a refractive field to estimate the geometry of the refractive surface (e.g., water),
then applies Snell's law to refract rays before sampling the radiance field.
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
from nerfstudio.fields.refractive_field import RefractiveField
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class NeRFracModelConfig(ModelConfig):
    """NeRFrac model configuration."""

    _target: Type = field(default_factory=lambda: NeRFracModel)

    # Refractive surface parameters
    init_depth: float = -0.6
    """Initial depth (Z-coordinate) of refractive surface.

    Interpretation depends on use_ndc:
    - NDC mode (use_ndc=True): Z in NDC coordinates, range [-1, 0]
      * 0.0 = camera position
      * -1.0 = infinity
      * Typical values: -0.3 (near), -0.6 (mid), -0.9 (far)

    - World mode (use_ndc=False): Z in world coordinates (meters/units)
      * Actual Z-coordinate in your scene's coordinate system
      * Example: 5.0 for water surface at 5 meters height
      * Example: 0.0 for ground-level water

    The refractive field learns an offset dD from this initial estimate.
    """
    refractive_index: float = 1.333
    """Refractive index of the medium below surface (water: 1.333, glass: ~1.5, ice: ~1.31)."""
    air_refractive_index: float = 1.0
    """Refractive index of the medium above surface (typically air: 1.0)."""

    # Positional encoding for refractive field
    multires_origin: int = 8
    """Log2 of max frequency for positional encoding of ray origins (x, y)."""
    multires_refrac_dir: int = 8
    """Log2 of max frequency for positional encoding of ray directions."""

    # Sampling parameters
    num_coarse_samples: int = 64
    """Number of coarse samples per ray."""
    num_importance_samples: int = 64
    """Number of importance samples per ray."""
    eval_num_rays_per_chunk: int = 256
    """Rays per chunk during evaluation."""

    # Rendering
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Background color."""
    raw_noise_std: float = 1.0
    """Std dev of noise added to regularize sigma output during training."""

    # Optional features
    use_gradient_scaling: bool = False
    """Apply distance-based gradient scaling."""
    use_ndc: bool = False
    """Use NDC ray parameterization (for forward-facing scenes like LLFF). Set to True for LLFF datasets."""

    # Camera parameters (needed for NDC transforms)
    image_height: Optional[int] = None
    """Image height in pixels. If None, will be inferred from data."""
    image_width: Optional[int] = None
    """Image width in pixels. If None, will be inferred from data."""
    focal_length: Optional[float] = None
    """Focal length in pixels. If None, will be inferred from data."""


class NeRFracModel(Model):
    """NeRFrac model with learned refractive surface.

    Implements the NeRFrac paper's approach:
    1. Learn a refractive field that predicts distance to the refractive surface
    2. Compute surface normals from neighboring surface points
    3. Apply Snell's law to refract rays at the surface
    4. Sample and render along refracted rays using a radiance field
    """

    config: NeRFracModelConfig

    def __init__(
        self,
        config: NeRFracModelConfig,
        **kwargs,
    ) -> None:
        self.refractive_field: Optional[RefractiveField] = None
        self.radiance_field_coarse: Optional[NeRFField] = None
        self.radiance_field_fine: Optional[NeRFField] = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self) -> None:
        """Initialize fields and samplers."""
        super().populate_modules()

        # Positional encodings for refractive field
        origin_encoding = NeRFEncoding(
            in_dim=2,
            num_frequencies=self.config.multires_origin,
            min_freq_exp=0.0,
            max_freq_exp=float(self.config.multires_origin - 1),
            include_input=True,
        )
        refrac_dir_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=self.config.multires_refrac_dir,
            min_freq_exp=0.0,
            max_freq_exp=float(self.config.multires_refrac_dir - 1),
            include_input=True,
        )

        # Refractive field for surface estimation
        self.refractive_field = RefractiveField(
            origin_encoding=origin_encoding,
            direction_encoding=refrac_dir_encoding,
        )

        # Positional encodings for radiance field
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        )

        # Radiance fields (coarse and fine)
        self.radiance_field_coarse = NeRFField(
            position_encoding=position_encoding, direction_encoding=direction_encoding
        )
        self.radiance_field_fine = NeRFField(
            position_encoding=position_encoding, direction_encoding=direction_encoding
        )

        # Samplers
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

        coord_mode = "NDC" if self.config.use_ndc else "World"
        CONSOLE.log(
            f"NeRFrac initialized: mode={coord_mode}, "
            f"refractive_index={self.config.refractive_index}, "
            f"init_depth={self.config.init_depth}, "
            f"{self.config.num_coarse_samples} coarse + {self.config.num_importance_samples} fine samples"
        )

    def _compute_normals_from_neighbors(self, surface_points_neighbors: Tensor) -> Tensor:
        """Compute surface normals using least squares plane fitting on neighbor points.

        Args:
            surface_points_neighbors: Surface points from 3x3 neighbors [N_rays, 9, 3]
                where index 0 is the center ray and 1-8 are neighbors

        Returns:
            normals: Unit normal vectors [N_rays, 3]
        """
        N = surface_points_neighbors.size(0)

        # Center the points by subtracting mean
        Xs_center = torch.mean(surface_points_neighbors, dim=1, keepdim=True)  # [N_rays, 1, 3]
        Xs_centered = surface_points_neighbors - Xs_center  # [N_rays, 9, 3]

        # Split into xy and z for plane fitting: z = w1*x + w2*y
        pts_xy = Xs_centered[..., :2]  # [N_rays, 9, 2]
        pts_z = Xs_centered[..., 2:3]  # [N_rays, 9, 1]

        # Solve least squares: A @ w = b, where A = pts_xy, b = pts_z
        # Use QR decomposition for numerical stability
        # w = (A^T A)^{-1} A^T b
        AtA = torch.bmm(pts_xy.transpose(1, 2), pts_xy)  # [N_rays, 2, 2]
        Atb = torch.bmm(pts_xy.transpose(1, 2), pts_z)  # [N_rays, 2, 1]

        # Solve using torch.linalg.solve (more stable than inverse)
        w = torch.linalg.solve(AtA, Atb)  # [N_rays, 2, 1]
        w = w.squeeze(-1)  # [N_rays, 2]

        # Normal is [-w1, -w2, 1] normalized
        w1, w2 = w[:, 0:1], w[:, 1:2]
        normals = torch.cat([-w1, -w2, torch.ones_like(w1)], dim=-1)  # [N_rays, 3]
        normals = F.normalize(normals, dim=-1)

        return normals

    def _compute_normals_from_gradients(
        self, surface_points_ndc: Tensor, ray_origins_xy: Tensor, ray_directions: Tensor
    ) -> Tensor:
        """Compute surface normals using gradients of the refractive field.

        This is more elegant than neighbor-based estimation and works with random ray sampling.
        We compute ∂dD/∂x and ∂dD/∂y to get the surface slope.

        Args:
            surface_points_ndc: Surface points in NDC [N_rays, 3]
            ray_origins_xy: Ray origin x,y coordinates [N_rays, 2]
            ray_directions: Ray directions [N_rays, 3]

        Returns:
            normals: Unit normal vectors in NDC space [N_rays, 3]
        """
        # Enable gradients for surface point computation
        ray_origins_xy_grad = ray_origins_xy.detach().clone().requires_grad_(True)
        ray_directions_grad = ray_directions.detach().clone().requires_grad_(True)

        # Query refractive field with gradient tracking
        dD = self.refractive_field(ray_origins_xy_grad, ray_directions_grad)

        # Compute gradients of surface depth offset w.r.t. x and y positions
        grad_outputs = torch.ones_like(dD)
        grads = torch.autograd.grad(
            outputs=dD,
            inputs=ray_origins_xy_grad,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )[0]  # [N_rays, 2]

        # Surface gradient: ∂z/∂x and ∂z/∂y
        dz_dx = grads[:, 0:1]
        dz_dy = grads[:, 1:2]

        # Normal is [-∂z/∂x, -∂z/∂y, 1] normalized
        # This represents the surface normal pointing upward
        normals = torch.cat([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=-1)
        normals = F.normalize(normals, dim=-1)

        return normals

    def _compute_refraction(self, incident_dirs: Tensor, normals: Tensor) -> Tensor:
        """Compute refracted ray directions using Snell's law.

        Args:
            incident_dirs: Incident ray directions (normalized) [N_rays, 3]
            normals: Surface normal vectors (normalized) [N_rays, 3]

        Returns:
            refracted_dirs: Refracted ray directions [N_rays, 3]
        """
        # Ensure inputs are normalized
        incident = F.normalize(incident_dirs, dim=-1)
        normal = F.normalize(normals, dim=-1)

        # Refractive index ratio (air to water)
        eta = self.config.air_refractive_index / self.config.refractive_index

        # Make sure normal points against incident ray (should point "up" from water)
        # If dot(incident, normal) > 0, flip the normal
        cos_i = (incident * normal).sum(dim=-1, keepdim=True)
        normal = torch.where(cos_i > 0, -normal, normal)

        # Recompute cos_i with corrected normal
        cos_i = -(incident * normal).sum(dim=-1, keepdim=True).clamp(min=0.0)

        # Snell's law: sin^2(theta_t) = eta^2 * (1 - cos^2(theta_i))
        sin2_t = eta**2 * (1.0 - cos_i**2)

        # Check for total internal reflection (shouldn't happen for air->water)
        if (sin2_t > 1.0).any():
            CONSOLE.log("[yellow]Warning: Total internal reflection detected (air→water, unexpected)[/yellow]")
            sin2_t = sin2_t.clamp(max=1.0)

        cos_t = torch.sqrt(1.0 - sin2_t)

        # Refracted direction: eta * incident + (eta * cos_i - cos_t) * normal
        refracted = eta * incident + (eta * cos_i - cos_t) * normal
        refracted = F.normalize(refracted, dim=-1)

        return refracted

    def _ndc_to_camera_rays(self, rays_d_ndc: Tensor, origins_ndc: Tensor, H: int, W: int, focal: float) -> Tensor:
        """Convert NDC ray directions to camera space.

        Args:
            rays_d_ndc: Ray directions in NDC [N, 3]
            origins_ndc: Ray origins in NDC [N, 3]
            H: Image height
            W: Image width
            focal: Focal length

        Returns:
            rays_d_cam: Ray directions in camera space [N, 3]
        """
        # Extract origin components
        t1 = -(W * origins_ndc[:, 0]) / (2 * focal)
        t2 = -(H * origins_ndc[:, 1]) / (2 * focal)
        t3 = origins_ndc[:, 2] - 1

        t1 = t1.unsqueeze(1)
        t2 = t2.unsqueeze(1)
        t3 = t3.unsqueeze(1)

        ndc_dz = -t3
        ndc_ray = rays_d_ndc * ndc_dz / (rays_d_ndc[:, 2].unsqueeze(1))
        dxz = -(W * ndc_ray[:, 0].unsqueeze(1)) / (2 * focal) + t1
        dyz = -(H * ndc_ray[:, 1].unsqueeze(1)) / (2 * focal) + t2

        z = -torch.ones_like(dxz)
        x = z * dxz
        y = z * dyz
        d_cam = torch.cat([x, y, z], dim=-1)
        d_cam = F.normalize(d_cam, dim=-1)

        return d_cam

    def _camera_to_ndc_rays(self, rays_d_cam: Tensor, origins_ndc: Tensor, H: int, W: int, focal: float) -> Tensor:
        """Convert camera ray directions to NDC space.

        Args:
            rays_d_cam: Ray directions in camera space [N, 3]
            origins_ndc: Ray origins in NDC [N, 3]
            H: Image height
            W: Image width
            focal: Focal length

        Returns:
            rays_d_ndc: Ray directions in NDC space [N, 3]
        """
        t1 = -(W * origins_ndc[:, 0]) / (2 * focal)
        t2 = -(H * origins_ndc[:, 1]) / (2 * focal)
        t3 = origins_ndc[:, 2] - 1

        t1 = t1.unsqueeze(1)
        t2 = t2.unsqueeze(1)
        t3 = t3.unsqueeze(1)

        dx = rays_d_cam[:, 0].unsqueeze(1)
        dy = rays_d_cam[:, 1].unsqueeze(1)
        dz = rays_d_cam[:, 2].unsqueeze(1)

        ndc_dx = -(2 * focal * (dx / dz - t1)) / W
        ndc_dy = -(2 * focal * (dy / dz - t2)) / H
        ndc_dz = -t3

        d_ndc = torch.cat([ndc_dx, ndc_dy, ndc_dz], dim=-1)

        return d_ndc

    def _ndc_to_camera_points(self, points_ndc: Tensor, H: int, W: int, focal: float) -> Tensor:
        """Convert NDC points to camera space.

        Args:
            points_ndc: Points in NDC [N, 3]
            H: Image height
            W: Image width
            focal: Focal length

        Returns:
            points_cam: Points in camera space [N, 3]
        """
        z = 2 / (points_ndc[:, 2] - 1)
        y = -(H * points_ndc[:, 1] * z) / (2 * focal)
        x = -(W * points_ndc[:, 0] * z) / (2 * focal)

        z = z.unsqueeze(1)
        y = y.unsqueeze(1)
        x = x.unsqueeze(1)

        points_camera = torch.cat([x, y, z], dim=-1)

        return points_camera

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Tensor]:
        """Forward pass with refractive surface rendering.

        Two modes supported:
        - NDC mode (use_ndc=True): Original NeRFrac with NDC↔Camera transforms
        - World mode (use_ndc=False): Direct world-space refraction

        Args:
            ray_bundle: Bundle of rays to render

        Returns:
            Dictionary of outputs including RGB, depth, accumulation
        """
        ray_bundle = ray_bundle.flatten()
        num_rays = len(ray_bundle)
        device = ray_bundle.origins.device

        origins = ray_bundle.origins  # [num_rays, 3]
        directions = F.normalize(ray_bundle.directions, dim=-1)  # [num_rays, 3]

        # Step 1: Query refractive field for surface intersection
        ray_origins_xy = origins[:, :2]  # [num_rays, 2]
        dD = self.refractive_field(ray_origins_xy, directions)  # [num_rays, 1]

        # Compute distance to initial surface
        oz = origins[:, 2:3]
        virz = directions[:, 2:3]
        d = (oz - self.config.init_depth) / (-virz + 1e-8)  # [num_rays, 1]

        # Surface intersection points
        surface_points = origins + (d + dD) * directions  # [num_rays, 3]

        # Step 2: Compute surface normals
        normals = self._compute_normals_from_gradients(
            surface_points, ray_origins_xy, directions
        )  # [num_rays, 3]

        # Step 3 & 4: Refraction calculation (mode-dependent)
        if self.config.use_ndc:
            # NDC MODE: Original NeRFrac with NDC↔Camera transforms
            H = self.config.image_height or 392
            W = self.config.image_width or 392
            focal = self.config.focal_length or 500.0

            # Convert to camera space
            surface_points_cam = self._ndc_to_camera_points(surface_points, H, W, focal)
            incident_dirs_cam = self._ndc_to_camera_rays(directions, origins, H, W, focal)
            normals_cam = normals  # Approximation (TODO: proper transform)

            # Apply Snell's law in camera space
            refracted_dirs_cam = self._compute_refraction(incident_dirs_cam, normals_cam)

            # Convert back to NDC
            refracted_dirs = self._camera_to_ndc_rays(refracted_dirs_cam, surface_points, H, W, focal)
            refracted_dirs = F.normalize(refracted_dirs, dim=-1)
        else:
            # WORLD MODE: Direct refraction in world coordinates
            incident_dirs = F.normalize(directions, dim=-1)
            refracted_dirs = self._compute_refraction(incident_dirs, normals)
            refracted_dirs = F.normalize(refracted_dirs, dim=-1)

        # Refracted ray origins: slightly below surface
        eps = 1e-4 if self.config.use_ndc else 1e-3  # Larger epsilon for world coords
        refracted_origins = surface_points + refracted_dirs * eps

        # Compute remaining distance to far plane
        if ray_bundle.fars is not None:
            remaining_dist = ray_bundle.fars[..., 0] - (d + dD).squeeze(-1)
            remaining_dist = torch.clamp(remaining_dist, min=0.0).unsqueeze(-1)
        else:
            remaining_dist = torch.ones((num_rays, 1), device=device) * 10.0

        # Create refracted ray bundle
        refracted_bundle = RayBundle(
            origins=refracted_origins,
            directions=refracted_dirs,
            pixel_area=ray_bundle.pixel_area,
            camera_indices=ray_bundle.camera_indices,
            nears=torch.zeros((num_rays, 1), device=device),
            fars=remaining_dist,
            metadata=ray_bundle.metadata,
            times=ray_bundle.times,
        )

        # Step 6: Sample along refracted rays (coarse)
        ray_samples_coarse = self.sampler_uniform(refracted_bundle)

        # Query radiance field (coarse)
        field_outputs_coarse = self.radiance_field_coarse(ray_samples_coarse)
        if self.config.use_gradient_scaling:
            field_outputs_coarse = scale_gradients_by_distance_squared(field_outputs_coarse, ray_samples_coarse)

        densities_coarse = field_outputs_coarse[FieldHeadNames.DENSITY]
        rgb_coarse = field_outputs_coarse[FieldHeadNames.RGB]

        # Add noise during training for regularization (like original NeRFrac)
        if self.training and self.config.raw_noise_std > 0:
            noise = torch.randn_like(densities_coarse) * self.config.raw_noise_std
            densities_coarse = densities_coarse + noise

        weights_coarse = ray_samples_coarse.get_weights(densities_coarse)

        rgb_coarse_rendered = self.renderer_rgb(rgb=rgb_coarse, weights=weights_coarse)
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_coarse)

        # Step 7: Fine sampling with PDF sampler (importance sampling)
        ray_samples_fine = self.sampler_pdf(refracted_bundle, ray_samples_coarse, weights_coarse.detach())

        field_outputs_fine = self.radiance_field_fine(ray_samples_fine)
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_fine)

        densities_fine = field_outputs_fine[FieldHeadNames.DENSITY]
        rgb_fine = field_outputs_fine[FieldHeadNames.RGB]

        # Add noise during training
        if self.training and self.config.raw_noise_std > 0:
            noise = torch.randn_like(densities_fine) * self.config.raw_noise_std
            densities_fine = densities_fine + noise

        weights_fine = ray_samples_fine.get_weights(densities_fine)

        rgb_fine_rendered = self.renderer_rgb(rgb=rgb_fine, weights=weights_fine)
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_fine)

        return {
            "rgb_coarse": rgb_coarse_rendered,
            "rgb_fine": rgb_fine_rendered,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "rgb": rgb_fine_rendered,
            "accumulation": accumulation_fine,
            "depth": depth_fine,
            # Additional outputs for debugging/visualization
            "surface_points": surface_points,
            "surface_depth_offset": dD,
            "surface_normals": normals,
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.refractive_field is None or self.radiance_field_coarse is None or self.radiance_field_fine is None:
            raise ValueError("populate_modules() must be called before get_param_groups")

        param_groups["refractive_field"] = list(self.refractive_field.parameters())
        param_groups["radiance_field"] = list(self.radiance_field_coarse.parameters()) + list(
            self.radiance_field_fine.parameters()
        )
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
