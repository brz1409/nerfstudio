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

"""Refractive field for NeRFrac-style refractive surface estimation."""

from typing import Tuple

import torch
from torch import Tensor, nn

from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import Field


class RefractiveField(Field):
    """Refractive Field for learning distance to refractive surface.

    Based on NeRFrac (ICCV 2023): learns a surface distance offset dD such that
    the actual surface point is at: Xs = ray_origin + (init_depth + dD) * ray_direction

    Args:
        origin_encoding: Encoding for ray origin (x, y) coordinates
        direction_encoding: Encoding for ray direction (vx, vy, vz)
        num_layers: Number of MLP layers
        layer_width: Width of MLP layers
        skip_connections: Where to add skip connections
    """

    def __init__(
        self,
        origin_encoding: Encoding = Identity(in_dim=2),
        direction_encoding: Encoding = Identity(in_dim=3),
        num_layers: int = 8,
        layer_width: int = 256,
        skip_connections: Tuple[int, ...] = (4,),
    ) -> None:
        super().__init__()

        self.origin_encoding = origin_encoding
        self.direction_encoding = direction_encoding

        # MLP for refractive surface estimation
        self.mlp = MLP(
            in_dim=self.origin_encoding.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=num_layers,
            layer_width=layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        # Output layer for distance offset
        self.output_layer = nn.Linear(layer_width, 1)

        # Initialize with small weights for flat surface bias (like NeRFrac)
        nn.init.uniform_(self.output_layer.weight, a=-1e-5, b=1e-5)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        ray_origins: Tensor,  # (batch, 2) - x, y coordinates
        ray_directions: Tensor,  # (batch, 3) - direction vectors
    ) -> Tensor:
        """Query refractive field for surface distance offset.

        Args:
            ray_origins: Ray origin coordinates (x, y) [batch, 2]
            ray_directions: Ray direction vectors [batch, 3]

        Returns:
            Distance offset dD [batch, 1]
        """
        encoded_origin = self.origin_encoding(ray_origins)
        encoded_direction = self.direction_encoding(ray_directions)

        # Concatenate encodings
        h = torch.cat([encoded_origin, encoded_direction], dim=-1)

        # Pass through MLP
        h = self.mlp(h)

        # Output distance offset
        dD = self.output_layer(h)

        return dD

    def get_outputs(self, *args, **kwargs):
        """Not used for refractive field, but required by Field interface."""
        raise NotImplementedError("RefractiveField should use forward() directly")
