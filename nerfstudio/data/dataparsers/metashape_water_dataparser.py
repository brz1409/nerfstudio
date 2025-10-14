# Copyright 2022 the Regents of the University of California,
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

"""Utilities to convert Metashape water markers into Nerfstudio dataset space."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import xml.etree.ElementTree as ET

from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    Nerfstudio as NerfstudioDataParser,
    NerfstudioDataParserConfig,
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


# ---------------------------------------------------------------------------
# Linear algebra helpers

def umeyama(A: np.ndarray, B: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return similarity transform (s, R, t) that maps A → B in least squares sense."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    assert A.shape == B.shape and A.ndim == 2 and A.shape[1] == 3

    n = A.shape[0]
    muA, muB = A.mean(0), B.mean(0)
    X, Y = A - muA, B - muB
    U, S, Vt = np.linalg.svd((Y.T @ X) / n)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    varA = (X**2).sum() / n
    scale = (np.trace(np.diag(S) @ D) / varA) if with_scale else 1.0
    translation = muB - scale * (R @ muA)
    return float(scale), R, translation


def plane_from_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit plane to points (>=3) using least squares; returns (normal, d) with ||normal||=1."""
    assert points.shape[0] >= 3 and points.shape[1] == 3
    centroid = points.mean(0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ centroid)
    return normal, offset


# ---------------------------------------------------------------------------
# Parsing helpers

@dataclass
class CameraRow:
    """Single camera entry exported by Metashape."""

    label: str
    matrix: np.ndarray  # 4x4 camera-to-world transform
    ref_utm: Optional[np.ndarray]  # 3-vector world position in UTM coordinates


def _parse_cameras_with_refs(cameras_xml: Path) -> List[CameraRow]:
    root = ET.parse(str(cameras_xml)).getroot()
    rows: List[CameraRow] = []
    for cam in root.findall(".//camera"):
        transform_text = cam.findtext("transform")
        if transform_text is None:
            continue
        matrix = np.fromstring(transform_text, sep=" ").reshape(4, 4)
        ref_utm = None
        reference = cam.find("reference")
        if reference is not None and all(key in reference.attrib for key in ("x", "y", "z")):
            ref_utm = np.array(
                [float(reference.get("x")), float(reference.get("y")), float(reference.get("z"))],
                dtype=float,
            )
        rows.append(CameraRow(label=cam.get("label", ""), matrix=matrix, ref_utm=ref_utm))
    return rows


def _parse_markers(reference_xml: Path, markers_xml: Optional[Path]) -> Dict[str, np.ndarray]:
    """Load marker positions from reference.xml (and optional markers.xml override)."""

    def _load(path: Path, tag: str) -> Dict[str, np.ndarray]:
        root = ET.parse(str(path)).getroot()
        markers: Dict[str, np.ndarray] = {}
        for marker in root.findall(tag):
            label = marker.get("label", "")
            reference = marker.find("reference")
            if label and reference is not None and all(key in reference.attrib for key in ("x", "y", "z")):
                markers[label] = np.array(
                    [float(reference.get("x")), float(reference.get("y")), float(reference.get("z"))],
                    dtype=float,
                )
        return markers

    base_markers = _load(reference_xml, ".//markers/marker")
    if markers_xml is not None and markers_xml.exists():
        override_markers = _load(markers_xml, ".//marker")
        base_markers.update(override_markers)
    return base_markers


# ---------------------------------------------------------------------------
# Coordinate transforms

def camera_center_from_c2w(matrix: np.ndarray) -> np.ndarray:
    """Extract camera center from camera-to-world transform (translation component)."""
    return matrix[:3, 3]


def camera_center_from_w2c(matrix: np.ndarray) -> np.ndarray:
    """Extract camera center from world-to-camera transform."""
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    return -rotation.T @ translation


def default_applied_transform() -> np.ndarray:
    """Axis permutation from Metashape/OpenCV to Nerfstudio/OpenGL (Z, X, Y)."""
    return np.array(
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=float,
    )


def apply_4x4(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply homogeneous 4x4 transform to a 3-vector."""
    homogeneous = np.concatenate([point, [1.0]])
    return (matrix @ homogeneous)[:3]


def apply_similarity(scale: float, rotation: np.ndarray, translation: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply similarity transform with scale, rotation, translation to a set of points."""
    points = np.atleast_2d(points)
    return (scale * (rotation @ points.T)).T + translation


# ---------------------------------------------------------------------------
# Result container

@dataclass
class WaterMarkersNSWorldResult:
    """Water markers in NS World space (after applied_transform, before auto_orient)."""
    markers_ns_world: Dict[str, np.ndarray]
    plane_ns_world: Optional[Tuple[np.ndarray, float]]
    utm_to_local: Tuple[float, np.ndarray, np.ndarray]  # (scale, rotation, translation)


# ---------------------------------------------------------------------------
# Core conversion

def compute_water_markers_ns_world(
    cameras_xml: Path,
    reference_xml: Path,
    markers_xml: Optional[Path],
    applied_transform_4x4: Optional[np.ndarray],
) -> WaterMarkersNSWorldResult:
    """Project Metashape water markers into NS World space.

    This function handles the transformation from UTM coordinates to NS World:
    1. UTM → Local (via Umeyama from camera UTM refs to camera local positions)
    2. Local → NS World (via applied_transform)

    The transformation to final training space is handled separately by the dataparser.

    Args:
        cameras_xml: Path to Metashape cameras.xml with camera transforms and UTM references
        reference_xml: Path to Metashape reference.xml with marker UTM coordinates
        markers_xml: Optional path to markers.xml that overrides coordinates from reference.xml
        applied_transform_4x4: The 4x4 applied_transform from transforms.json (or default)

    Returns:
        WaterMarkersNSWorldResult containing markers and plane in NS World space

    Raises:
        ValueError: If fewer than 3 cameras with references, or no markers found
    """
    # 1. Load cameras with UTM references
    cameras = [row for row in _parse_cameras_with_refs(cameras_xml) if row.ref_utm is not None]
    if len(cameras) < 3:
        raise ValueError("MetashapeWaterDataParser requires at least three cameras with reference coordinates.")

    # 2. Extract camera positions and fit Umeyama (UTM → Local)
    utm_positions = np.stack([row.ref_utm for row in cameras])
    local_centers_c2w = np.stack([camera_center_from_c2w(row.matrix) for row in cameras])
    local_centers_w2c = np.stack([camera_center_from_w2c(row.matrix) for row in cameras])

    s_c2w, R_c2w, t_c2w = umeyama(utm_positions, local_centers_c2w, with_scale=True)
    s_w2c, R_w2c, t_w2c = umeyama(utm_positions, local_centers_w2c, with_scale=True)

    rms_c2w = np.linalg.norm(apply_similarity(s_c2w, R_c2w, t_c2w, utm_positions) - local_centers_c2w, axis=1).mean()
    rms_w2c = np.linalg.norm(apply_similarity(s_w2c, R_w2c, t_w2c, utm_positions) - local_centers_w2c, axis=1).mean()

    # Choose best fit (c2w vs w2c)
    similarity = (s_c2w, R_c2w, t_c2w) if rms_c2w <= rms_w2c else (s_w2c, R_w2c, t_w2c)

    # 3. Load markers from reference.xml
    markers_ref = _parse_markers(reference_xml, markers_xml)
    if len(markers_ref) == 0:
        raise ValueError("MetashapeWaterDataParser could not find any water markers in the provided XML files.")

    labels = sorted(markers_ref.keys())
    points_utm = np.stack([markers_ref[label] for label in labels])

    # 4. Transform UTM → Local
    points_local = apply_similarity(*similarity, points_utm)

    # 5. Transform Local → NS World
    applied_transform = np.array(applied_transform_4x4 if applied_transform_4x4 is not None else default_applied_transform())
    points_ns = np.stack([apply_4x4(applied_transform, point) for point in points_local])

    # 6. Compute plane in NS World
    plane_ns: Optional[Tuple[np.ndarray, float]] = None
    if len(labels) >= 3:
        plane_ns = plane_from_points(points_ns)

    # 7. Build result
    markers_ns_world = {label: points_ns[idx] for idx, label in enumerate(labels)}

    return WaterMarkersNSWorldResult(
        markers_ns_world=markers_ns_world,
        plane_ns_world=plane_ns,
        utm_to_local=similarity,
    )


# ---------------------------------------------------------------------------
# Helper function for coordinate transformation

def _to_homogeneous_4x4(transform_3x4: np.ndarray) -> np.ndarray:
    """Expand a 3x4 matrix to 4x4 homogeneous form."""
    assert transform_3x4.shape == (3, 4), f"Expected (3, 4), got {transform_3x4.shape}"
    transform_4x4 = np.eye(4, dtype=float)
    transform_4x4[:3, :] = transform_3x4
    return transform_4x4


# ---------------------------------------------------------------------------
# Dataparser wiring


@dataclass
class MetashapeWaterDataParserConfig(NerfstudioDataParserConfig):
    """Config for MetashapeWaterDataParser."""

    _target: Type = field(default_factory=lambda: MetashapeWaterDataParser)
    cameras_xml: Optional[Path] = None
    """Path to Metashape `cameras.xml` containing camera transforms with UTM references."""
    reference_xml: Optional[Path] = None
    """Path to Metashape `reference.xml` containing marker coordinates."""
    markers_xml: Optional[Path] = None
    """Optional extra `markers.xml` file whose entries override coordinates from `reference.xml`."""
    require_water_plane: bool = True
    """Raise an error when no water plane can be estimated."""
    store_full_marker_metadata: bool = False
    """Save the full marker bundle inside metadata for debugging."""


class MetashapeWaterDataParser(NerfstudioDataParser):
    """Dataparser that adds water-plane metadata derived from Metashape markers."""

    config: MetashapeWaterDataParserConfig

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        outputs = super()._generate_dataparser_outputs(split=split, **kwargs)
        water_metadata = self._compute_water_surface_metadata(outputs)
        if water_metadata is None:
            return outputs

        metadata = dict(outputs.metadata)
        metadata["water_surface"] = water_metadata
        return replace(outputs, metadata=metadata)

    def save_dataparser_transform(self, path: Path) -> None:
        """Override to save water surface metadata along with transform.

        This ensures markers_model and plane_model are persisted to disk
        so they can be visualized and used consistently across training runs.
        """
        # First, generate outputs to get the metadata
        outputs = self._generate_dataparser_outputs(split="train")

        # Build the data dict with transform, scale, AND metadata
        import json
        data = {
            "transform": outputs.dataparser_transform.tolist(),
            "scale": float(outputs.dataparser_scale),
        }

        # Add water surface metadata if available
        if "water_surface" in outputs.metadata:
            data["water_surface"] = outputs.metadata["water_surface"]

        # Write to file
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, "w", encoding="UTF-8") as file:
            json.dump(data, file, indent=4)

        CONSOLE.log(f"[cyan]Saved dataparser transform with water surface metadata to {path}[/cyan]")

    # ------------------------------------------------------------------
    # Internal helpers

    def _transform_markers_to_model_space(
        self,
        markers_ns_world: Dict[str, np.ndarray],
        transforms_json: Dict,
        cameras_xml: Path,
        reference_xml: Path,
    ) -> Tuple[Dict[str, np.ndarray], Tuple[np.ndarray, float]]:
        """Transform water markers from UTM to final training space.

        IMPORTANT: This function RE-DOES the marker transformation, because the
        markers from compute_water_markers_ns_world() are in the wrong coordinate system!

        We need to fit the markers directly to transforms.json cameras, not CAMERAS.xml.

        Args:
            markers_ns_world: Markers dict (keys only, we'll reload from reference.xml)
            transforms_json: The transforms.json dict (to get camera poses)
            cameras_xml: Path to CAMERAS.xml (for UTM references)
            reference_xml: Path to REFERENCE.xml (for marker UTM coords)

        Returns:
            (markers_model, plane_model) where:
            - markers_model: Dict mapping label -> 3D position in training space
            - plane_model: (normal, d) of water plane in training space
        """
        import torch
        from nerfstudio.cameras import camera_utils

        # Step 1: Load camera UTM references from CAMERAS.xml
        cameras = [row for row in _parse_cameras_with_refs(cameras_xml) if row.ref_utm is not None]
        if len(cameras) < 3:
            raise ValueError("Need at least 3 cameras with UTM references")

        camera_utm = np.stack([row.ref_utm for row in cameras])
        camera_labels = [row.label for row in cameras]

        # Step 2: Load camera positions from transforms.json (match by label if possible, or use order)
        camera_poses_dict = {}
        for frame in transforms_json.get("frames", []):
            file_path = frame.get("file_path", "")
            # Extract label from file_path (e.g., "images/IMG_001.jpg" -> "IMG_001")
            label = Path(file_path).stem
            M = np.array(frame["transform_matrix"], dtype=float)
            if M.shape == (3, 4):
                M = np.vstack([M, [0, 0, 0, 1]])
            camera_poses_dict[label] = M

        # Match cameras by label or use order
        camera_poses = []
        camera_transforms_positions = []
        for label in camera_labels:
            if label in camera_poses_dict:
                camera_poses.append(camera_poses_dict[label])
                camera_transforms_positions.append(camera_poses_dict[label][:3, 3])
            else:
                # Fallback: try to match by index (not ideal but better than crashing)
                idx = len(camera_poses)
                if idx < len(camera_poses_dict):
                    all_poses = list(camera_poses_dict.values())
                    camera_poses.append(all_poses[idx])
                    camera_transforms_positions.append(all_poses[idx][:3, 3])

        if len(camera_poses) == 0:
            raise ValueError("No matching camera poses found in transforms.json")

        camera_poses = np.array(camera_poses)
        camera_transforms_positions = np.array(camera_transforms_positions)

        # Step 3: Fit Umeyama: UTM → transforms.json positions (NOT CAMERAS.xml local!)
        s_fit, R_fit, t_fit = umeyama(camera_utm, camera_transforms_positions, with_scale=True)

        CONSOLE.log("[cyan]═══ Marker Transformation Debug (FIXED) ═══[/cyan]")
        CONSOLE.log(f"[cyan]Umeyama fit (UTM → transforms.json):[/cyan]")
        CONSOLE.log(f"  Scale: {s_fit:.6f}")
        CONSOLE.log(f"  Camera UTM mean: {camera_utm.mean(axis=0)}")
        CONSOLE.log(f"  Camera transforms.json mean: {camera_transforms_positions.mean(axis=0)}")

        # Step 4: Load markers from reference.xml (UTM coordinates)
        markers_ref = _parse_markers(reference_xml, None)
        if len(markers_ref) == 0:
            raise ValueError("No markers found in reference.xml")

        marker_labels = sorted(markers_ref.keys())
        markers_utm = np.stack([markers_ref[label] for label in marker_labels])

        # Step 5: Transform markers: UTM → transforms.json coordinate system
        markers_transforms = apply_similarity(s_fit, R_fit, t_fit, markers_utm)

        CONSOLE.log(f"[cyan]Markers (transforms.json coords, BEFORE auto_orient):[/cyan]")
        CONSOLE.log(f"  Range: [{markers_transforms.min(axis=0)}, {markers_transforms.max(axis=0)}]")
        CONSOLE.log(f"  Mean: {markers_transforms.mean(axis=0)}")

        # Apply auto_orient_and_center_poses (mimics nerfstudio_dataparser.py:237-241)
        poses_torch = torch.from_numpy(camera_poses.astype(np.float32))
        oriented_poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses_torch,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Compute scale_factor (mimics nerfstudio_dataparser.py:244-249)
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(oriented_poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        # Step 6: Apply same transformation to markers (auto_orient + scale)
        transform_matrix_np = transform_matrix.numpy()
        transform_4x4 = _to_homogeneous_4x4(transform_matrix_np)

        markers_model = {}
        marker_positions = []

        for label in marker_labels:
            # Get marker position in transforms.json coords (before auto_orient)
            idx = marker_labels.index(label)
            pos_transforms = markers_transforms[idx]

            # Apply transform_matrix (auto_orient)
            pos_4d = np.append(pos_transforms, 1.0)
            pos_oriented = (transform_4x4 @ pos_4d)[:3]

            # Apply scale_factor
            pos_model = pos_oriented * scale_factor
            markers_model[label] = pos_model
            marker_positions.append(pos_model)

        # Compute plane from transformed markers
        marker_positions = np.array(marker_positions)
        plane_model = plane_from_points(marker_positions)

        # DEBUG: Log AFTER transformation
        camera_positions_model = oriented_poses[:, :3, 3].numpy() * scale_factor
        CONSOLE.log(f"[green]Camera positions (Model Space):[/green]")
        CONSOLE.log(f"  Range: [{camera_positions_model.min(axis=0)}, {camera_positions_model.max(axis=0)}]")
        CONSOLE.log(f"  Mean: {camera_positions_model.mean(axis=0)}")

        CONSOLE.log(f"[green]Marker positions (Model Space):[/green]")
        CONSOLE.log(f"  Range: [{marker_positions.min(axis=0)}, {marker_positions.max(axis=0)}]")
        CONSOLE.log(f"  Mean: {marker_positions.mean(axis=0)}")
        CONSOLE.log(f"[green]Scale factor: {scale_factor:.6f}[/green]")
        CONSOLE.log("[cyan]═══════════════════════════════════[/cyan]")

        return markers_model, plane_model

    def _compute_water_surface_metadata(self, outputs) -> Optional[Dict]:
        """Compute water surface metadata using the new simplified pipeline.

        Pipeline:
        1. Load markers in NS World (compute_water_markers_ns_world)
        2. Transform to model space (_transform_markers_to_model_space)
        3. Build metadata dict
        """
        cameras_xml = self._resolve_path(self.config.cameras_xml)
        reference_xml = self._resolve_path(self.config.reference_xml)
        markers_xml = self._resolve_path(self.config.markers_xml)

        if cameras_xml is None or reference_xml is None:
            message = (
                "MetashapeWaterDataParser requires both `cameras_xml` and `reference_xml` to compute the water plane."
            )
            if self.config.require_water_plane:
                raise ValueError(message)
            CONSOLE.log(f"[yellow]{message}[/yellow]")
            return None

        if not cameras_xml.exists() or not reference_xml.exists():
            missing = [str(path) for path in (cameras_xml, reference_xml) if not path.exists()]
            raise FileNotFoundError(f"MetashapeWaterDataParser missing required XML file(s): {missing}")

        # Load transforms.json
        transforms_json = self._load_transforms_json()
        applied_transform = transforms_json.get("applied_transform")
        if applied_transform is not None:
            applied_transform_np = np.array(applied_transform, dtype=float)
            if applied_transform_np.shape == (3, 4):
                applied_transform_np = _to_homogeneous_4x4(applied_transform_np)
        else:
            applied_transform_np = None

        try:
            # Step 1: Compute markers in NS World
            water_result_ns = compute_water_markers_ns_world(
                cameras_xml=cameras_xml,
                reference_xml=reference_xml,
                markers_xml=markers_xml,
                applied_transform_4x4=applied_transform_np,
            )

            # Step 2: Transform to model space (pass XMLs to refit markers directly)
            markers_model, plane_model = self._transform_markers_to_model_space(
                markers_ns_world=water_result_ns.markers_ns_world,  # Only used for keys
                transforms_json=transforms_json,
                cameras_xml=cameras_xml,
                reference_xml=reference_xml,
            )

        except Exception as exc:  # pylint: disable=broad-except
            if self.config.require_water_plane:
                raise
            CONSOLE.log(f"[yellow]MetashapeWaterDataParser: Failed to compute water plane ({exc}).[/yellow]")
            return None

        # Step 3: Build metadata
        return self._build_water_surface_dict(
            markers_ns_world=water_result_ns.markers_ns_world,
            markers_model=markers_model,
            plane_ns_world=water_result_ns.plane_ns_world,
            plane_model=plane_model,
        )

    def _build_water_surface_dict(
        self,
        markers_ns_world: Dict[str, np.ndarray],
        markers_model: Dict[str, np.ndarray],
        plane_ns_world: Optional[Tuple[np.ndarray, float]],
        plane_model: Tuple[np.ndarray, float],
    ) -> Dict:
        """Build metadata dict from water markers and plane.

        Args:
            markers_ns_world: Markers in NS World coordinates
            markers_model: Markers in model (training) coordinates
            plane_ns_world: Plane in NS World coordinates (normal, d)
            plane_model: Plane in model coordinates (normal, d)

        Returns:
            Metadata dict to be stored in dataparser outputs
        """
        assert plane_model is not None, "plane_model must be available when building metadata."

        metadata: Dict[str, Dict] = {
            "plane_model": {
                "normal": plane_model[0].tolist(),
                "d": float(plane_model[1]),
            },
            "source": "metashape_water_dataparser",
            "marker_count": len(markers_ns_world),
            "marker_labels": sorted(markers_ns_world.keys()),
        }

        if plane_ns_world is not None:
            metadata["plane_world"] = {
                "normal": plane_ns_world[0].tolist(),
                "d": float(plane_ns_world[1]),
            }

        if markers_ns_world:
            metadata["markers_ns_world"] = {
                label: value.tolist() for label, value in markers_ns_world.items()
            }

        if markers_model:
            metadata["markers_model"] = {
                label: value.tolist() for label, value in markers_model.items()
            }

        return metadata

    def _resolve_path(self, value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return None
        value = Path(value)
        if value.is_absolute():
            return value
        candidate = self.config.data / value
        return candidate if candidate.exists() else value

    def _load_transforms_json(self) -> Dict:
        data_path = self.config.data
        if data_path.suffix == ".json":
            return load_from_json(data_path)
        return load_from_json(data_path / "transforms.json")
