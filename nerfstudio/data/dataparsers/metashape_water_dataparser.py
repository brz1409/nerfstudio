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

import math
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
    component_id: Optional[str]


@dataclass
class ComponentTransform:
    """Chunk/component similarity transform defined in cameras.xml."""

    rotation: np.ndarray  # 3x3 orthonormal matrix
    translation_ecef: np.ndarray  # 3-vector in meters (ECEF)
    scale: float

    @property
    def inverse_rotation(self) -> np.ndarray:
        return self.rotation.T


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
        component_id = cam.get("component_id")
        if reference is not None and all(key in reference.attrib for key in ("x", "y", "z")):
            ref_utm = np.array(
                [float(reference.get("x")), float(reference.get("y")), float(reference.get("z"))],
                dtype=float,
            )
        rows.append(
            CameraRow(label=cam.get("label", ""), matrix=matrix, ref_utm=ref_utm, component_id=component_id)
        )
    return rows


def _parse_component_transforms(cameras_xml: Path) -> Dict[str, ComponentTransform]:
    """Extract chunk/component transforms from cameras.xml."""
    root = ET.parse(str(cameras_xml)).getroot()
    component_map: Dict[str, ComponentTransform] = {}

    # components live under chunk/components/component
    chunk = root.find(".//chunk")
    if chunk is None:
        return component_map

    components = chunk.find("components")
    if components is None:
        return component_map

    for component in components.iter("component"):
        component_id = component.get("id")
        if component_id is None:
            continue

        rotation_xml = component.find("transform/rotation")
        translation_xml = component.find("transform/translation")
        scale_xml = component.find("transform/scale")

        if rotation_xml is None or translation_xml is None or rotation_xml.text is None or translation_xml.text is None:
            continue

        rotation = np.array([float(x) for x in rotation_xml.text.split()], dtype=float).reshape(3, 3)
        translation = np.array([float(x) for x in translation_xml.text.split()], dtype=float)
        scale = float(scale_xml.text) if (scale_xml is not None and scale_xml.text is not None) else 1.0

        component_map[component_id] = ComponentTransform(
            rotation=rotation,
            translation_ecef=translation,
            scale=scale,
        )

    return component_map


def _select_component_transform(cameras: List[CameraRow], component_map: Dict[str, ComponentTransform]) -> ComponentTransform:
    """Return the unique component transform referenced by the provided cameras."""
    component_ids = {cam.component_id for cam in cameras if cam.component_id is not None}
    if not component_ids:
        return ComponentTransform(rotation=np.eye(3), translation_ecef=np.zeros(3), scale=1.0)
    if len(component_ids) > 1:
        raise ValueError("MetashapeWaterDataParser found multiple component_ids across reference cameras.")

    component_id = component_ids.pop()
    if component_id not in component_map:
        raise ValueError(f"Component transform '{component_id}' not found in cameras.xml.")
    return component_map[component_id]


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


def _utm_to_latlon(easting: float, northing: float, zone: int, northern: bool = True) -> Tuple[float, float]:
    """Convert UTM coordinates to geodetic latitude/longitude in degrees."""
    k0 = 0.9996
    a = 6378137.0
    e_sq = 6.69437999014e-3
    e = math.sqrt(e_sq)
    e_prime_sq = e_sq / (1 - e_sq)

    x = easting - 500000.0
    y = northing
    if not northern:
        y -= 10000000.0

    m = y / k0
    mu = m / (
        a
        * (
            1
            - e_sq / 4.0
            - 3 * e_sq**2 / 64.0
            - 5 * e_sq**3 / 256.0
        )
    )

    e1 = (1 - math.sqrt(1 - e_sq)) / (1 + math.sqrt(1 - e_sq))
    j1 = (3 * e1 / 2) - (27 * e1**3 / 32)
    j2 = (21 * e1**2 / 16) - (55 * e1**4 / 32)
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512

    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)

    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)

    c1 = e_prime_sq * cos_fp**2
    t1 = tan_fp**2
    n1 = a / math.sqrt(1 - e_sq * sin_fp**2)
    r1 = a * (1 - e_sq) / ((1 - e_sq * sin_fp**2) ** 1.5)
    d = x / (n1 * k0)

    lat_rad = fp - (n1 * tan_fp / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e_prime_sq) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * e_prime_sq - 3 * c1**2) * d**6 / 720
    )
    lon_rad = (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * e_prime_sq + 24 * t1**2) * d**5 / 120
    ) / cos_fp

    lon_origin = math.radians((zone - 1) * 6 - 180 + 3)

    lat = math.degrees(lat_rad)
    lon = math.degrees(lon_origin + lon_rad)
    return lat, lon


def _latlon_to_ecef(lat_deg: float, lon_deg: float, height: float) -> np.ndarray:
    """Convert latitude/longitude (degrees) and ellipsoidal height to ECEF coordinates."""
    a = 6378137.0
    e_sq = 6.69437999014e-3

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    n = a / math.sqrt(1 - e_sq * sin_lat**2)

    x = (n + height) * cos_lat * cos_lon
    y = (n + height) * cos_lat * sin_lon
    z = (n * (1 - e_sq) + height) * sin_lat

    return np.array([x, y, z], dtype=float)


def _utm_to_ecef(easting: float, northing: float, altitude: float, zone: int) -> Tuple[np.ndarray, float, float]:
    """Convert UTM coordinates (ETRS89/WGS84 ellipsoid) to ECEF and return intermediate lat/lon."""
    lat_deg, lon_deg = _utm_to_latlon(easting, northing, zone, northern=True)
    ecef = _latlon_to_ecef(lat_deg, lon_deg, altitude)
    return ecef, lat_deg, lon_deg


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
    component_transform: ComponentTransform


# ---------------------------------------------------------------------------
# Core conversion

def compute_water_markers_ns_world(
    cameras_xml: Path,
    reference_xml: Path,
    markers_xml: Optional[Path],
    applied_transform_4x4: Optional[np.ndarray],
    utm_zone: int,
) -> WaterMarkersNSWorldResult:
    """Project Metashape water markers into NS World space.

    This function handles the transformation from UTM coordinates to NS World by following the same
    geodetic pipeline used in `ns-process-data metashape`:
    1. Convert UTM → latitude/longitude → ECEF.
    2. Apply the component (chunk) similarity transform from cameras.xml.
    3. Apply the dataset-level applied_transform recorded in transforms.json.
    Args:
        cameras_xml: Path to Metashape cameras.xml with camera transforms and UTM references
        reference_xml: Path to Metashape reference.xml with marker UTM coordinates
        markers_xml: Optional path to markers.xml that overrides coordinates from reference.xml
        applied_transform_4x4: The 4x4 applied_transform from transforms.json (or default)
        utm_zone: UTM zone (e.g. 33) used for the coordinate conversion.

    Returns:
        WaterMarkersNSWorldResult containing markers and plane in NS World space

    Raises:
        ValueError: If no cameras with references, or no markers found
    """
    cameras = [row for row in _parse_cameras_with_refs(cameras_xml) if row.ref_utm is not None]
    if len(cameras) == 0:
        raise ValueError("MetashapeWaterDataParser requires at least one camera with reference coordinates.")

    component_map = _parse_component_transforms(cameras_xml)
    component_transform = _select_component_transform(cameras, component_map)

    applied_transform = np.array(
        applied_transform_4x4 if applied_transform_4x4 is not None else default_applied_transform(),
        dtype=float,
    )
    if applied_transform.shape == (3, 4):
        applied_transform = _to_homogeneous_4x4(applied_transform)

    markers_ref = _parse_markers(reference_xml, markers_xml)
    if len(markers_ref) == 0:
        raise ValueError("MetashapeWaterDataParser could not find any water markers in the provided XML files.")

    labels = sorted(markers_ref.keys())
    points_ns = []
    for label in labels:
        utm_point = markers_ref[label]
        ecef_point, _, _ = _utm_to_ecef(utm_point[0], utm_point[1], utm_point[2], zone=utm_zone)

        # ECEF -> chunk local
        chunk_local = component_transform.inverse_rotation @ (
            (ecef_point - component_transform.translation_ecef) / component_transform.scale
        )
        # chunk local -> component space (same step used when exporting transforms.json)
        component_space = component_transform.rotation @ chunk_local + (
            component_transform.translation_ecef / component_transform.scale
        )
        # component space -> NS world (before auto orient/scale)
        point_ns = apply_4x4(applied_transform, component_space)
        points_ns.append(point_ns)

    points_ns = np.array(points_ns)
    plane_ns: Optional[Tuple[np.ndarray, float]] = None
    if len(labels) >= 3:
        plane_ns = plane_from_points(points_ns)

    markers_ns_world = {label: points_ns[idx] for idx, label in enumerate(labels)}

    return WaterMarkersNSWorldResult(
        markers_ns_world=markers_ns_world,
        plane_ns_world=plane_ns,
        component_transform=component_transform,
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
    utm_zone: int = 33
    """UTM zone used for cameras and markers (default: 33 for ETRS89 / UTM 33N)."""


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

        cameras = [row for row in _parse_cameras_with_refs(cameras_xml) if row.ref_utm is not None]
        if len(cameras) == 0:
            raise ValueError("Need at least one camera with UTM references")

        component_map = _parse_component_transforms(cameras_xml)
        component_transform = _select_component_transform(cameras, component_map)

        applied_transform = transforms_json.get("applied_transform")
        if applied_transform is not None:
            applied_transform_np = np.array(applied_transform, dtype=float)
            if applied_transform_np.shape == (3, 4):
                applied_transform_np = _to_homogeneous_4x4(applied_transform_np)
        else:
            applied_transform_np = default_applied_transform()
        applied_transform_np = np.array(applied_transform_np, dtype=float)

        markers_ref = _parse_markers(reference_xml, None)
        if len(markers_ref) == 0:
            raise ValueError("No markers found in reference.xml")

        marker_labels = sorted(markers_ref.keys())
        markers_transforms = []
        for label in marker_labels:
            utm_point = markers_ref[label]
            ecef_point, _, _ = _utm_to_ecef(utm_point[0], utm_point[1], utm_point[2], zone=self.config.utm_zone)
            chunk_local = component_transform.inverse_rotation @ (
                (ecef_point - component_transform.translation_ecef) / component_transform.scale
            )
            component_space = component_transform.rotation @ chunk_local + (
                component_transform.translation_ecef / component_transform.scale
            )
            pos_transforms = apply_4x4(applied_transform_np, component_space)
            markers_transforms.append(pos_transforms)

        markers_transforms = np.stack(markers_transforms)

        CONSOLE.log(f"[cyan]Markers (transforms.json coords, BEFORE auto_orient):[/cyan]")
        CONSOLE.log(f"  Range: [{markers_transforms.min(axis=0)}, {markers_transforms.max(axis=0)}]")
        CONSOLE.log(f"  Mean: {markers_transforms.mean(axis=0)}")

        # Apply auto_orient_and_center_poses (mimics nerfstudio_dataparser.py:237-241)
        camera_poses = []
        for frame in transforms_json.get("frames", []):
            matrix = np.array(frame["transform_matrix"], dtype=float)
            if matrix.shape == (3, 4):
                matrix = np.vstack([matrix, [0, 0, 0, 1]])
            camera_poses.append(matrix)
        if len(camera_poses) == 0:
            raise ValueError("No camera poses found in transforms.json")

        camera_poses_np = np.array(camera_poses)
        poses_torch = torch.from_numpy(camera_poses_np.astype(np.float32))
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

        transform_matrix_np = transform_matrix.numpy()
        transform_4x4 = _to_homogeneous_4x4(transform_matrix_np)

        markers_model = {}
        marker_positions = []

        for label, pos_transforms in zip(marker_labels, markers_transforms):
            pos_4d = np.append(pos_transforms, 1.0)
            pos_oriented = (transform_4x4 @ pos_4d)[:3]
            pos_model = pos_oriented * scale_factor
            markers_model[label] = pos_model
            marker_positions.append(pos_model)

        marker_positions = np.array(marker_positions)
        plane_model = plane_from_points(marker_positions)

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
                utm_zone=self.config.utm_zone,
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
