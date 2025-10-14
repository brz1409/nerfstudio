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


def _extract_chunk_transform(paths: List[Path]) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """Look for <chunk><transform> blocks and extract (scale, rotation, translation)."""
    for candidate in paths:
        if candidate is None or not candidate.exists():
            continue
        try:
            root = ET.parse(str(candidate)).getroot()
        except ET.ParseError:
            continue
        chunk_transform = root.find(".//chunk/transform")
        if chunk_transform is None:
            continue
        rotation_text = chunk_transform.findtext("rotation")
        translation_text = chunk_transform.findtext("translation")
        scale_text = chunk_transform.findtext("scale")
        if rotation_text is None or translation_text is None or scale_text is None:
            continue
        try:
            rotation = np.fromstring(rotation_text, sep=" ").reshape(3, 3)
            translation = np.fromstring(translation_text, sep=" ")
            scale = float(scale_text)
        except ValueError:
            continue
        return float(scale), rotation, translation
    return None


# ---------------------------------------------------------------------------
# Coordinate transforms

def camera_center_from_c2w(matrix: np.ndarray) -> np.ndarray:
    """Extract camera center from camera-to-world transform (translation component)."""
    return matrix[:3, 3]


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
class WaterMarkersResult:
    markers_ns_world: Dict[str, np.ndarray]
    markers_model: Optional[Dict[str, np.ndarray]]
    plane_ns_world: Optional[Tuple[np.ndarray, float]]
    plane_model: Optional[Tuple[np.ndarray, float]]
    utm_to_local: Tuple[float, np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Core conversion

def compute_water_markers(
    cameras_xml: Path,
    reference_xml: Path,
    markers_xml: Optional[Path],
    applied_transform_4x4: Optional[np.ndarray],
    dataparser_transform_4x4: Optional[np.ndarray],
    scene_scale: Optional[float],
) -> WaterMarkersResult:
    """Project Metashape water markers into Nerfstudio dataset space."""

    cameras = [row for row in _parse_cameras_with_refs(cameras_xml) if row.ref_utm is not None]
    if len(cameras) < 3:
        raise ValueError("MetashapeWaterDataParser requires at least three cameras with reference coordinates.")

    utm_positions = np.stack([row.ref_utm for row in cameras])
    local_centers = np.stack([camera_center_from_c2w(row.matrix) for row in cameras])
    similarity = umeyama(utm_positions, local_centers, with_scale=True)
    candidate_paths = [cameras_xml, reference_xml]
    if markers_xml is not None:
        candidate_paths.append(markers_xml)
    chunk_transform = _extract_chunk_transform(candidate_paths)
    if chunk_transform is not None:
        s_chunk, R_chunk, t_chunk = chunk_transform
        if s_chunk != 0:
            s_inv = 1.0 / float(s_chunk)
            R_inv = R_chunk.T
            t_inv = -(R_inv @ t_chunk) * s_inv
            similarity = (s_inv, R_inv, t_inv)

    markers_ref = _parse_markers(reference_xml, markers_xml)
    if len(markers_ref) == 0:
        raise ValueError("MetashapeWaterDataParser could not find any water markers in the provided XML files.")

    labels = sorted(markers_ref.keys())
    points_utm = np.stack([markers_ref[label] for label in labels])
    points_local = apply_similarity(*similarity, points_utm)

    applied_transform = np.array(applied_transform_4x4 if applied_transform_4x4 is not None else default_applied_transform())
    points_ns = np.stack([apply_4x4(applied_transform, point) for point in points_local])

    plane_ns: Optional[Tuple[np.ndarray, float]] = None
    if len(labels) >= 3:
        plane_ns = plane_from_points(points_ns)

    markers_model: Optional[Dict[str, np.ndarray]] = None
    plane_model: Optional[Tuple[np.ndarray, float]] = None
    if dataparser_transform_4x4 is not None and scene_scale is not None:
        to_model = np.array(dataparser_transform_4x4, dtype=float)
        points_model = np.stack([apply_4x4(to_model, point) for point in points_ns]) * float(scene_scale)
        markers_model = {label: points_model[idx] for idx, label in enumerate(labels)}
        if len(labels) >= 3:
            plane_model = plane_from_points(points_model)

    markers_ns_world = {label: points_ns[idx] for idx, label in enumerate(labels)}
    return WaterMarkersResult(
        markers_ns_world=markers_ns_world,
        markers_model=markers_model,
        plane_ns_world=plane_ns,
        plane_model=plane_model,
        utm_to_local=similarity,
    )


# ---------------------------------------------------------------------------
# Metadata construction

def to_metadata_dict(result: WaterMarkersResult) -> Dict:
    """Serialize results for storage inside Dataparser metadata."""
    metadata: Dict[str, Dict] = {
        "water_markers": {
            "markers_ns_world": {label: value.tolist() for label, value in result.markers_ns_world.items()},
            "utm_to_local": {
                "scale": float(result.utm_to_local[0]),
                "rotation": result.utm_to_local[1].tolist(),
                "translation": result.utm_to_local[2].tolist(),
            },
        }
    }
    if result.markers_model is not None:
        metadata["water_markers"]["markers_model"] = {
            label: value.tolist() for label, value in result.markers_model.items()
        }
    if result.plane_ns_world is not None:
        normal, offset = result.plane_ns_world
        metadata["water_markers"]["plane_ns_world"] = {"normal": normal.tolist(), "d": float(offset)}
    if result.plane_model is not None:
        normal, offset = result.plane_model
        metadata["water_markers"]["plane_model"] = {"normal": normal.tolist(), "d": float(offset)}
    return metadata


# ---------------------------------------------------------------------------
# Dataparser wiring

def _to_homogeneous(transform_3x4: np.ndarray) -> np.ndarray:
    """Expand a 3x4 matrix to 4x4 homogeneous form."""
    assert transform_3x4.shape == (3, 4)
    transform_4x4 = np.eye(4, dtype=float)
    transform_4x4[:3, :] = transform_3x4
    return transform_4x4


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

    # ------------------------------------------------------------------
    # Internal helpers

    def _compute_water_surface_metadata(self, outputs) -> Optional[Dict]:
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

        transforms = self._load_transforms_json()
        applied_transform = transforms.get("applied_transform")
        if applied_transform is not None:
            applied_transform_np = np.array(applied_transform, dtype=float)
            if applied_transform_np.shape == (3, 4):
                applied_transform_np = _to_homogeneous(applied_transform_np)
        else:
            applied_transform_np = None

        dataparser_transform_np = outputs.dataparser_transform.detach().cpu().numpy()
        dataparser_transform_np = np.array(dataparser_transform_np, dtype=float)
        if dataparser_transform_np.shape == (3, 4):
            dataparser_transform_4x4 = _to_homogeneous(dataparser_transform_np)
        else:
            dataparser_transform_4x4 = dataparser_transform_np
        scene_scale = float(outputs.dataparser_scale)

        try:
            water_result = compute_water_markers(
                cameras_xml=cameras_xml,
                reference_xml=reference_xml,
                markers_xml=markers_xml,
                applied_transform_4x4=applied_transform_np,
                dataparser_transform_4x4=dataparser_transform_4x4,
                scene_scale=scene_scale,
            )
        except Exception as exc:  # pylint: disable=broad-except
            if self.config.require_water_plane:
                raise
            CONSOLE.log(f"[yellow]MetashapeWaterDataParser: Failed to compute water plane ({exc}).[/yellow]")
            return None

        if water_result.plane_model is None:
            if self.config.require_water_plane:
                raise ValueError("MetashapeWaterDataParser could not estimate a water plane from the provided markers.")
            CONSOLE.log("[yellow]MetashapeWaterDataParser: fewer than three markers; skipping water plane metadata.[/yellow]")
            return None

        return self._build_water_surface_dict(water_result)

    def _build_water_surface_dict(self, result: WaterMarkersResult) -> Dict:
        plane_model = result.plane_model
        assert plane_model is not None, "plane_model must be available when building metadata."

        metadata: Dict[str, Dict] = {
            "plane_model": {
                "normal": plane_model[0].tolist(),
                "d": float(plane_model[1]),
            },
            "source": "metashape_water_dataparser",
            "marker_count": len(result.markers_ns_world),
            "marker_labels": sorted(result.markers_ns_world.keys()),
        }

        if result.plane_ns_world is not None:
            metadata["plane_world"] = {
                "normal": result.plane_ns_world[0].tolist(),
                "d": float(result.plane_ns_world[1]),
            }

        if result.markers_ns_world:
            metadata["markers_ns_world"] = {
                label: value.tolist() for label, value in result.markers_ns_world.items()
            }
        if result.markers_model is not None:
            metadata["markers_model"] = {
                label: value.tolist() for label, value in result.markers_model.items()
            }

        if self.config.store_full_marker_metadata:
            metadata["markers"] = to_metadata_dict(result)["water_markers"]

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
