# nerfstudio/data/utils/water_markers.py
# -*- coding: utf-8 -*-
"""
Water markers → Nerfstudio space

Lieste Kameras und Marker aus Metashape-XML, fitted UTM→Local (Umeyama),
transformiert Marker wie Kameras durch applied_transform und dataparser_transform+scale
und berechnet eine Wasser-Ebene aus >=3 Punkten.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import xml.etree.ElementTree as ET

from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    Nerfstudio as NerfstudioDataParser,
    NerfstudioDataParserConfig,
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


# ---------- Mathe/Geometrie ----------

def umeyama(A: np.ndarray, B: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """Ähnlichkeitstransformation B ≈ s * R @ A + t (N×3)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    assert A.shape == B.shape and A.ndim == 2 and A.shape[1] == 3
    n = A.shape[0]
    muA, muB = A.mean(0), B.mean(0)
    X, Y = A - muA, B - muB
    U, S, Vt = np.linalg.svd((Y.T @ X) / n)
    D = np.eye(3);
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    varA = (X**2).sum() / n
    s = (np.trace(np.diag(S) @ D) / varA) if with_scale else 1.0
    t = muB - s * (R @ muA)
    return float(s), R, t


def rms_of_fit(s: float, R: np.ndarray, t: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    pred = (s * (R @ A.T)).T + t
    return float(np.sqrt(((pred - B)**2).sum(axis=1).mean()))


def plane_from_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least-squares Ebene durch >=3 Punkte. Liefert (n, d) mit ||n||=1, n·x + d = 0."""
    assert points.shape[0] >= 3 and points.shape[1] == 3
    c = points.mean(0)
    Q = points - c
    # letzte SV ist Normale
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    n = Vt[-1]
    n = n / np.linalg.norm(n)
    d = -float(n @ c)
    return n, d


# ---------- Metashape Parsing ----------

@dataclass
class CameraRow:
    label: str
    M: np.ndarray          # 4x4
    ref_utm: Optional[np.ndarray]  # (3,) or None


def _parse_cameras_with_refs(cameras_xml: Path) -> List[CameraRow]:
    root = ET.parse(str(cameras_xml)).getroot()
    rows: List[CameraRow] = []
    for cam in root.findall(".//camera"):
        label = cam.get("label", "")
        tf = cam.findtext("transform")
        if tf is None:
            continue
        M = np.fromstring(tf, sep=" ").reshape(4, 4)
        ref_xyz = None
        ref = cam.find("reference")
        if ref is not None and all(k in ref.attrib for k in ("x", "y", "z")):
            ref_xyz = np.array([float(ref.get("x")), float(ref.get("y")), float(ref.get("z"))], dtype=float)
        rows.append(CameraRow(label=label, M=M, ref_utm=ref_xyz))
    return rows


def _parse_markers_from_reference_xml(reference_xml: Path) -> Dict[str, np.ndarray]:
    """Liest <markers><marker label=""><reference x y z/></marker></markers>."""
    root = ET.parse(str(reference_xml)).getroot()
    out: Dict[str, np.ndarray] = {}
    for m in root.findall(".//markers/marker"):
        label = m.get("label", "")
        ref = m.find("reference")
        if label and ref is not None and all(k in ref.attrib for k in ("x", "y", "z")):
            out[label] = np.array([float(ref.get("x")), float(ref.get("y")), float(ref.get("z"))], dtype=float)
    return out


def _parse_markers_from_markers_xml(markers_xml: Path) -> Dict[str, np.ndarray]:
    """Liest <markers><marker label=""><reference x y z/></marker></markers> (alternative Datei)."""
    root = ET.parse(str(markers_xml)).getroot()
    out: Dict[str, np.ndarray] = {}
    for m in root.findall(".//marker"):
        label = m.get("label", "")
        ref = m.find("reference")
        if label and ref is not None and all(k in ref.attrib for k in ("x", "y", "z")):
            out[label] = np.array([float(ref.get("x")), float(ref.get("y")), float(ref.get("z"))], dtype=float)
    return out


# ---------- Transformations ----------

def camera_center_from_c2w(M: np.ndarray) -> np.ndarray:
    """Bei deinen Metashape-Exports ist die 4x4 eine c2w (Übersetzung = Kamerazentrum)."""
    return M[:3, 3]


def camera_center_from_w2c(M: np.ndarray) -> np.ndarray:
    """Falls doch w2c: C = -R^T t."""
    R, t = M[:3, :3], M[:3, 3]
    return -R.T @ t


def default_applied_transform() -> np.ndarray:
    """Standard-Achstausch Metashape→Nerfstudio-World: (z, x, y)."""
    P = np.array([[0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=float)
    return P


def apply_4x4(M: np.ndarray, p: np.ndarray) -> np.ndarray:
    ph = np.concatenate([p, [1.0]])
    return (M @ ph)[:3]


def apply_similarity(s: float, R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts)
    return (s * (R @ pts.T)).T + t


# ---------- Haupt-API ----------

@dataclass
class WaterMarkersResult:
    # Punkte in verschiedenen Räumen
    markers_local: Dict[str, np.ndarray]
    markers_ns_world: Dict[str, np.ndarray]
    markers_model: Optional[Dict[str, np.ndarray]]
    # Ebene (Optional je Raum, falls >=3 Punkte)
    plane_ns_world: Optional[Tuple[np.ndarray, float]]
    plane_model: Optional[Tuple[np.ndarray, float]]
    # Debug
    used_c2w: bool
    utm_to_local: Tuple[float, np.ndarray, np.ndarray]  # (s,R,t)
    fit_method: str
    chunk_transform: Optional[Tuple[float, np.ndarray, np.ndarray]]


def compute_water_markers(
    cameras_xml: Path,
    reference_xml: Path,
    markers_xml: Optional[Path] = None,
    applied_transform_4x4: Optional[np.ndarray] = None,
    dataparser_transform_4x4: Optional[np.ndarray] = None,
    scene_scale: Optional[float] = None,
) -> WaterMarkersResult:
    """
    Liest Kameras+Referenzen und Marker (UTM) und liefert transformierte Marker.
    """
    chunk_transform: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
    candidate_paths = []
    for candidate in (cameras_xml, reference_xml, markers_xml):
        if candidate is None:
            continue
        candidate_paths.append(Path(candidate))
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        try:
            root_candidate = ET.parse(str(candidate)).getroot()
        except ET.ParseError:
            continue
        chunk_node = root_candidate.find(".//chunk/transform")
        if chunk_node is None:
            continue
        rot_text = chunk_node.findtext("rotation")
        trans_text = chunk_node.findtext("translation")
        scale_text = chunk_node.findtext("scale")
        if rot_text is None or trans_text is None or scale_text is None:
            continue
        try:
            R_chunk = np.fromstring(rot_text, sep=" ").reshape(3, 3)
            t_chunk = np.fromstring(trans_text, sep=" ")
            s_chunk = float(scale_text)
        except ValueError:
            continue
        chunk_transform = (s_chunk, R_chunk, t_chunk)
        break

    # 1) Kameras + UTM-Referenzen
    cams = _parse_cameras_with_refs(Path(cameras_xml))
    utm, local_c2w, local_w2c = [], [], []
    for c in cams:
        if c.ref_utm is None:
            continue
        utm.append(c.ref_utm)
        local_c2w.append(camera_center_from_c2w(c.M))
        local_w2c.append(camera_center_from_w2c(c.M))
    utm = np.stack(utm)
    local_c2w = np.stack(local_c2w)
    local_w2c = np.stack(local_w2c)

    # 2) Fit: wähle das bessere Modell (c2w vs. w2c) per RMS
    s1, R1, t1 = umeyama(utm, local_c2w, with_scale=True)
    s2, R2, t2 = umeyama(utm, local_w2c, with_scale=True)
    rms1 = rms_of_fit(s1, R1, t1, utm, local_c2w)
    rms2 = rms_of_fit(s2, R2, t2, utm, local_w2c)
    use_c2w = rms1 <= rms2
    s, R, t = (s1, R1, t1) if use_c2w else (s2, R2, t2)

    # 3) Marker-UTM einlesen
    markers_ref = _parse_markers_from_reference_xml(Path(reference_xml))
    if markers_xml is not None and Path(markers_xml).exists():
        # ergänze/override aus separater Datei
        markers_ref.update(_parse_markers_from_markers_xml(Path(markers_xml)))

    if len(markers_ref) == 0:
        raise ValueError("Keine Marker im REFERENCE/MARKERS-XML gefunden.")

    # 4) UTM → Local
    labels = sorted(markers_ref.keys())
    P_utm = np.stack([markers_ref[k] for k in labels])
    fit_method = "chunk_transform" if chunk_transform is not None else "umeyama"
    if chunk_transform is not None:
        s_chunk, R_chunk, t_chunk = chunk_transform
        s = 1.0 / float(s_chunk)
        R = R_chunk.T
        t = -(R @ t_chunk) / float(s_chunk)
        use_c2w = True
    P_local = apply_similarity(s, R, t, P_utm)

    # 5) Local → Nerfstudio-World (applied_transform)
    A = applied_transform_4x4 if applied_transform_4x4 is not None else default_applied_transform()
    P_ns = np.stack([apply_4x4(A, p) for p in P_local])

    # 6) NS-World → Model (dataparser_transform + scene_scale)
    markers_model = None
    if dataparser_transform_4x4 is not None and scene_scale is not None:
        P_mod = np.stack([apply_4x4(dataparser_transform_4x4, p) for p in P_ns]) * float(scene_scale)
        markers_model = {lbl: P_mod[i] for i, lbl in enumerate(labels)}

    # 7) Ebenen (falls >=3 Punkte)
    plane_ns = plane_mod = None
    if len(labels) >= 3:
        n_ns, d_ns = plane_from_points(P_ns)
        plane_ns = (n_ns, float(d_ns))
        if markers_model is not None:
            Pm = np.stack([markers_model[k] for k in labels])
            n_m, d_m = plane_from_points(Pm)
            plane_mod = (n_m, float(d_m))

    return WaterMarkersResult(
        markers_local={lbl: P_local[i] for i, lbl in enumerate(labels)},
        markers_ns_world={lbl: P_ns[i] for i, lbl in enumerate(labels)},
        markers_model=markers_model,
        plane_ns_world=plane_ns,
        plane_model=plane_mod,
        used_c2w=use_c2w,
        utm_to_local=(s, R, t),
        fit_method=fit_method,
        chunk_transform=chunk_transform,
    )


# ---------- Helper für DataparserOutputs.metadata ----------

def to_metadata_dict(res: WaterMarkersResult) -> Dict:
    md = {
        "water_markers": {
            "markers_local": {k: v.tolist() for k, v in res.markers_local.items()},
            "markers_ns_world": {k: v.tolist() for k, v in res.markers_ns_world.items()},
            "used_c2w": bool(res.used_c2w),
            "utm_to_local": {
                "scale": float(res.utm_to_local[0]),
                "R": res.utm_to_local[1].tolist(),
                "t": res.utm_to_local[2].tolist(),
            },
            "fit_method": res.fit_method,
        }
    }
    if res.markers_model is not None:
        md["water_markers"]["markers_model"] = {k: v.tolist() for k, v in res.markers_model.items()}
    if res.plane_ns_world is not None:
        n, d = res.plane_ns_world
        md["water_markers"]["plane_ns_world"] = {"n": n.tolist(), "d": float(d)}
    if res.plane_model is not None:
        n, d = res.plane_model
        md["water_markers"]["plane_model"] = {"n": n.tolist(), "d": float(d)}
    if res.chunk_transform is not None:
        scale_chunk, R_chunk, t_chunk = res.chunk_transform
        md["water_markers"]["chunk_transform"] = {
            "scale": float(scale_chunk),
            "rotation": R_chunk.tolist(),
            "translation": t_chunk.tolist(),
        }
    return md


# ---------- Dataparser-Integration ----------


def _to_homogeneous(transform_3x4: np.ndarray) -> np.ndarray:
    """Extend 3x4 array to 4x4 homogeneous form."""
    assert transform_3x4.shape == (3, 4)
    transform_4x4 = np.eye(4, dtype=float)
    transform_4x4[:3, :] = transform_3x4
    return transform_4x4


@dataclass
class MetashapeWaterDataParserConfig(NerfstudioDataParserConfig):
    """Config for Metashape water-aware dataparser."""

    _target: Type = field(default_factory=lambda: MetashapeWaterDataParser)
    metashape_xml: Optional[Path] = None
    """Single Metashape export (e.g. MARKERS_*.xml) containing both cameras and markers."""
    cameras_xml: Optional[Path] = None
    """Path to Agisoft Metashape cameras.xml containing camera transforms and references."""
    reference_xml: Optional[Path] = None
    """Path to reference.xml (or equivalent) with water marker world coordinates."""
    markers_xml: Optional[Path] = None
    """Optional additional markers XML to merge (overrides entries from reference.xml)."""
    require_water_plane: bool = True
    """If True, raises when water plane cannot be computed."""
    store_full_marker_metadata: bool = False
    """Attach full marker coordinates to metadata for debugging."""


class MetashapeWaterDataParser(NerfstudioDataParser):
    """Dataparser that augments Nerfstudio data with water-plane metadata."""

    config: MetashapeWaterDataParserConfig

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        outputs = super()._generate_dataparser_outputs(split=split, **kwargs)
        water_metadata = self._compute_water_surface_metadata(outputs)
        if water_metadata is None:
            return outputs

        metadata = dict(outputs.metadata)
        metadata["water_surface"] = water_metadata
        return replace(outputs, metadata=metadata)

    # ----- internal helpers -----

    def _compute_water_surface_metadata(self, outputs) -> Optional[Dict]:
        cameras_xml: Optional[Path]
        reference_xml: Optional[Path]
        markers_xml: Optional[Path]

        combined_xml = self._resolve_path(self.config.metashape_xml)
        if combined_xml is not None:
            cameras_xml = combined_xml
            reference_xml = combined_xml
            markers_xml = self._resolve_path(self.config.markers_xml)
        else:
            cameras_xml = self._resolve_path(self.config.cameras_xml)
            reference_xml = self._resolve_path(self.config.reference_xml)
            markers_xml = self._resolve_path(self.config.markers_xml)

        if cameras_xml is None or reference_xml is None:
            if self.config.require_water_plane:
                raise ValueError(
                    "MetashapeWaterDataParser requires `metashape_xml` or both `cameras_xml` and `reference_xml` to compute the water plane."
                )
            CONSOLE.log("[yellow]MetashapeWaterDataParser: Missing XML inputs, skipping water surface computation.[/yellow]")
            return None

        if not cameras_xml.exists() or not reference_xml.exists():
            missing = [str(p) for p in (cameras_xml, reference_xml) if not p.exists()]
            raise FileNotFoundError(f"MetashapeWaterDataParser: Missing required XML file(s): {missing}")

        transforms = self._load_transforms_json()
        applied_transform_list = transforms.get("applied_transform")
        applied_transform = None
        if applied_transform_list is not None:
            applied_transform = np.array(applied_transform_list, dtype=float)
            if applied_transform.shape == (3, 4):
                applied_transform = _to_homogeneous(applied_transform)
        dataparser_transform = outputs.dataparser_transform.detach().cpu().numpy()
        dataparser_transform_4x4 = _to_homogeneous(dataparser_transform)
        dataparser_scale = float(outputs.dataparser_scale)

        try:
            water_result = compute_water_markers(
                cameras_xml=cameras_xml,
                reference_xml=reference_xml,
                markers_xml=markers_xml if markers_xml is not None and markers_xml.exists() else None,
                applied_transform_4x4=applied_transform,
                dataparser_transform_4x4=dataparser_transform_4x4,
                scene_scale=dataparser_scale,
            )
        except Exception as exc:  # pylint: disable=broad-except
            if self.config.require_water_plane:
                raise
            CONSOLE.log(f"[yellow]MetashapeWaterDataParser: Failed to compute water plane ({exc}).[/yellow]")
            return None

        if water_result.plane_model is None:
            if self.config.require_water_plane:
                raise ValueError("MetashapeWaterDataParser could not estimate a water plane from the provided markers.")
            CONSOLE.log("[yellow]MetashapeWaterDataParser: Marker count < 3, skipping water plane metadata.[/yellow]")
            return None

        return self._build_water_surface_dict(water_result)

    def _build_water_surface_dict(self, result: WaterMarkersResult) -> Dict:
        plane_model = result.plane_model
        assert plane_model is not None, "plane_model must be available."

        meta: Dict[str, Dict] = {
            "plane_model": {
                "normal": plane_model[0].tolist(),
                "d": float(plane_model[1]),
            },
            "source": "metashape_water_dataparser",
            "marker_count": len(result.markers_ns_world),
            "marker_labels": sorted(result.markers_ns_world.keys()),
            "used_c2w": bool(result.used_c2w),
            "fit_method": result.fit_method,
        }

        if result.plane_ns_world is not None:
            meta["plane_world"] = {
                "normal": result.plane_ns_world[0].tolist(),
                "d": float(result.plane_ns_world[1]),
            }

        if result.chunk_transform is not None:
            scale_chunk, R_chunk, t_chunk = result.chunk_transform
            meta["chunk_transform"] = {
                "scale": float(scale_chunk),
                "rotation": R_chunk.tolist(),
                "translation": t_chunk.tolist(),
            }

        if self.config.store_full_marker_metadata:
            marker_meta = to_metadata_dict(result)
            meta["markers"] = marker_meta["water_markers"]

        return meta

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
