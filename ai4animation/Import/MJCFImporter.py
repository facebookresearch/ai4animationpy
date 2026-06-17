# Copyright (c) Meta Platforms, Inc. and affiliates.
"""MJCF importer.

Parses a MuJoCo MJCF (``<mujoco>`` / ``<worldbody>``) humanoid description into
the framework's ``ModelImporter`` contract, in the same spirit as the URDF
importer:

  - ``<body>`` elements become the bones (each carries a local ``pos``/``quat``
    offset from its parent).
  - primitive ``<geom>`` shapes (capsule / box / sphere) are tessellated into
    triangle meshes, baked into world rest space, and rigidly skinned 100% to
    their owning body's bone.

This targets the SMPL-X humanoid used by InterMimic / InterPrior
(``omomo.xml``): 52 bodies (the SMPL-X body + fully articulated fingers), no
external mesh files — every limb is a capsule, feet/torso are boxes, the
pelvis/head are spheres.

MuJoCo is right-handed, Z-up; the framework (glTF convention) is right-handed,
Y-up, so every global transform is pre-multiplied by ``R_x(-90deg)``
(``_MJCF_TO_FRAMEWORK``), exactly like the URDF importer.
"""
import os
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
from ai4animation.Animation.Motion import Hierarchy, Motion
from ai4animation.Import.ModelImporter import Mesh, ModelImporter, Skin
from ai4animation.Math import Tensor


# MuJoCo (right-handed, Z-up) -> framework (right-handed, Y-up): rotate -90deg about X.
_MJCF_TO_FRAMEWORK = np.eye(4, dtype=np.float64)
_MJCF_TO_FRAMEWORK[:3, :3] = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


def _floats(s: Optional[str], n: int, default=0.0) -> np.ndarray:
    if s is None:
        return np.full(n, default, dtype=np.float64)
    vals = [float(v) for v in s.split()]
    return np.array(vals, dtype=np.float64)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) -> 3x3 rotation."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ], dtype=np.float64)


def _local_matrix(pos: np.ndarray, quat: Optional[np.ndarray]) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = pos
    if quat is not None:
        m[:3, :3] = _quat_to_matrix(quat)
    return m


def _align_z_to(direction: np.ndarray) -> np.ndarray:
    """3x3 rotation mapping +Z onto the unit ``direction``."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    c = float(np.dot(z, d))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


class MJCF(ModelImporter):
    def __init__(self, path: str, cap_sections: int = 12) -> None:
        self._path = path
        self._cap_sections = cap_sections
        root = ET.parse(path).getroot()
        worldbody = root.find("worldbody")

        # --- Walk the nested <body> tree (DFS pre-order = root-first) ---
        self._names: List[str] = []
        self._parents: List[Optional[str]] = []
        self._local: List[np.ndarray] = []        # local offset from parent (4x4)
        self._geoms: List[list] = []              # per-body list of geom elements

        def _walk(body: ET.Element, parent: Optional[str]) -> None:
            name = body.get("name")
            self._names.append(name)
            self._parents.append(parent)
            self._local.append(
                _local_matrix(_floats(body.get("pos"), 3),
                              _floats(body.get("quat"), 4) if body.get("quat") else None)
            )
            self._geoms.append([g for g in body.findall("geom")])
            for child in body.findall("body"):
                _walk(child, name)

        for body in worldbody.findall("body"):
            _walk(body, None)

        self._jointNames = self._names
        self._jointParents = self._parents
        idx_of = {n: i for i, n in enumerate(self._names)}

        # --- FK rest pose (Z-up), then convert to framework Y-up ---
        globals_z: List[np.ndarray] = [None] * len(self._names)
        for i, name in enumerate(self._names):
            p = self._parents[i]
            globals_z[i] = self._local[i] if p is None else globals_z[idx_of[p]] @ self._local[i]
        self._globals_z = np.stack(globals_z)                       # (N,4,4) Z-up
        self._jointMatrices = np.stack(
            [_MJCF_TO_FRAMEWORK @ g for g in globals_z]
        ).astype(np.float32)

        # --- Tessellate geoms, bake into world rest space, rigid-skin ---
        self._meshes: List[Mesh] = []
        for i, name in enumerate(self._names):
            world = self._jointMatrices[i].astype(np.float64)
            for g in self._geoms[i]:
                tri = self._geom_mesh(g)
                if tri is None:
                    continue
                verts, faces = tri
                rot, t = world[:3, :3], world[:3, 3]
                verts = verts @ rot.T + t
                normals = self._vertex_normals(verts, faces)
                n = verts.shape[0]
                skin_idx = np.zeros((n, 4), dtype=np.int64)
                skin_w = np.zeros((n, 4), dtype=np.float32)
                skin_idx[:, 0] = i
                skin_w[:, 0] = 1.0
                self._meshes.append(
                    Mesh(
                        name=name,
                        vertices=verts.astype(np.float32),
                        normals=normals.astype(np.float32),
                        triangles=faces.reshape(-1).astype(np.int64),
                        skin_indices=skin_idx,
                        skin_weights=skin_w,
                        image=None,
                    )
                )

        joints = np.arange(len(self._names), dtype=np.int64)
        self._skin = Skin(joints=joints, bind_pose_matrices=self._jointMatrices)
        print(
            f"[MJCF] {len(self._names)} bodies, {len(self._meshes)} geom meshes "
            f"from {os.path.basename(path)}"
        )

    # --- geom tessellation (mesh-local space) -------------------------------

    def _geom_mesh(self, g: ET.Element):
        import trimesh

        gtype = g.get("type", "sphere")
        try:
            if gtype == "capsule":
                ft = g.get("fromto")
                size = _floats(g.get("size"), 1)
                r = float(size[0])
                if ft is not None:
                    p = _floats(ft, 6)
                    p0, p1 = p[:3], p[3:]
                else:                       # pos + (radius, halflength)
                    pos = _floats(g.get("pos"), 3)
                    hl = float(size[1]) if size.size > 1 else 0.0
                    p0, p1 = pos - np.array([0, 0, hl]), pos + np.array([0, 0, hl])
                seg = p1 - p0
                L = float(np.linalg.norm(seg))
                tm = trimesh.creation.capsule(height=L, radius=r, count=[self._cap_sections, self._cap_sections])
                # trimesh capsule spans z in [0, L] (caps beyond); align +Z to seg, base at p0.
                M = np.eye(4)
                M[:3, :3] = _align_z_to(seg)
                M[:3, 3] = p0
                tm.apply_transform(M)
            elif gtype == "box":
                pos = _floats(g.get("pos"), 3)
                half = _floats(g.get("size"), 3)
                tm = trimesh.creation.box(extents=2.0 * half)
                M = _local_matrix(pos, _floats(g.get("quat"), 4) if g.get("quat") else None)
                tm.apply_transform(M)
            elif gtype == "sphere":
                pos = _floats(g.get("pos"), 3)
                r = float(_floats(g.get("size"), 1)[0])
                tm = trimesh.creation.icosphere(subdivisions=2, radius=r)
                tm.apply_translation(pos)
            else:
                return None
        except Exception as e:  # noqa: BLE001
            print(f"[MJCF] geom '{gtype}' failed: {e}")
            return None
        return np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int64)

    @staticmethod
    def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
        normals = np.zeros_like(verts)
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for k in range(3):
            np.add.at(normals, faces[:, k], fn)
        nrm = np.linalg.norm(normals, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return normals / nrm

    # --- ModelImporter interface --------------------------------------------

    @property
    def Filename(self) -> str:
        return os.path.splitext(os.path.basename(self._path))[0]

    @property
    def JointNames(self) -> List[str]:
        return self._jointNames

    @property
    def JointParents(self) -> List[str]:
        return self._jointParents

    @property
    def JointMatrices(self) -> np.ndarray:
        return self._jointMatrices

    @property
    def Meshes(self) -> List[Mesh]:
        return self._meshes

    @property
    def Skin(self) -> Optional[Skin]:
        return self._skin

    @property
    def LocalOffsets(self) -> np.ndarray:
        """(N,3) per-body local translation offset from parent, in Z-up."""
        return np.stack([m[:3, 3] for m in self._local])

    @classmethod
    @lru_cache(maxsize=2)
    def Create(cls, path: str) -> "MJCF":
        return cls(path)

    def LoadMotion(self, names=None, floor=None) -> Motion:
        """Static rest-pose Motion (2 identical frames) for standalone viewing."""
        if names is None:
            bone_names, parent_names, frames = self._jointNames, self._jointParents, self._jointMatrices
        else:
            idx_of = {n: i for i, n in enumerate(self._jointNames)}
            indices = [idx_of[n] for n in names if n in idx_of]
            bone_names = [self._jointNames[i] for i in indices]
            parent_names = [self._jointParents[i] for i in indices]
            frames = self._jointMatrices[indices]
        frames = np.broadcast_to(frames, (2,) + frames.shape).copy()
        return Motion(
            name=self.Filename,
            hierarchy=Hierarchy(bone_names, parent_names),
            frames=Tensor.Create(frames),
            framerate=30.0,
        )
