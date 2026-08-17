# Copyright (c) Meta Platforms, Inc. and affiliates.
"""GRAB dataset importer (SMPL-X body + grasped object).

Loads a GRAB sequence (``s*/<obj>_<action>_<n>.npz``) into the framework's
``ModelImporter`` contract:

  - the 55 SMPL-X joints (22 body + jaw/eyes + 30 finger joints) become bones,
    driven per-frame by our own SMPL-X forward kinematics over ``fullpose``
  - the subject's personalized body mesh (``vtemp``) is skinned to those joints
    via the model's LBS weights (top-4 influences per vertex)
  - the grasped object and the table are appended as extra rigid bones, each
    carrying its ``.ply`` mesh rigid-skinned 100% to that bone

GRAB's capture world is Z-up (head sits +Z above pelvis), so every animated
global transform is pre-multiplied by ``R_x(-90deg)`` to reach the framework's
Y-up convention. Skinning rest-space (v_template / object-local) is left in its
native frame; the basis change rides in via the per-frame bone matrices.
"""
import os
from functools import lru_cache
from typing import List, Optional

import numpy as np
from ai4animation.Animation.Motion import Hierarchy, Motion
from ai4animation.Import.ModelImporter import Mesh, ModelImporter, Skin
from ai4animation.Math import Tensor

# Z-up (GRAB world) -> Y-up (framework): rotate -90deg about X.
_ZUP_TO_YUP = np.eye(4, dtype=np.float64)
_ZUP_TO_YUP[:3, :3] = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


def _rodrigues(rotvecs: np.ndarray) -> np.ndarray:
    """Batched axis-angle (...,3) -> rotation matrices (...,3,3)."""
    theta = np.linalg.norm(rotvecs, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-8, 1.0, theta)
    k = rotvecs / safe
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack(
        [zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis=-1
    ).reshape(rotvecs.shape[:-1] + (3, 3))
    s = np.sin(theta)[..., np.newaxis]
    c = np.cos(theta)[..., np.newaxis]
    eye = np.eye(3)
    R = eye + s * K + (1.0 - c) * (K @ K)
    # theta ~ 0 -> identity
    small = (theta < 1e-8)[..., np.newaxis]
    return np.where(small, eye, R)


def _trs(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compose rotation (...,3,3) + translation (...,3) into (...,4,4)."""
    m = np.zeros(R.shape[:-2] + (4, 4), dtype=R.dtype)
    m[..., :3, :3] = R
    m[..., :3, 3] = t
    m[..., 3, 3] = 1.0
    return m


def _smooth1d(x: np.ndarray, n: int) -> np.ndarray:
    """Hann-windowed smoothing with edge padding (no edge attenuation)."""
    x = np.asarray(x, dtype=np.float64)
    if n <= 1 or x.shape[0] < 2:
        return x
    w = np.hanning(n)
    w = w / w.sum()
    pad = n // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, w, mode="same")[pad : pad + x.shape[0]]


def _load_ply(path: str):
    import trimesh

    tm = trimesh.load(path, force="mesh", process=False)
    return (
        np.asarray(tm.vertices, dtype=np.float64),
        np.asarray(tm.vertex_normals, dtype=np.float64),
        np.asarray(tm.faces, dtype=np.int64),
    )


class GRAB(ModelImporter):
    def __init__(
        self,
        seq_path: str,
        smplx_dir: str,
        grab_root: str,
        downsample_fps: Optional[float] = 30.0,
    ) -> None:
        import smplx
        from smplx.joint_names import JOINT_NAMES

        self._path = seq_path
        seq = np.load(seq_path, allow_pickle=True)
        gender = str(seq["gender"])
        src_fps = float(seq["framerate"])
        body = seq["body"].item()["params"]
        obj = seq["object"].item()
        table = seq["table"].item()

        # --- SMPL-X topology (shape-independent): weights, regressor, parents ---
        model = smplx.create(
            smplx_dir, model_type="smplx", gender=gender,
            use_pca=False, flat_hand_mean=True, batch_size=1,
        )
        J_regressor = model.J_regressor.detach().cpu().numpy()      # (55, V)
        lbs_weights = model.lbs_weights.detach().cpu().numpy()      # (V, 55)
        parents = model.parents.detach().cpu().numpy()[:55]         # (55,)
        faces = model.faces.astype(np.int64)
        njoints = 55

        # Subject-personalized template mesh (betas already baked in).
        vtemp_path = os.path.join(grab_root, seq["body"].item()["vtemp"])
        v_template = _load_ply(vtemp_path)[0]                        # (V, 3)
        J_rest = J_regressor @ v_template                           # (55, 3)

        # --- Frame selection / downsample ---
        n_frames = int(seq["n_frames"])
        if downsample_fps and downsample_fps < src_fps:
            step = max(int(round(src_fps / downsample_fps)), 1)
        else:
            step = 1
        idx = np.arange(0, n_frames, step)
        self._framerate = src_fps / step

        fullpose = np.asarray(body["fullpose"], dtype=np.float64)[idx]  # (F,165)
        transl = np.asarray(body["transl"], dtype=np.float64)[idx]      # (F,3)
        F = len(idx)

        # --- SMPL-X forward kinematics -> per-frame global 4x4 per joint ---
        rotvecs = fullpose.reshape(F, njoints, 3)
        R = _rodrigues(rotvecs)                                     # (F,55,3,3)
        rel = J_rest - J_rest[parents]
        rel[0] = J_rest[0]
        local = _trs(np.broadcast_to(R, (F, njoints, 3, 3)),
                     np.broadcast_to(rel, (F, njoints, 3)))         # (F,55,4,4)

        glob = np.zeros((F, njoints, 4, 4), dtype=np.float64)
        glob[:, 0] = local[:, 0]
        for j in range(1, njoints):
            glob[:, j] = glob[:, parents[j]] @ local[:, j]
        # fold root translation in (smplx adds transl after posing)
        T = np.broadcast_to(np.eye(4), (F, 4, 4)).copy()
        T[:, :3, 3] = transl
        glob = T[:, None] @ glob                                    # world (Z-up)

        # --- Object + table rigid transforms ---
        def _rigid(p):
            R_ = _rodrigues(np.asarray(p["global_orient"], dtype=np.float64)[idx])
            return _trs(R_, np.asarray(p["transl"], dtype=np.float64)[idx])

        obj_g = _rigid(obj["params"])                               # (F,4,4)
        table_g = _rigid(table["params"])

        # --- Assemble joints: 55 SMPL-X + object + table ---
        names = list(JOINT_NAMES[:njoints]) + ["object", "table"]
        parent_names = [None] + [JOINT_NAMES[parents[j]] for j in range(1, njoints)]
        parent_names += [None, None]                                # object, table = roots
        self._jointNames = names
        self._jointParents = parent_names

        # framework-frame per-frame globals for ALL bones
        world = np.concatenate([glob, obj_g[:, None], table_g[:, None]], axis=1)
        self._frames = (_ZUP_TO_YUP[None, None] @ world).astype(np.float32)  # (F,57,4,4)

        # rest pose (frame 0) for the Actor entity layout
        self._jointMatrices = self._frames[0].copy()

        # --- Skin bind matrices (native rest space, NOT converted) ---
        bind = np.zeros((njoints + 2, 4, 4), dtype=np.float64)
        bind[:njoints] = _trs(
            np.broadcast_to(np.eye(3), (njoints, 3, 3)), J_rest
        )
        bind[njoints] = np.eye(4)     # object: verts are object-local
        bind[njoints + 1] = np.eye(4) # table
        self._skin = Skin(
            joints=np.arange(njoints + 2, dtype=np.int64),
            bind_pose_matrices=bind.astype(np.float32),
        )

        # --- Meshes ---
        # GRAB_SKELETON=1 drops the SMPL-X body mesh (keep object + table) so a
        # skeleton-only contact view isn't occluded by the skinned body.
        self._meshes: List[Mesh] = [] if os.environ.get("GRAB_SKELETON") else [
            self._body_mesh(v_template, faces, lbs_weights)
        ]
        self.ObjectMeshPath = os.path.join(grab_root, obj["object_mesh"])
        self.TableMeshPath = os.path.join(grab_root, table["table_mesh"])
        self._meshes.append(self._rigid_mesh(self.ObjectMeshPath, njoints))
        self._meshes.append(self._rigid_mesh(self.TableMeshPath, njoints + 1))

        # --- Per-finger SURFACE-to-SURFACE contact ---
        # Joint/pad heuristics mis-locate contact (the distal joint sits ~3cm behind
        # the pad, etc.). Instead detect on the actual hand SURFACE: take each
        # finger's skinned mesh vertices, downsample to ~SAMPLES_PER_FINGER, pose
        # them per frame (LBS), and measure the true min distance from those hand
        # vertices to the object's surface mesh. A finger is in contact when its
        # closest vertex is within CONTACT_DIST of the object surface.
        from scipy.spatial import cKDTree

        SAMPLES_PER_FINGER = 20
        CONTACT_DIST = 0.005   # m: full contact when a finger vertex is within 5mm
        RAMP = 0.006           # m: soft zone above CONTACT_DIST where weight ramps 1->0
        obj_verts_local = _load_ply(self.ObjectMeshPath)[0]
        ov = obj_verts_local[::4]                              # object surface (downsampled)
        tree = cKDTree(ov)
        obj_world = self._frames[:, njoints]                  # (F,4,4) object pose
        self.Fingers = ["thumb", "index", "middle", "ring", "little"]

        # SMPL-X finger joint indices per hand (index1-3, middle1-3, pinky1-3,
        # ring1-3, thumb1-3); a vertex belongs to the finger of its dominant joint.
        FINGER_JOINTS = {
            "right": {"thumb": [52, 53, 54], "index": [40, 41, 42], "middle": [43, 44, 45],
                      "ring": [49, 50, 51], "little": [46, 47, 48]},
            "left":  {"thumb": [37, 38, 39], "index": [25, 26, 27], "middle": [28, 29, 30],
                      "ring": [34, 35, 36], "little": [31, 32, 33]},
        }
        dom = np.argmax(lbs_weights, axis=1)                  # (V,) dominant joint/vertex
        order = np.argsort(-lbs_weights, axis=1)[:, :4]       # top-4 LBS influences
        wtop = np.take_along_axis(lbs_weights, order, axis=1)
        wtop = wtop / np.clip(wtop.sum(axis=1, keepdims=True), 1e-8, None)
        R55 = self._frames[:, :njoints, :3, :3]               # (F,55,3,3) Y-up world
        t55 = self._frames[:, :njoints, :3, 3]                # (F,55,3)

        def _pose(vidx):
            """LBS-pose vertices ``vidx`` over all frames -> (F, n, 3) Y-up world."""
            v = v_template[vidx]
            oo, ww = order[vidx], wtop[vidx]
            out = np.zeros((F, len(vidx), 3))
            for k in range(4):
                jk = oo[:, k]
                vrel = v - J_rest[jk]                          # vertex in joint rest frame
                out += ww[:, k][None, :, None] * (
                    np.einsum("fnij,nj->fni", R55[:, jk], vrel) + t55[:, jk]
                )
            return out

        def _hand_fingers(hand):
            out = {}
            for fg, jts in FINGER_JOINTS[hand].items():
                vidx = np.where(np.isin(dom, jts))[0]
                if len(vidx) > SAMPLES_PER_FINGER:             # even downsample
                    vidx = vidx[np.linspace(0, len(vidx) - 1, SAMPLES_PER_FINGER).astype(int)]
                out[fg] = (vidx, _pose(vidx))
            return out

        def _finger_contact(posed):
            """Per-frame (min surface dist, contact pt obj-local, closest hand pt world)."""
            d = np.zeros(F); cp = np.zeros((F, 3)); hp = np.zeros((F, 3))
            for f in range(F):
                objR, objt = obj_world[f, :3, :3], obj_world[f, :3, 3]
                loc = (posed[f] - objt) @ objR                 # hand verts -> object frame
                dist, idx = tree.query(loc)
                j = int(np.argmin(dist))
                d[f] = dist[j]; cp[f] = ov[idx[j]]; hp[f] = posed[f, j]
            return d, cp, hp

        # choose grasping hand = more finger-contact frames (surface within CONTACT_DIST)
        hands = {h: _hand_fingers(h) for h in ("right", "left")}
        contact = {h: {fg: _finger_contact(p) for fg, (vi, p) in hands[h].items()}
                   for h in ("right", "left")}
        counts = {h: int(np.sum(np.any(
            [contact[h][fg][0] < CONTACT_DIST for fg in self.Fingers], axis=0)))
            for h in ("right", "left")}
        self.ContactHand = "right" if counts["right"] >= counts["left"] else "left"
        # SMPL-X distal (tip) joint index per finger of the grasping hand
        self.ContactTipIndex = {fg: FINGER_JOINTS[self.ContactHand][fg][-1]
                                for fg in self.Fingers}

        # per-finger soft weight + object-local contact point + world hand point +
        # the posed sample cloud (for visualization)
        self.FingerWeight = {}
        self.FingerContactLocal = {}   # contact point on object, object-local (smoothed)
        self.FingerHandWorld = {}      # closest hand vertex, world (smoothed)
        self.FingerSampleWorld = {}    # all sampled finger verts, world (per frame)
        for fg in self.Fingers:
            d, cp, hp = contact[self.ContactHand][fg]
            raw = np.clip((CONTACT_DIST + RAMP - d) / RAMP, 0.0, 1.0)
            has = raw > 0
            if has.any():                                      # hold contact pt off-contact
                o = np.where(has)[0]
                for f in range(F):
                    if not has[f]:
                        cp[f] = cp[o[np.argmin(np.abs(o - f))]]
            self.FingerWeight[fg] = _smooth1d(raw, 11)
            self.FingerContactLocal[fg] = np.stack(
                [_smooth1d(cp[:, k], 11) for k in range(3)], axis=1).astype(np.float32)
            self.FingerHandWorld[fg] = np.stack(
                [_smooth1d(hp[:, k], 7) for k in range(3)], axis=1).astype(np.float32)
            self.FingerSampleWorld[fg] = hands[self.ContactHand][fg][1].astype(np.float32)

        # backward-compat alias: the omomo two-stage grasp consumes FingerLocal as
        # an object-local contact target (it then snaps to the surface) -> point it
        # at the surface contact point computed above.
        self.FingerLocal = self.FingerContactLocal

        # --- skeleton/contact visualization metadata (used by viewer Draw) ---
        self.NumBodyJoints = njoints              # 55 SMPL-X joints precede obj/table
        self.ObjectIndex = njoints
        n2i = {n: i for i, n in enumerate(self._jointNames)}
        self._skelParent = np.array(
            [n2i.get(self._jointParents[i], -1) for i in range(njoints)], dtype=np.int64
        )

        print(
            f"[GRAB] {os.path.basename(seq_path)}: {F} frames @ {self._framerate:.0f}fps "
            f"({gender}), {njoints} SMPL-X joints + object + table, "
            f"{sum(m.VertexCount for m in self._meshes)} verts"
        )

    def _body_mesh(self, v_template, faces, lbs_weights) -> Mesh:
        import trimesh

        normals = trimesh.Trimesh(
            vertices=v_template, faces=faces, process=False
        ).vertex_normals
        # top-4 LBS influences per vertex
        order = np.argsort(-lbs_weights, axis=1)[:, :4]
        w = np.take_along_axis(lbs_weights, order, axis=1)
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-8, None)
        from PIL import Image

        skin_blue = Image.new("RGBA", (1, 1), color=(150, 185, 230, 255))  # soft light blue
        return Mesh(
            name="smplx_body",
            vertices=v_template.astype(np.float32),
            normals=np.asarray(normals, dtype=np.float32),
            triangles=faces.reshape(-1).astype(np.int64),
            skin_indices=order.astype(np.int64),
            skin_weights=w.astype(np.float32),
            image=skin_blue,
        )

    def _rigid_mesh(self, ply_path: str, bone_index: int) -> Mesh:
        verts, normals, faces = _load_ply(ply_path)
        n = verts.shape[0]
        skin_idx = np.zeros((n, 4), dtype=np.int64)
        skin_w = np.zeros((n, 4), dtype=np.float32)
        skin_idx[:, 0] = bone_index
        skin_w[:, 0] = 1.0
        return Mesh(
            name=os.path.splitext(os.path.basename(ply_path))[0],
            vertices=verts.astype(np.float32),
            normals=normals.astype(np.float32),
            triangles=faces.reshape(-1).astype(np.int64),
            skin_indices=skin_idx,
            skin_weights=skin_w,
        )

    # --- ModelImporter interface ---

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

    @classmethod
    @lru_cache(maxsize=2)
    def Create(cls, seq_path: str, smplx_dir: str, grab_root: str) -> "GRAB":
        return cls(seq_path, smplx_dir, grab_root)

    def LoadMotion(self, names=None, floor=None) -> Motion:
        if names is None:
            bone_names = self._jointNames
            parent_names = self._jointParents
            frames = self._frames
        else:
            n2i = {n: i for i, n in enumerate(self._jointNames)}
            indices = [n2i[n] for n in names if n in n2i]
            bone_names = [self._jointNames[i] for i in indices]
            parent_names = [self._jointParents[i] for i in indices]
            frames = self._frames[:, indices]
        return Motion(
            name=self.Filename,
            hierarchy=Hierarchy(bone_names, parent_names),
            frames=Tensor.Create(frames),
            framerate=self._framerate,
        )
