# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Export skeletal animations as glTF 2.0 binary (.glb) files.

Inputs are expressed in WORLD space (positions and quaternion rotations per
joint per frame). The exporter converts them into LOCAL TRS values per node,
which is what glTF animation channels require.
"""

from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from pygltflib import (
    Accessor,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Asset,
    Attributes,
    Buffer,
    BufferView,
    GLTF2,
    Mesh,
    Node,
    Primitive,
    Scene,
)


# glTF component type constants.
_COMPONENT_FLOAT = 5126
_COMPONENT_UNSIGNED_SHORT = 5123
# glTF bufferView target hints.
_TARGET_ARRAY_BUFFER = 34962
_TARGET_ELEMENT_ARRAY_BUFFER = 34963


def _quat_normalize(q: NDArray) -> NDArray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return q / n


def _quat_conjugate(q: NDArray) -> NDArray:
    """Conjugate of a unit quaternion (x, y, z, w)."""
    out = q.copy()
    out[..., :3] = -out[..., :3]
    return out


def _quat_multiply(a: NDArray, b: NDArray) -> NDArray:
    """Hamilton product of two quaternions in (x, y, z, w) order."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty_like(a)
    out[..., 0] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 1] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 2] = aw * bz + ax * by - ay * bx + az * bw
    out[..., 3] = aw * bw - ax * bx - ay * by - az * bz
    return out


def _quat_rotate(q: NDArray, v: NDArray) -> NDArray:
    """Rotate vector v by quaternion q (xyzw)."""
    qxyz = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * np.cross(qxyz, v)
    return v + qw * t + np.cross(qxyz, t)


def _world_to_local(
    world_positions: NDArray,
    world_rotations: NDArray,
    parent_indices: Sequence[int],
) -> tuple[NDArray, NDArray]:
    """Convert world TR per joint to local TR per joint.

    Args:
        world_positions: (F, J, 3) float
        world_rotations: (F, J, 4) float quaternions in (x, y, z, w)
        parent_indices: length J, parent index per joint, -1 for root

    Returns:
        local_positions (F, J, 3), local_rotations (F, J, 4)
    """
    world_rotations = _quat_normalize(world_rotations.astype(np.float32))
    world_positions = world_positions.astype(np.float32)

    local_positions = np.zeros_like(world_positions)
    local_rotations = np.zeros_like(world_rotations)

    for j, parent in enumerate(parent_indices):
        if parent < 0:
            local_positions[:, j] = world_positions[:, j]
            local_rotations[:, j] = world_rotations[:, j]
        else:
            inv_parent_q = _quat_conjugate(world_rotations[:, parent])
            delta = world_positions[:, j] - world_positions[:, parent]
            local_positions[:, j] = _quat_rotate(inv_parent_q, delta)
            local_rotations[:, j] = _quat_normalize(
                _quat_multiply(inv_parent_q, world_rotations[:, j])
            )

    return local_positions, local_rotations


def _pad4(buf: bytearray) -> None:
    """Pad bytearray to 4-byte alignment with zeros."""
    while len(buf) % 4 != 0:
        buf.append(0)


class _BufferBuilder:
    """Accumulates raw bytes and emits BufferViews / Accessors."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.buffer_views: List[BufferView] = []
        self.accessors: List[Accessor] = []

    def _add_view(self, byte_length: int, target: Optional[int] = None) -> int:
        offset = len(self.data) - byte_length
        view = BufferView(
            buffer=0, byteOffset=offset, byteLength=byte_length, target=target
        )
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def add_scalar_float(self, values: NDArray) -> int:
        arr = np.ascontiguousarray(values.astype(np.float32).reshape(-1))
        raw = arr.tobytes()
        self.data.extend(raw)
        view_idx = self._add_view(len(raw))
        _pad4(self.data)
        accessor = Accessor(
            bufferView=view_idx,
            byteOffset=0,
            componentType=_COMPONENT_FLOAT,
            count=int(arr.size),
            type="SCALAR",
            min=[float(arr.min())] if arr.size > 0 else None,
            max=[float(arr.max())] if arr.size > 0 else None,
        )
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_vec_float(
        self,
        values: NDArray,
        components: int,
        target: Optional[int] = None,
        with_minmax: bool = False,
    ) -> int:
        assert values.shape[-1] == components
        arr = np.ascontiguousarray(values.astype(np.float32).reshape(-1, components))
        raw = arr.tobytes()
        self.data.extend(raw)
        view_idx = self._add_view(len(raw), target=target)
        _pad4(self.data)
        type_name = {2: "VEC2", 3: "VEC3", 4: "VEC4"}[components]
        accessor = Accessor(
            bufferView=view_idx,
            byteOffset=0,
            componentType=_COMPONENT_FLOAT,
            count=int(arr.shape[0]),
            type=type_name,
            min=arr.min(axis=0).astype(float).tolist() if with_minmax else None,
            max=arr.max(axis=0).astype(float).tolist() if with_minmax else None,
        )
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_indices_uint16(self, indices: NDArray) -> int:
        arr = np.ascontiguousarray(indices.astype(np.uint16).reshape(-1))
        raw = arr.tobytes()
        self.data.extend(raw)
        view_idx = self._add_view(len(raw), target=_TARGET_ELEMENT_ARRAY_BUFFER)
        _pad4(self.data)
        accessor = Accessor(
            bufferView=view_idx,
            byteOffset=0,
            componentType=_COMPONENT_UNSIGNED_SHORT,
            count=int(arr.size),
            type="SCALAR",
        )
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def _cube_geometry(size: float) -> tuple[NDArray, NDArray, NDArray]:
    """Return (positions, normals, indices) for a flat-shaded cube of edge `size` centered at origin."""
    s = size * 0.5
    # 6 faces, each with its own 4 vertices for flat-shaded normals.
    face_data = [
        # (normal, four corner positions ccw when viewed from outside)
        ((0, 0, 1), [(-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]),  # +Z
        ((0, 0, -1), [(s, -s, -s), (-s, -s, -s), (-s, s, -s), (s, s, -s)]),  # -Z
        ((1, 0, 0), [(s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s)]),  # +X
        ((-1, 0, 0), [(-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s)]),  # -X
        ((0, 1, 0), [(-s, s, s), (s, s, s), (s, s, -s), (-s, s, -s)]),  # +Y
        ((0, -1, 0), [(-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s)]),  # -Y
    ]
    positions: List[List[float]] = []
    normals: List[List[float]] = []
    indices: List[int] = []
    for normal, corners in face_data:
        base = len(positions)
        for corner in corners:
            positions.append(list(corner))
            normals.append(list(normal))
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(indices, dtype=np.uint16),
    )


def _validate_export_inputs(
    positions: NDArray,
    rotations: NDArray,
    bone_names: Sequence[str],
    parent_indices: Sequence[int],
    fps: float,
) -> None:
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"positions must have shape (F, J, 3), got {positions.shape}")
    if rotations.ndim != 3 or rotations.shape[-1] != 4:
        raise ValueError(f"rotations must have shape (F, J, 4), got {rotations.shape}")
    if positions.shape[:2] != rotations.shape[:2]:
        raise ValueError(
            f"positions {positions.shape} and rotations {rotations.shape} "
            "must share the (F, J) prefix"
        )
    num_joints = positions.shape[1]
    if len(parent_indices) != num_joints or len(bone_names) != num_joints:
        raise ValueError(
            f"bone_names ({len(bone_names)}) and parent_indices "
            f"({len(parent_indices)}) must match J={num_joints}"
        )
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")


def _build_nodes(
    bone_names: Sequence[str],
    parents: Sequence[int],
    local_positions: NDArray,
    local_rotations: NDArray,
) -> tuple[List[Node], List[int]]:
    num_joints = len(parents)
    children_of: List[List[int]] = [[] for _ in range(num_joints)]
    roots: List[int] = []
    for j, p in enumerate(parents):
        if p < 0:
            roots.append(j)
        else:
            children_of[p].append(j)
    nodes: List[Node] = []
    for j in range(num_joints):
        nodes.append(
            Node(
                name=str(bone_names[j]),
                translation=local_positions[0, j].astype(np.float32).tolist(),
                rotation=local_rotations[0, j].astype(np.float32).tolist(),
                children=children_of[j] if children_of[j] else [],
            )
        )
    return nodes, roots


def _build_joint_box_mesh(builder: "_BufferBuilder", box_size: float) -> Mesh:
    cube_positions, cube_normals, cube_indices = _cube_geometry(box_size)
    pos_acc = builder.add_vec_float(
        cube_positions, 3, target=_TARGET_ARRAY_BUFFER, with_minmax=True
    )
    nrm_acc = builder.add_vec_float(cube_normals, 3, target=_TARGET_ARRAY_BUFFER)
    idx_acc = builder.add_indices_uint16(cube_indices)
    return Mesh(
        name="JointBox",
        primitives=[
            Primitive(
                attributes=Attributes(POSITION=pos_acc, NORMAL=nrm_acc),
                indices=idx_acc,
                mode=4,  # TRIANGLES
            )
        ],
    )


def _build_animation(
    builder: "_BufferBuilder",
    local_positions: NDArray,
    local_rotations: NDArray,
    fps: float,
    name: str,
) -> Animation:
    num_frames, num_joints = local_positions.shape[:2]
    timestamps = np.arange(num_frames, dtype=np.float32) / float(fps)
    time_accessor = builder.add_scalar_float(timestamps)

    samplers: List[AnimationSampler] = []
    channels: List[AnimationChannel] = []
    for j in range(num_joints):
        t_acc = builder.add_vec_float(local_positions[:, j], 3)
        r_acc = builder.add_vec_float(local_rotations[:, j], 4)
        t_sampler_idx = len(samplers)
        samplers.append(
            AnimationSampler(input=time_accessor, output=t_acc, interpolation="LINEAR")
        )
        channels.append(
            AnimationChannel(
                sampler=t_sampler_idx,
                target=AnimationChannelTarget(node=j, path="translation"),
            )
        )
        r_sampler_idx = len(samplers)
        samplers.append(
            AnimationSampler(input=time_accessor, output=r_acc, interpolation="LINEAR")
        )
        channels.append(
            AnimationChannel(
                sampler=r_sampler_idx,
                target=AnimationChannelTarget(node=j, path="rotation"),
            )
        )
    return Animation(name=name, samplers=samplers, channels=channels)


class GLBExporter:
    """Writes a skeletal animation to a binary glTF (.glb) file."""

    @staticmethod
    def Export(
        positions: NDArray,
        rotations: NDArray,
        bone_names: Sequence[str],
        parent_indices: Sequence[int],
        out_path: str,
        fps: float,
        animation_name: Optional[str] = None,
        joint_boxes: bool = True,
        box_size: float = 0.05,
    ) -> GLTF2:
        """Export world-space joint animation to a .glb file.

        Args:
            positions: (F, J, 3) float, world joint positions per frame.
            rotations: (F, J, 4) float, world joint quaternions (x, y, z, w)
                per frame.
            bone_names: length J sequence of joint names.
            parent_indices: length J sequence of parent indices; -1 for roots.
            out_path: destination path. ".glb" is appended if missing.
            fps: animation frame rate in Hz.
            animation_name: optional name for the embedded animation.
            joint_boxes: if True, attach a small cube mesh to every joint so
                the skeleton is visible in standard glTF viewers.
            box_size: edge length (in scene units) of the per-joint cubes.

        Returns:
            The constructed GLTF2 object (after saving).
        """
        positions = np.asarray(positions)
        rotations = np.asarray(rotations)
        _validate_export_inputs(positions, rotations, bone_names, parent_indices, fps)

        parents = list(parent_indices)

        if not out_path.lower().endswith(".glb"):
            out_path = out_path + ".glb"

        # Convert to local TRS (translation + rotation only, no scale).
        local_positions, local_rotations = _world_to_local(
            positions, rotations, parents
        )

        nodes, roots = _build_nodes(
            bone_names, parents, local_positions, local_rotations
        )

        builder = _BufferBuilder()

        # Optional: build one shared cube mesh referenced by every joint node.
        meshes: List[Mesh] = []
        if joint_boxes:
            meshes.append(_build_joint_box_mesh(builder, box_size))
            for node in nodes:
                node.mesh = 0

        animation = _build_animation(
            builder,
            local_positions,
            local_rotations,
            fps,
            animation_name or "Animation",
        )

        gltf = GLTF2(
            asset=Asset(version="2.0", generator="ai4animation.GLBExporter"),
            scene=0,
            scenes=[Scene(name="Scene", nodes=roots)],
            nodes=nodes,
            meshes=meshes,
            buffers=[Buffer(byteLength=len(builder.data))],
            bufferViews=builder.buffer_views,
            accessors=builder.accessors,
            animations=[animation],
        )

        gltf.set_binary_blob(bytes(builder.data))
        gltf.save_binary(out_path)
        return gltf
