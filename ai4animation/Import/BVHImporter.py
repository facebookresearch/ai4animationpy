# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import re

import numpy as np
from ai4animation.Animation.Motion import Hierarchy, Motion
from ai4animation.Math import Rotation, Tensor, Transform, Vector3

channelmap = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}


def _euler_to_rotation_matrix(angles, order):
    angles = Tensor.Create(angles)

    axis_to_rotation = {
        "x": Rotation.RotationX,
        "y": Rotation.RotationY,
        "z": Rotation.RotationZ,
    }

    r0 = axis_to_rotation[order[0]](angles[..., 0])
    r1 = axis_to_rotation[order[1]](angles[..., 1])
    r2 = axis_to_rotation[order[2]](angles[..., 2])

    return Tensor.MatMul(r0, Tensor.MatMul(r1, r2))


def _resolve_joint_corrections(joint_names, joint_corrections):
    corrections = Vector3.Zero(len(joint_names))
    if joint_corrections is None:
        return corrections

    if isinstance(joint_corrections, dict):
        name_to_index = {name: i for i, name in enumerate(joint_names)}
        for joint_name, correction in joint_corrections.items():
            joint_idx = name_to_index.get(joint_name)
            if joint_idx is None:
                raise ValueError(
                    f"Joint correction specified for unknown BVH joint '{joint_name}'."
                )
            corrections[joint_idx] = Vector3.Create(correction)
        return corrections

    joint_corrections = Tensor.Create(joint_corrections)
    if joint_corrections.shape != corrections.shape:
        raise ValueError(
            f"joint_corrections must have shape {corrections.shape}, got {joint_corrections.shape}."
        )
    return joint_corrections


class BVH:
    def __init__(
        self,
        path,
        scale=1.0,
        mirror_axis: Vector3.Axis | None = None,
        joint_corrections=None,
    ):
        self._path = path
        self._scale = scale
        if mirror_axis is not None and not isinstance(mirror_axis, Vector3.Axis):
            raise TypeError(
                "mirror_axis must be a Vector3.Axis value, e.g. Vector3.Axis.XPositive."
            )
        self._mirror_axis = mirror_axis

        if not os.path.isfile(path):
            raise FileNotFoundError(f"BVH file not found: {path}")

        with open(path, "r") as f:
            lines = f.readlines()

        names, offsets, parents, channel_counts, order = self._parse_hierarchy(
            lines, path
        )
        positions, rotations, framerate = self._parse_motion(
            lines, names, offsets, channel_counts, path
        )

        if order is None:
            raise ValueError(f"Could not detect rotation order from BVH file: {path}")
        if framerate is None:
            raise ValueError(f"Could not detect frame time from BVH file: {path}")

        self._names = names
        self._parents = parents
        self._offsets = offsets
        self._positions = positions
        self._rotations = rotations
        self._order = order
        self._framerate = 1.0 / framerate
        self._channels = channel_counts
        self._joint_corrections = _resolve_joint_corrections(
            self._names, joint_corrections
        )

    @staticmethod
    def _parse_hierarchy(lines, path):
        active = -1
        names = []
        offsets = np.array([], dtype=np.float32).reshape((0, 3))
        parents = np.array([], dtype=int)
        channel_counts = []
        order = None

        for line in lines:
            if "HIERARCHY" in line:
                continue
            if "MOTION" in line:
                break

            rmatch = re.match(r"\s*ROOT\s+(.+?)\s*$", line)
            if rmatch:
                names.append(rmatch.group(1))
                offsets = np.append(
                    offsets, np.array([[0, 0, 0]], dtype=np.float32), axis=0
                )
                parents = np.append(parents, active)
                channel_counts.append(0)
                active = len(parents) - 1
                continue

            if "{" in line:
                continue

            if "}" in line:
                if active != -1:
                    active = parents[active]
                continue

            offmatch = re.match(
                r"\s*OFFSET\s+([\-\d\.eE\+]+)\s+([\-\d\.eE\+]+)\s+([\-\d\.eE\+]+)", line
            )
            if offmatch:
                offsets[active] = np.array(
                    [list(map(float, offmatch.groups()))], dtype=np.float32
                )
                continue

            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                channel_counts[active] = channels
                if order is None:
                    channelis = 0 if channels == 3 else 3
                    channelie = 3 if channels == 3 else 6
                    parts = line.split()[2 + channelis : 2 + channelie]
                    if all(p in channelmap for p in parts):
                        order = "".join([channelmap[p] for p in parts])
                continue

            jmatch = re.match(r"\s*JOINT\s+(.+?)\s*$", line)
            if jmatch:
                names.append(jmatch.group(1))
                offsets = np.append(
                    offsets, np.array([[0, 0, 0]], dtype=np.float32), axis=0
                )
                parents = np.append(parents, active)
                channel_counts.append(0)
                active = len(parents) - 1
                continue

            if "End Site" in line:
                parent_name = names[active] if active != -1 else "End"
                site_name = f"{parent_name}Site"
                suffix = 1
                while site_name in names:
                    suffix += 1
                    site_name = f"{parent_name}Site{suffix}"

                names.append(site_name)
                offsets = np.append(
                    offsets, np.array([[0, 0, 0]], dtype=np.float32), axis=0
                )
                parents = np.append(parents, active)
                channel_counts.append(0)
                active = len(parents) - 1
                continue

        return names, offsets, parents, channel_counts, order

    @staticmethod
    def _parse_motion(lines, names, offsets, channel_counts, path):
        positions = None
        rotations = None
        framerate = None
        i = 0

        for line in lines:
            fmatch = re.match(r"\s*Frames:\s+(\d+)", line)
            if fmatch:
                fnum = int(fmatch.group(1))
                positions = offsets[np.newaxis].repeat(fnum, axis=0)
                rotations = np.zeros((fnum, len(names), 3), dtype=np.float32)
                continue

            fmatch = re.match(r"\s*Frame Time:\s+([\d\.eE\+\-]+)", line)
            if fmatch:
                framerate = float(fmatch.group(1))
                continue

            if positions is None:
                continue

            dmatch = line.strip().split()
            if not dmatch:
                continue

            data_block = np.array(list(map(float, dmatch)), dtype=np.float32)
            fi = i
            cursor = 0

            for joint_idx, num_channels in enumerate(channel_counts):
                if num_channels == 0:
                    continue

                joint_block = data_block[cursor : cursor + num_channels]
                if joint_block.size != num_channels:
                    raise ValueError(
                        f"Invalid BVH frame data in {path}: expected {num_channels} values for joint {names[joint_idx]}, got {joint_block.size}."
                    )

                if num_channels == 3:
                    rotations[fi, joint_idx] = joint_block
                elif num_channels == 6:
                    positions[fi, joint_idx] = joint_block[0:3]
                    rotations[fi, joint_idx] = joint_block[3:6]
                elif num_channels == 9:
                    positions[fi, joint_idx] += joint_block[0:3] * joint_block[6:9]
                    rotations[fi, joint_idx] = joint_block[3:6]
                else:
                    raise ValueError(
                        f"Unsupported BVH channel count {num_channels} for joint {names[joint_idx]}."
                    )

                cursor += num_channels

            if cursor != len(data_block):
                raise ValueError(
                    f"Invalid BVH frame data in {path}: consumed {cursor} values but frame contains {len(data_block)}."
                )

            i += 1

        return positions, rotations, framerate

    @property
    def Filename(self) -> str:
        return os.path.splitext(os.path.basename(self._path))[0]

    def FindParent(self, name, candidates):
        idx = self._names.index(name)
        parent_idx = self._parents[idx]
        while parent_idx != -1:
            if self._names[parent_idx] in candidates:
                return self._names[parent_idx]
            parent_idx = self._parents[parent_idx]
        return None

    def LoadMotion(self, names=None, floor=None) -> Motion:
        num_frames = self._rotations.shape[0]
        num_joints = self._rotations.shape[1]

        rotation_matrices = _euler_to_rotation_matrix(self._rotations, self._order)
        local_positions = Tensor.Create(self._positions)
        local_positions = self._scale * local_positions
        local_matrices = Transform.TR(local_positions, rotation_matrices)

        if self._mirror_axis is not None:
            local_matrices = Transform.GetMirror(local_matrices, self._mirror_axis)

        global_matrices = np.zeros((num_frames, num_joints, 4, 4), dtype=np.float32)
        for joint_idx in range(num_joints):
            parent_idx = self._parents[joint_idx]
            if parent_idx == -1:
                global_matrices[:, joint_idx] = local_matrices[:, joint_idx]
            else:
                global_matrices[:, joint_idx] = Transform.Multiply(
                    global_matrices[:, parent_idx], local_matrices[:, joint_idx]
                )

        if not Tensor.All(self._joint_corrections == 0.0):
            correction_update = Transform.TR(
                Vector3.Zero(num_joints), Rotation.Euler(self._joint_corrections)
            ).reshape(1, num_joints, 4, 4)
            global_matrices = Transform.Multiply(global_matrices, correction_update)

        if names is None:
            names = [
                name
                for name, num_channels in zip(self._names, self._channels)
                if num_channels > 0
            ]

        parent_names = [self.FindParent(name, names) for name in names]
        hierarchy = Hierarchy(bone_names=names, parent_names=parent_names)
        name_to_index = {name: i for i, name in enumerate(self._names)}
        indices = [name_to_index[name] for name in names if name in name_to_index]
        frames = global_matrices[:, indices]

        if floor is not None:
            if floor not in self._names:
                print(
                    f"Floor node '{floor}' not found in BVH file. Available nodes: {self._names}"
                )
            else:
                floor_idx = self._names.index(floor)
                offset = global_matrices[:, floor_idx]
                frames = Transform.TransformationTo(frames, offset.reshape(-1, 1, 4, 4))

        return Motion(
            name=self.Filename,
            hierarchy=hierarchy,
            frames=frames,
            framerate=self._framerate,
        )
