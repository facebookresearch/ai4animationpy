# # Copyright (c) Meta Platforms, Inc. and affiliates.
# import os
# import re

# import numpy as np
# from ai4animation.Animation.Motion import Hierarchy, Motion
# from ai4animation.Math import Rotation, Tensor, Transform

# channelmap = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}

# channelmap_inv = {
#     "x": "Xrotation",
#     "y": "Yrotation",
#     "z": "Zrotation",
# }

# ordermap = {
#     "x": 0,
#     "y": 1,
#     "z": 2,
# }


# def _euler_to_rotation_matrix(angles, order):
#     angles = Tensor.Create(angles)

#     axis_to_rotation = {
#         "x": Rotation.RotationX,
#         "y": Rotation.RotationY,
#         "z": Rotation.RotationZ,
#     }

#     axis_to_index = {"x": 0, "y": 1, "z": 2}

#     r0 = axis_to_rotation[order[0]](angles[..., axis_to_index[order[0]]])
#     r1 = axis_to_rotation[order[1]](angles[..., axis_to_index[order[1]]])
#     r2 = axis_to_rotation[order[2]](angles[..., axis_to_index[order[2]]])

#     return Tensor.MatMul(r0, Tensor.MatMul(r1, r2))


# def LoadMotion(filename, order=None, fps = 60.0) -> Motion:
#     if not os.path.isfile(filename):
#         raise FileNotFoundError(f"BVH file not found: {filename}")

#     f = open(filename, "r")

#     i = 0
#     active = -1
#     end_site = False

#     names = []
#     offsets = np.array([]).reshape((0, 3))
#     parents = np.array([], dtype=int)

#     channels = None
#     framerate = 1.0 / fps

#     for line in f:
#         if "HIERARCHY" in line:
#             continue
#         if "MOTION" in line:
#             continue

#         rmatch = re.match(r"ROOT (\w+)", line)
#         if rmatch:
#             names.append(rmatch.group(1))
#             offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
#             parents = np.append(parents, active)
#             active = len(parents) - 1
#             continue

#         if "{" in line:
#             continue

#         if "}" in line:
#             if end_site:
#                 end_site = False
#             else:
#                 active = parents[active]
#             continue

#         offmatch = re.match(
#             r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line
#         )
#         if offmatch:
#             if not end_site:
#                 offsets[active] = np.array([list(map(float, offmatch.groups()))])
#             continue

#         chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
#         if chanmatch:
#             channels = int(chanmatch.group(1))
#             if order is None:
#                 channelis = 0 if channels == 3 else 3
#                 channelie = 3 if channels == 3 else 6
#                 parts = line.split()[2 + channelis : 2 + channelie]
#                 if any([p not in channelmap for p in parts]):
#                     continue
#                 order = "".join([channelmap[p] for p in parts])
#             continue

#         jmatch = re.match(r"\s*JOINT\s+(\w+)", line)
#         if jmatch:
#             names.append(jmatch.group(1))
#             offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
#             parents = np.append(parents, active)
#             active = len(parents) - 1
#             continue

#         if "End Site" in line:
#             end_site = True
#             continue

#         fmatch = re.match(r"\s*Frames:\s+(\d+)", line)
#         if fmatch:
#             fnum = int(fmatch.group(1))
#             positions = offsets[np.newaxis].repeat(fnum, axis=0)
#             rotations = np.zeros((fnum, len(names), 3))
#             continue

#         fmatch = re.match(r"\s*Frame Time:\s+([\d\.]+)", line)
#         if fmatch:
#             framerate = float(fmatch.group(1))
#             continue

#         dmatch = line.strip().split(" ")
#         if dmatch:
#             data_block = np.array(list(map(float, dmatch)))
#             N = len(parents)
#             fi = i
#             if channels == 3:
#                 positions[fi, 0:1] = data_block[0:3]
#                 rotations[fi, :] = data_block[3:].reshape(N, 3)
#             elif channels == 6:
#                 data_block = data_block.reshape(N, 6)
#                 positions[fi, :] = data_block[:, 0:3]
#                 rotations[fi, :] = data_block[:, 3:6]
#             elif channels == 9:
#                 positions[fi, 0] = data_block[0:3]
#                 data_block = data_block[3:].reshape(N - 1, 9)
#                 rotations[fi, 1:] = data_block[:, 3:6]
#                 positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
#             else:
#                 raise Exception("Too many channels! %i" % channels)

#             i += 1

#     f.close()

#     data = {
#         "rotations": rotations, # [num_frames, num_joints, 3] Euler angles
#         "positions": positions, # [num_frames, num_joints, 3] local positions
#         "offsets": offsets,
#         "parents": parents, # [num_joints] parent indices
#         "names": names, # [num_joints] joint names
#         "order": order if order is not None else "zyx", # rotation order string
#         "framerate": framerate, # time per frame in seconds
#     }

#     num_frames = rotations.shape[0]
#     num_joints = rotations.shape[1]

#     # localPositions = 0.01 * bvhData['positions'].copy().astype(np.float32)
#     # localRotations = quat.unroll(quat.from_euler(np.radians(bvhData['rotations']), order=bvhData['order']))
#     # globalRotations, globalPositions = quat.fk(localRotations, localPositions, parents)

#     rotation_matrices = _euler_to_rotation_matrix(rotations, order)
#     local_positions = Tensor.Create(positions)
#     local_positions = 0.01 * local_positions
#     local_matrices = Transform.TR(local_positions, rotation_matrices)

#     global_matrices = np.zeros((num_frames, num_joints, 4, 4))
#     for joint_idx in range(num_joints):
#         parent_idx = parents[joint_idx]
#         if parent_idx == -1:
#             global_matrices[:, joint_idx] = local_matrices[:, joint_idx]
#         else:
#             global_matrices[:, joint_idx] = Transform.Multiply(
#                 global_matrices[:, parent_idx], local_matrices[:, joint_idx]
#             )

#     parent_names = []
#     for parent_idx in parents:
#         if parent_idx == -1:
#             parent_names.append(None)
#         else:
#             parent_names.append(names[parent_idx])

#     hierarchy = Hierarchy(bone_names=names, parent_names=parent_names)

#     framerate = 1.0 / framerate
#     name = os.path.splitext(os.path.basename(filename))[0]

#     return Motion(
#         name=name,
#         hierarchy=hierarchy,
#         frames=global_matrices,
#         framerate=framerate,
#     )
