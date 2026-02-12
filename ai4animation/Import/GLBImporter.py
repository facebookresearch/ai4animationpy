import struct
from enum import Enum
from functools import cached_property, lru_cache
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
from ai4animation.Animation.Motion import Hierarchy, Motion
from ai4animation.Math import Quaternion, Tensor, Transform, Vector3
from numpy.typing import NDArray
from PIL import Image
from pygltflib import Accessor, GLTF2, Primitive


class ComponentType(Enum):
    FLOAT = 5126
    UNSIGNED_SHORT = 5123

    SIGNED_BYTE = 5120
    UNSIGNED_BYTE = 5121
    SIGNED_SHORT = 5122
    UNSIGNED_INT = 5125


class ComponentLengthType(Enum):
    FLOAT = 4
    UNSIGNED_SHORT = 2

    SIGNED_BYTE = 1
    UNSIGNED_BYTE = 1
    SIGNED_SHORT = 2
    UNSIGNED_INT = 4


class ComponentFormat(Enum):  #          ctype     standar_size
    #                             ------------   --------------
    FLOAT = "f"  #                       float                4
    UNSIGNED_SHORT = "H"  #     unsigned short                2
    SIGNED_BYTE = "b"  #           signed char                1
    UNSIGNED_BYTE = "B"  #       unsigned char                1
    SIGNED_SHORT = "h"  #                short                2
    UNSIGNED_INT = "I"  #         unsigned int                4


class AccessorType(Enum):
    SCALAR = 1  # 1 elements per accessor type
    VEC2 = 2  #   2 elements per accessor type
    VEC3 = 3  #   3 elements per accessor type
    VEC4 = 4  #   4 elements per accessor type
    MAT4 = 16  # 16 elements per accessor type


class Node:
    def __init__(self, name, index, parent, children, translation, rotation, scale):
        self.Name = name
        self.Index = index
        self.Parent = parent
        self.Children = children
        self.Translation = (
            Vector3.Create() if translation is None else Vector3.Create(translation)
        )  # 1x3
        self.Rotation = (
            Quaternion.ToMatrix(Quaternion.Create())
            if rotation is None
            else Quaternion.ToMatrix(Quaternion.Create(rotation))
        )  # 3x3
        self.Scale = (
            Vector3.Create((1.0, 1.0, 1.0)) if scale is None else Vector3.Create(scale)
        )  # 1x3
        # self.LocalMatrix = Transform.GetMirror(Transform.TRS(self.Translation, self.Rotation, self.Scale), Tensor.AXIS.XPositive)
        self.LocalMatrix = Transform.TR(self.Translation, self.Rotation)
        self.Has_translation_data = False
        self.Has_rotation_data = False
        self.Has_scale_data = False
        # print("Node:", name, index, parent, translation, rotation, scale, self.LocalMatrix)


class Animation:
    def __init__(self, framerate, local_matrices, global_matrices):
        self.Framerate = framerate
        self.DeltaTime = 1 / framerate
        self.LocalTransformations = local_matrices
        self.GlobalTransformations = global_matrices


class Skin:
    def __init__(self, inverse_bind_mats, joints):
        self.Inverse_bind_matrices = inverse_bind_mats
        self.Joints = joints


class Mesh:
    def __init__(
        self,
        name,
        vertices,
        normals,
        triangles,
        skin_indices,
        skin_weights,
        tex_coord,
        image,
    ):
        self.Name = name
        self.Vertices = vertices  # [vertexcount,3]
        self.VertexCount = self.Vertices.shape[0]
        self.Normals = normals  # [vertexcount,3]
        self.Triangles = triangles  # [trianglecount,1]
        self.TriangleCount = int(triangles.shape[0] / 3)
        self.SkinIndices = skin_indices
        self.SkinWeights = skin_weights
        self.HasSkinning = (
            self.SkinIndices is not None
            and self.SkinWeights is not None
            and len(self.SkinIndices) > 0
            and len(self.SkinWeights) > 0
        )
        self.TexCoord = tex_coord
        self.Image = image
        # print(image, tex_coord)
        # print(f"Mesh: Name{name}, Vertices{type(vertices)}, Normals{type(normals)}, Triangles{type(triangles)}, SkinIndices {type(skin_indices)}, SkinWeights{type(skin_weights)}")
        # print(f"Mesh: Name{name}, Vertices{vertices.shape}, Normals{normals.shape}, Triangles{triangles.shape}, SkinIndices {skin_indices.shape}, SkinWeights{skin_weights.shape}")


def bytes_len(component_type: ComponentType, accessor_type: AccessorType) -> int:
    return accessor_type.value * ComponentLengthType[component_type.name].value


def get_struct_flag(component_type: ComponentType, accessor_type: AccessorType) -> str:
    struct_flag = ComponentFormat[component_type.name].value
    flag = struct_flag * accessor_type.value
    return flag


def reshape(data: NDArray, accessor_type: AccessorType) -> NDArray:
    if accessor_type.value == AccessorType.SCALAR.value:
        return data.reshape(1)
    elif accessor_type.value == AccessorType.VEC2.value:
        return data.reshape(2)
    elif accessor_type.value == AccessorType.VEC3.value:
        return data.reshape(3)
    elif accessor_type.value == AccessorType.VEC4.value:
        return data.reshape(4)
    elif accessor_type.value == AccessorType.MAT4.value:
        return data.reshape(4, 4).transpose()
    else:
        raise NotImplementedError(f"Accessor type {accessor_type} is not supported.")


def parse_from_accessor(accessor: Accessor, gltf: GLTF2) -> NDArray:
    assert accessor.bufferView is not None
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[buffer_view.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)

    component_type = ComponentType(accessor.componentType)
    accessor_type = AccessorType[accessor.type]

    b_len = bytes_len(component_type, accessor_type)
    flag = get_struct_flag(component_type, accessor_type)

    assert buffer_view.byteOffset is not None
    assert accessor.byteOffset is not None
    start_loc = buffer_view.byteOffset + accessor.byteOffset

    total_size = accessor.count * b_len
    all_data = data[start_loc : start_loc + total_size]

    elements_per_item = accessor_type.value
    all_values = struct.unpack(flag * accessor.count, all_data)

    result = np.array(all_values).reshape(accessor.count, elements_per_item)

    return result


def parse_texcoords(primitive: Primitive, glb: GLTF2) -> Dict[int, NDArray]:
    if not hasattr(primitive.attributes, "TEXCOORD_0"):
        return {}
    it = 0
    texcoords = {}
    while True:
        texcoord_i = f"TEXCOORD_{it}"
        if not hasattr(primitive.attributes, texcoord_i):
            break
        accessor_id = getattr(primitive.attributes, texcoord_i)
        if accessor_id is None:
            it += 1
            continue
        accessor = glb.accessors[accessor_id]
        texcoord = parse_from_accessor(accessor, glb)
        texcoords[it] = texcoord
        it += 1
    return texcoords


def parse_material(
    primitive: Primitive, glb: GLTF2, image_cache: Dict[int, Image.Image]
) -> Tuple[int, Image.Image]:
    material_id = primitive.material

    if material_id is None or glb.materials is None:
        return -1, Image.new("RGB", (1, 1))

    material = glb.materials[material_id]  # pyre-ignore

    if material.pbrMetallicRoughness.baseColorTexture is None:
        return -1, Image.new("RGB", (1, 1))

    image_id = material.pbrMetallicRoughness.baseColorTexture.index
    texcoord_i = material.pbrMetallicRoughness.baseColorTexture.texCoord

    if image_id in image_cache:
        return texcoord_i, image_cache[image_id]

    image = glb.images[image_id]

    buffer_view = glb.bufferViews[image.bufferView]
    buffer = glb.buffers[buffer_view.buffer]

    data = glb.get_data_from_buffer_uri(buffer.uri)
    loc = buffer_view.byteOffset or 0
    image_data = data[loc : loc + buffer_view.byteLength]

    image_cache[image_id] = Image.open(BytesIO(image_data))

    return texcoord_i, image_cache[image_id]


def parse_joint_indices_and_weights(
    primitive: Primitive, glb: GLTF2
) -> Tuple[NDArray, NDArray]:
    if (
        not hasattr(primitive.attributes, "JOINTS_0")
        or getattr(primitive.attributes, "JOINTS_0") is None
    ):
        skin_joint_indices = np.empty((0, 4), dtype=np.int64)
        joint_weights = np.empty((0, 4), dtype=np.float32)
        # print(f"Final skin_joint_indices shape: {skin_joint_indices.shape}, joint_weights shape: {joint_weights.shape}")
        return skin_joint_indices, joint_weights

    iteration = 0

    skin_joint_indices_all = []
    joint_weights_all = []

    while True:
        joints_i = f"JOINTS_{iteration}"
        weights_i = f"WEIGHTS_{iteration}"

        iteration += 1

        if not hasattr(primitive.attributes, joints_i):
            break

        assert hasattr(primitive.attributes, weights_i), primitive.attributes

        accessor_id = getattr(primitive.attributes, joints_i)

        if accessor_id is None:
            break

        accessor = glb.accessors[accessor_id]
        skin_joint_indices = parse_from_accessor(accessor, glb)

        accessor_id = getattr(primitive.attributes, weights_i)
        accessor = glb.accessors[accessor_id]
        joint_weights = parse_from_accessor(accessor, glb)

        # print(f"JOINTS_{iteration-1} shape: {skin_joint_indices.shape}, WEIGHTS_{iteration-1} shape: {joint_weights.shape}")
        skin_joint_indices_all.append(skin_joint_indices)
        joint_weights_all.append(joint_weights)

    skin_joint_indices = np.concatenate(skin_joint_indices_all, axis=1)
    joint_weights = np.concatenate(joint_weights_all, axis=1)
    # print(f"Final skin_joint_indices shape: {skin_joint_indices.shape}, joint_weights shape: {joint_weights.shape}")
    return skin_joint_indices, joint_weights


class GLB:
    def __init__(self, path) -> None:
        self._path = path
        self._glb = GLTF2().load(path)

    @property
    def Filename(self) -> str:
        import os

        return os.path.splitext(os.path.basename(self._path))[0]

    @classmethod
    @lru_cache(maxsize=1)
    def Create(cls, glb_path: str) -> "GLB":
        return cls(glb_path)

    @cached_property
    def SkinnedMesh(self) -> Mesh:
        vertices_all = []
        normals_all = []
        triangles_all = []
        skin_indices_all = []
        skin_weights_all = []

        vertex_offset = 0
        for mesh in self.Meshes:
            vertices_all.append(mesh.Vertices)
            normals_all.append(mesh.Normals)

            # Adjust triangle indices to account for vertex offset from previous meshes
            triangles = mesh.Triangles + vertex_offset
            triangles_all.append(triangles)

            skin_indices_all.append(mesh.SkinIndices)
            skin_weights_all.append(mesh.SkinWeights)

            # Update vertex offset for next mesh
            vertex_offset += mesh.Vertices.shape[0]

        return Mesh(
            name=self._path,
            vertices=Tensor.Create(np.concatenate(vertices_all, axis=0)),
            normals=Tensor.Create(np.concatenate(normals_all, axis=0)),
            triangles=Tensor.Create(np.concatenate(triangles_all, axis=0)),
            skin_indices=Tensor.Create(np.concatenate(skin_indices_all, axis=0)),
            skin_weights=Tensor.Create(np.concatenate(skin_weights_all, axis=0)),
            tex_coord=None,
            image=None,
        )

    @cached_property
    def Skin(self):
        # print("Skin count: ", len(self._glb.skins))
        assert len(self._glb.skins) >= 1
        skin = self._glb.skins[0]  # only handles first skin for now
        # print("Skin[0]: ", skin)
        assert skin.inverseBindMatrices is not None
        accessor = self._glb.accessors[skin.inverseBindMatrices]
        inverse_bind_mats = parse_from_accessor(accessor, self._glb)
        inverse_bind_mats = (
            np.array(inverse_bind_mats).astype(np.float32).reshape(-1, 4, 4)
        )
        joints = np.asarray(skin.joints)
        assert len(inverse_bind_mats) == len(joints)
        # print("Skin Bind Pose : ", bindPose.shape)
        # print("Skin joints count: ", len(joints))
        return Skin(inverse_bind_mats, joints)

    @cached_property
    def Meshes(self) -> List[Mesh]:
        meshes = []
        images: Dict[int, Image.Image] = {}
        # print("Mesh Count: ", len(self._glb.meshes))
        for mesh_obj in self._glb.meshes:
            assert len(mesh_obj.primitives) >= 1
            for i, primitive in enumerate(mesh_obj.primitives):
                # print("Mesh Primitive: ", primitive)
                accessor = self._glb.accessors[primitive.attributes.POSITION]
                mesh_vertices = parse_from_accessor(accessor, self._glb)
                num_vertices = mesh_vertices.shape[0]

                assert primitive.indices is not None
                accessor = self._glb.accessors[primitive.indices]
                mesh_triangles = parse_from_accessor(accessor, self._glb)

                # skin joint indices and weights
                skin_joint_indices, skin_joint_weights = (
                    parse_joint_indices_and_weights(primitive, self._glb)
                )

                # parse normal/uv texture
                if primitive.attributes.NORMAL is not None:
                    accessor = self._glb.accessors[primitive.attributes.NORMAL]
                    mesh_normals = parse_from_accessor(accessor, self._glb)
                else:
                    mesh_normals = np.zeros((num_vertices, 3), dtype=np.float32)

                texcoords = parse_texcoords(primitive, self._glb)

                texcoord_i, image = parse_material(primitive, self._glb, images)
                assert texcoord_i == -1 or texcoord_i in texcoords, texcoords

                name = mesh_obj.name or ""
                if i > 0:
                    name += f"_{i}"

                mesh = Mesh(
                    name=name,
                    vertices=np.array(mesh_vertices).astype(
                        np.float32
                    ),  # .reshape(-1,3),
                    normals=np.array(mesh_normals).astype(
                        np.float32
                    ),  # .reshape(-1,3),
                    triangles=np.array(mesh_triangles).astype(
                        np.int64
                    ),  # .reshape(-1),
                    skin_indices=np.array(skin_joint_indices).astype(
                        np.int64
                    ),  # .reshape(-1,4),
                    skin_weights=np.array(skin_joint_weights).astype(
                        np.float32
                    ),  # .reshape(-1,4)
                    tex_coord=texcoord_i if texcoord_i != -1 else np.zeros(0),
                    image=image,
                )

                meshes.append(mesh)

        return meshes

    @cached_property
    def _nodes(self) -> List[Node]:
        child_to_parent = {}
        for i, node in enumerate(self._glb.nodes):
            for child in node.children:
                child_to_parent[child] = i
        nodes = [None] * len(self._glb.nodes)
        for i, node in enumerate(self._glb.nodes):
            parent = child_to_parent.get(i, None)
            nodes[i] = Node(
                node.name,
                i,
                parent,
                node.children,
                node.translation,
                node.rotation,
                node.scale,
            )
        return nodes

    @cached_property
    def _nodeNames(self) -> List[str]:
        return [node.Name or None for node in self._nodes]

    @cached_property
    def _nodeParentNames(self) -> List[str]:
        return [
            self._nodes[node.Parent].Name if node.Parent is not None else None
            for node in self._nodes
        ]

    @cached_property
    def _nodeGlobalMatrices(self):
        # FK on nodes
        global_matrices = np.zeros((len(self._nodes), 4, 4))
        for node_idx, node in enumerate(self._nodes):
            global_matrices[node_idx] = (
                node.LocalMatrix
                if node.Parent is None
                else Transform.Multiply(global_matrices[node.Parent], node.LocalMatrix)
            )
        return global_matrices

    @cached_property
    def JointNames(self) -> List[str]:
        return [self._nodeNames[joint_idx] or None for joint_idx in self.Skin.Joints]

    @cached_property
    def JointParents(self) -> List[str]:
        return [
            self._nodeParentNames[joint_idx] or None for joint_idx in self.Skin.Joints
        ]

    @cached_property
    def JointMatrices(self):
        return self._nodeGlobalMatrices[self.Skin.Joints]

    @cached_property
    def _animations(self) -> List[Animation]:
        if len(self._glb.animations) == 0:  # no animation
            print("No animation found in " + self._path)
            return []

        animations: List[Animation] = []
        glb_anim = self._glb.animations[0]  # only handles first animation for now
        # print("Animation[0]: ", glb_anim.name)

        # timestamps
        accessor_input = self._glb.accessors[glb_anim.samplers[0].input]
        timestamps = parse_from_accessor(accessor_input, self._glb)
        num_nodes = len(self._nodes)
        num_frames = len(timestamps)
        framerate = float((num_frames - 1) / timestamps[-1])

        anim_trans = np.zeros((num_frames, num_nodes, 3))
        anim_rots = np.zeros((num_frames, num_nodes, 4))
        anim_rots[..., -1] = 1.0
        anim_scales = np.ones((num_frames, num_nodes, 3))

        translation_channels = []
        rotation_channels = []
        scale_channels = []

        for channel in glb_anim.channels:
            if channel.target.path == "translation":
                translation_channels.append(channel)
            elif channel.target.path == "rotation":
                rotation_channels.append(channel)
            elif channel.target.path == "scale":
                scale_channels.append(channel)

        # Process all channels in a single pass to reduce redundant operations
        for channel in translation_channels:
            sampler = glb_anim.samplers[channel.sampler]
            accessor_output = self._glb.accessors[sampler.output]
            output = parse_from_accessor(accessor_output, self._glb)
            num_frames = max(num_frames, output.shape[0])
            self._nodes[channel.target.node].Has_translation_data = True
            # Handle frame count mismatch: use minimum of available frames
            frames_to_copy = min(output.shape[0], anim_trans.shape[0])
            anim_trans[:frames_to_copy, channel.target.node, :] = output[
                :frames_to_copy
            ]

        for channel in rotation_channels:
            sampler = glb_anim.samplers[channel.sampler]
            accessor_output = self._glb.accessors[sampler.output]
            output = parse_from_accessor(accessor_output, self._glb)
            num_frames = max(num_frames, output.shape[0])
            self._nodes[channel.target.node].Has_rotation_data = True
            # Handle frame count mismatch: use minimum of available frames
            frames_to_copy = min(output.shape[0], anim_rots.shape[0])
            anim_rots[:frames_to_copy, channel.target.node, :] = output[:frames_to_copy]

        for channel in scale_channels:
            sampler = glb_anim.samplers[channel.sampler]
            accessor_output = self._glb.accessors[sampler.output]
            output = parse_from_accessor(accessor_output, self._glb)
            num_frames = max(num_frames, output.shape[0])
            self._nodes[channel.target.node].Has_scale_data = True
            # Handle frame count mismatch: use minimum of available frames
            frames_to_copy = min(output.shape[0], anim_scales.shape[0])
            anim_scales[:frames_to_copy, channel.target.node, :] = output[
                :frames_to_copy
            ]

        # pre-compute scales
        scales = np.ones((num_nodes, num_frames, 3))
        for idx, node in enumerate(self._nodes):
            if node.Parent is not None:
                parent = self._nodes[node.Parent]
                scales[idx] = scales[node.Parent]
                scales[idx] *= (
                    anim_scales[:, parent.Index]
                    if parent.Has_scale_data
                    else parent.Scale
                )

        # Build local transformation matrices
        local_matrices = np.zeros((num_frames, num_nodes, 4, 4))
        for idx, node in enumerate(self._nodes):
            if node.Has_translation_data:
                translations = anim_trans[:, idx]
            else:
                translations = np.broadcast_to(node.Translation, (num_frames, 3))

            if node.Parent is not None:
                translations = translations * scales[idx]

            translations = Tensor.Create(translations)

            if node.Has_rotation_data:
                rotations = Quaternion.ToMatrix(Quaternion.Create(anim_rots[:, idx]))
            else:
                rotations = Tensor.Create(
                    np.broadcast_to(node.Rotation, (num_frames, 3, 3))
                )

            local_matrices[:, idx] = Transform.TR(translations, rotations)

        global_matrices = np.zeros((num_frames, num_nodes, 4, 4))
        # FK
        for node_idx, node in enumerate(self._nodes):
            global_matrices[:, node_idx] = (
                local_matrices[:, node_idx]
                if node.Parent is None
                else Transform.Multiply(
                    global_matrices[:, node.Parent], local_matrices[:, node_idx]
                )
            )

        animations.append(Animation(framerate, local_matrices, global_matrices))
        return animations

    def FindParent(self, name, candidates):
        node = next(x for x in self._nodes if x.Name == name)
        parent = None
        pivot = node
        while pivot.Parent is not None:
            pivot = self._nodes[pivot.Parent]
            if pivot.Name in candidates:
                parent = pivot
                break
        return parent

    def LoadMotion(self, names=None, floor=None):
        anim = self._animations[0]
        if names is None:
            hierarchy = Hierarchy(self._nodeNames, self._nodeParentNames)
            frames = anim.GlobalTransformations
        else:
            parents = [self.FindParent(name, names) for name in names]
            parentNames = [
                parent.Name if parent is not None else None for parent in parents
            ]
            hierarchy = Hierarchy(names, parentNames)
            name_to_index = {node.Name: i for i, node in enumerate(self._nodes)}
            indices = [
                name_to_index[name]
                for name in hierarchy.BoneNames
                if name in name_to_index
            ]
            frames = anim.GlobalTransformations[:, indices]

        if floor is not None:
            floor_node = next((x for x in self._nodes if x.Name == floor), None)
            if floor_node is None:
                print(
                    f"Floor node '{floor}' not found in GLB file. Available nodes: {self._nodeNames}"
                )
            else:
                index = floor_node.Index
                offset = anim.GlobalTransformations[:, index]
                frames = Transform.TransformationTo(frames, offset.reshape(-1, 1, 4, 4))

        return Motion(
            name=self.Filename,
            hierarchy=hierarchy,
            frames=frames,
            framerate=anim.Framerate,
        )

    def Debug(self):
        glb = self
        print("=== GLB DATA EXTRACTION DEBUG ===")

        # Check skinned meshes
        print(f"Number of skinned meshes: {len(glb.Meshes)}")
        skinnedMesh = glb.Meshes[-1]  # Last mesh (the one being used)

        print(f"\nSkinned Mesh Data:")
        print(f"  Vertices shape: {skinnedMesh.Vertices.shape}")
        print(f"  Normals shape: {skinnedMesh.Normals.shape}")
        print(f"  Triangles shape: {skinnedMesh.Triangles.shape}")
        print(f"  SkinIndices shape: {skinnedMesh.SkinIndices.shape}")
        print(f"  SkinWeights shape: {skinnedMesh.SkinWeights.shape}")

        # Check skin data
        skin_data = glb.Skin
        print(f"\nSkin Data:")
        print(f"  Bind matrices shape: {skin_data.Inverse_bind_matrices.shape}")
        print(f"  Joint indices shape: {skin_data.Joints.shape}")

        # Check first few vertices
        print(f"\nFirst 3 vertices:")
        for i in range(min(3, skinnedMesh.Vertices.shape[0])):
            print(f"  Vertex {i}: {skinnedMesh.Vertices[i]}")
            print(f"    Skin indices: {skinnedMesh.SkinIndices[i]}")
            print(f"    Skin weights: {skinnedMesh.SkinWeights[i]}")
            print(f"    Weight sum: {np.sum(skinnedMesh.SkinWeights[i])}")

        # Check joint names
        print(f"\nJoint Names (first 10):")
        for i, name in enumerate(glb.JointNames[:10]):
            print(f"  {i}: {name}")

        print(f"\nTotal joints: {len(glb.JointNames)}")
        print(f"Joint matrices shape: {glb.JointMatrices.shape}")
