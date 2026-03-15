# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

try:
    from fbx import (
        FbxAMatrix,
        FbxAnimLayer,
        FbxAnimStack,
        FbxAxisSystem,
        FbxCriteria,
        FbxDeformer,
        FbxGeometryConverter,
        FbxImporter,
        FbxIOSettings,
        FbxLayerElement,
        FbxManager,
        FbxScene,
        FbxTime,
        IOSROOT,
    )
except ImportError:
    raise ImportError(
        "Autodesk FBX SDK Python bindings not found.\n"
        "Install guide:\n"
        "  1. Download FBX SDK from https://aps.autodesk.com/developer/overview/fbx-sdk \n"
        "  2. Download FBX SDK Python Bindings\n"
        "  3. Set FBXSDK_ROOT env var to the FBX SDK install path (i.e. $env:FBXSDK_ROOT = \"C:\Program Files\Autodesk\FBX\FBX SDK\\2020.3.9\") \n"
        "  4. Set FBXSDK_COMPILER env var (i.e. $env:FBXSDK_COMPILER=\"vs2022\")\n"
        "  5. pip install --force-reinstall -v sip==6.6.2\n"
        "  6. pip install . (in the Python Bindings folder)\n"
    )

from ai4animation.Animation.Motion import Hierarchy, Motion
from ai4animation.Import.ModelImporter import Mesh, ModelImporter, Skin
from ai4animation.Math import Quaternion, Tensor, Transform, Vector3


class Node:
    def __init__(self, name, index, parent, children, translation, rotation):
        self.Name = name
        self.Index = index
        self.Parent = parent
        self.Children = children
        self.Translation = Vector3.Create() if translation is None else Vector3.Create(translation)
        self.Rotation = Quaternion.ToMatrix(Quaternion.Create()) if rotation is None else Quaternion.ToMatrix(Quaternion.Create(rotation))
        self.LocalMatrix = Transform.TR(self.Translation, self.Rotation)


class Animation:
    def __init__(self, framerate, local_matrices, global_matrices):
        self.Framerate = framerate
        self.DeltaTime = 1 / framerate
        self.LocalTransformations = local_matrices
        self.GlobalTransformations = global_matrices


def _collect_nodes(root):
    result = []

    def _recurse(node, parent_idx):
        idx = len(result)
        result.append((node, parent_idx))
        for i in range(node.GetChildCount()):
            _recurse(node.GetChild(i), idx)

    _recurse(root, -1)
    return result


def _detect_framerate(scene, anim_stack, flat_nodes):
    time_mode = scene.GetGlobalSettings().GetTimeMode()
    framerate = FbxTime.GetFrameRate(time_mode)
    if framerate >= 15:
        return framerate

    num_layers = anim_stack.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimLayer.ClassId))
    if num_layers > 0:
        layer = anim_stack.GetSrcObject(FbxCriteria.ObjectType(FbxAnimLayer.ClassId), 0)
        for fbx_node, _ in flat_nodes:
            curve = fbx_node.LclTranslation.GetCurve(layer, "X")
            if curve and curve.KeyGetCount() > 1:
                n = curve.KeyGetCount()
                t0 = curve.KeyGetTime(0).GetSecondDouble()
                t1 = curve.KeyGetTime(n - 1).GetSecondDouble()
                if t1 > t0:
                    return (n - 1) / (t1 - t0)
                break
    return 30.0


def _get_unit_scale(scene):
    return scene.GetGlobalSettings().GetSystemUnit().GetScaleFactor() / 100.0


def _extract_normals(fbx_mesh, num_vertices):
    normals = np.zeros((num_vertices, 3), dtype=np.float32)
    for layer_idx in range(fbx_mesh.GetLayerCount()):
        layer_normals = fbx_mesh.GetLayer(layer_idx).GetNormals()
        if not layer_normals:
            continue

        mapping = layer_normals.GetMappingMode()
        reference = layer_normals.GetReferenceMode()

        if mapping == FbxLayerElement.EMappingMode.eByControlPoint:
            for vi in range(num_vertices):
                if reference == FbxLayerElement.EReferenceMode.eDirect:
                    n = layer_normals.GetDirectArray().GetAt(vi)
                elif reference == FbxLayerElement.EReferenceMode.eIndexToDirect:
                    n = layer_normals.GetDirectArray().GetAt(layer_normals.GetIndexArray().GetAt(vi))
                else:
                    continue
                normals[vi] = [n[0], n[1], n[2]]
            break

        elif mapping == FbxLayerElement.EMappingMode.eByPolygonVertex:
            counts = np.zeros(num_vertices, dtype=np.int32)
            pv_idx = 0
            for pi in range(fbx_mesh.GetPolygonCount()):
                for pvi in range(fbx_mesh.GetPolygonSize(pi)):
                    vi = fbx_mesh.GetPolygonVertex(pi, pvi)
                    if reference == FbxLayerElement.EReferenceMode.eDirect:
                        n = layer_normals.GetDirectArray().GetAt(pv_idx)
                    elif reference == FbxLayerElement.EReferenceMode.eIndexToDirect:
                        n = layer_normals.GetDirectArray().GetAt(layer_normals.GetIndexArray().GetAt(pv_idx))
                    else:
                        pv_idx += 1
                        continue
                    normals[vi] += [n[0], n[1], n[2]]
                    counts[vi] += 1
                    pv_idx += 1
            mask = counts > 0
            normals[mask] /= counts[mask, np.newaxis]
            break

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals /= norms
    return normals

def _extract_skin(fbx_mesh, num_vertices, bone_name_to_joint_idx):
    skin_indices = np.zeros((num_vertices, 4), dtype=np.int64)
    skin_weights = np.zeros((num_vertices, 4), dtype=np.float32)

    num_skins = fbx_mesh.GetDeformerCount(FbxDeformer.EDeformerType.eSkin)
    if num_skins == 0:
        return skin_indices, skin_weights

    skin = fbx_mesh.GetDeformer(0, FbxDeformer.EDeformerType.eSkin)
    influence_counts = np.zeros(num_vertices, dtype=np.int32)

    for ci in range(skin.GetClusterCount()):
        cluster = skin.GetCluster(ci)
        bone = cluster.GetLink()
        if bone is None:
            continue
        joint_idx = bone_name_to_joint_idx.get(bone.GetName(), -1)
        if joint_idx < 0:
            continue

        lIndices = cluster.GetControlPointIndices()
        lWeights = cluster.GetControlPointWeights()
        for wi in range(cluster.GetControlPointIndicesCount()):
            vi = lIndices[wi]
            w = lWeights[wi]
            if vi >= num_vertices or w <= 0:
                continue
            count = influence_counts[vi]
            if count < 4:
                skin_indices[vi, count] = joint_idx
                skin_weights[vi, count] = w
                influence_counts[vi] += 1
            else:
                min_idx = np.argmin(skin_weights[vi])
                if w > skin_weights[vi, min_idx]:
                    skin_indices[vi, min_idx] = joint_idx
                    skin_weights[vi, min_idx] = w

    row_sums = skin_weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    skin_weights /= row_sums
    return skin_indices, skin_weights


def _extract_animation(flat_nodes, scene, anim_stack, timestamps, unit_scale):
    num_nodes = len(flat_nodes)
    num_frames = len(timestamps)

    anim_layer = None
    num_layers = anim_stack.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimLayer.ClassId))
    if num_layers > 0:
        anim_layer = anim_stack.GetSrcObject(FbxCriteria.ObjectType(FbxAnimLayer.ClassId), 0)

    translations = np.zeros((num_frames, num_nodes, 3), dtype=np.float32)
    rotations = np.zeros((num_frames, num_nodes, 4), dtype=np.float32)
    rotations[..., 3] = 1.0

    #only consider animated nodes
    animated_indices = []
    animated_fbx_nodes = []
    for ni, (fbx_node, _) in enumerate(flat_nodes):
        has_curves = False
        if anim_layer:
            for ch in ("X", "Y", "Z"):
                if fbx_node.LclTranslation.GetCurve(anim_layer, ch):
                    has_curves = True
                    break
                if fbx_node.LclRotation.GetCurve(anim_layer, ch):
                    has_curves = True
                    break
        if has_curves:
            animated_indices.append(ni)
            animated_fbx_nodes.append(fbx_node)
        else:
            m = fbx_node.EvaluateLocalTransform()
            t, q = m.GetT(), m.GetQ()
            translations[:, ni] = [t[0] * unit_scale, t[1] * unit_scale, t[2] * unit_scale]
            rotations[:, ni] = [q[0], q[1], q[2], q[3]]

    num_animated = len(animated_indices)
    print(f"[FBX] {num_nodes} nodes, {num_frames} frames")

    #this needs to be optimized
    eval_time = FbxTime()
    us = unit_scale
    for fi in range(num_frames):
        eval_time.SetSecondDouble(timestamps[fi])
        for j in range(num_animated):
            ni = animated_indices[j]
            m = animated_fbx_nodes[j].EvaluateLocalTransform(eval_time)
            t, q = m.GetT(), m.GetQ()
            translations[fi, ni, 0] = t[0] * us
            translations[fi, ni, 1] = t[1] * us
            translations[fi, ni, 2] = t[2] * us
            rotations[fi, ni, 0] = q[0]
            rotations[fi, ni, 1] = q[1]
            rotations[fi, ni, 2] = q[2]
            rotations[fi, ni, 3] = q[3]
        if fi > 0 and fi % 1000 == 0:
            print(f"[FBX]   frame {fi}/{num_frames}")

    rot_matrices = Quaternion.ToMatrix(Quaternion.Create(rotations))
    local_matrices = Transform.TR(Tensor.Create(translations), rot_matrices)
    return local_matrices


class FBX(ModelImporter):
    def __init__(self, path) -> None:
        self._path = path

        manager = FbxManager.Create()
        ios = FbxIOSettings.Create(manager, IOSROOT)
        manager.SetIOSettings(ios)

        lImporter = FbxImporter.Create(manager, "")
        abs_path = os.path.abspath(path)
        if not lImporter.Initialize(abs_path, -1, manager.GetIOSettings()):
            manager.Destroy()
            raise FileNotFoundError(f"Failed to load: {abs_path}")

        scene = FbxScene.Create(manager, "")
        lImporter.Import(scene)
        lImporter.Destroy()

        FbxAxisSystem.MayaYUp.ConvertScene(scene)
        unit_scale = _get_unit_scale(scene)

        # Triangulate all meshes
        converter = FbxGeometryConverter(manager)
        converter.Triangulate(scene, True)

        flat_nodes = _collect_nodes(scene.GetRootNode())
        num_nodes = len(flat_nodes)

        skin_cluster_bone_names = set()
        for fbx_node, _ in flat_nodes:
            mesh = fbx_node.GetMesh()
            if mesh is None:
                continue
            for si in range(mesh.GetDeformerCount(FbxDeformer.EDeformerType.eSkin)):
                sd = mesh.GetDeformer(si, FbxDeformer.EDeformerType.eSkin)
                for ci in range(sd.GetClusterCount()):
                    bone = sd.GetCluster(ci).GetLink()
                    if bone is not None:
                        skin_cluster_bone_names.add(bone.GetName())

        # Map original FBX names to flat_nodes indices
        orig_name_to_indices: Dict[str, List[int]] = {}
        for i, (fbx_node, _) in enumerate(flat_nodes):
            orig_name_to_indices.setdefault(fbx_node.GetName(), []).append(i)

        # Find skeleton node indices (cluster bones + all ancestors)
        skeleton_node_indices = set()
        for bone_name in skin_cluster_bone_names:
            for ni in orig_name_to_indices.get(bone_name, []):
                skeleton_node_indices.add(ni)
                # Walk up parent chain
                pi = flat_nodes[ni][1]
                while pi >= 0:
                    skeleton_node_indices.add(pi)
                    pi = flat_nodes[pi][1]

        # set of node indices that are part of skeleton
        skin_joint_node_idx_set = skeleton_node_indices

        # assign original names to skin joints first
        final_names = [None] * num_nodes
        claimed_names = set()
        for i, (fbx_node, _) in enumerate(flat_nodes):
            if i in skin_joint_node_idx_set:
                name = fbx_node.GetName()
                if name not in claimed_names:
                    final_names[i] = name
                    claimed_names.add(name)

        #assign names to remaining nodes
        name_counts = {name: 0 for name in claimed_names}
        for i, (fbx_node, _) in enumerate(flat_nodes):
            if final_names[i] is not None:
                continue
            name = fbx_node.GetName()
            if name in name_counts:
                name_counts[name] += 1
                final_names[i] = f"{name}_{name_counts[name]}"
            else:
                name_counts[name] = 0
                final_names[i] = name

        # --- Build node hierarchy ---
        self._nodes: List[Node] = []
        node_name_to_idx: Dict[str, int] = {}
        for i, (fbx_node, parent_idx) in enumerate(flat_nodes):
            m = fbx_node.EvaluateLocalTransform()
            t = m.GetT()
            q = m.GetQ()
            name = final_names[i]
            node_name_to_idx[name] = i
            self._nodes.append(Node(
                name=name,
                index=i,
                parent=parent_idx if parent_idx >= 0 else None,
                children=[],
                translation=(t[0] * unit_scale, t[1] * unit_scale, t[2] * unit_scale),
                rotation=(q[0], q[1], q[2], q[3]),
            ))

        for node in self._nodes:
            if node.Parent is not None:
                self._nodes[node.Parent].Children.append(node.Index)

        self._nodeNames = [n.Name for n in self._nodes]
        self._nodeParentNames = [
            self._nodes[n.Parent].Name if n.Parent is not None else None
            for n in self._nodes
        ]

        # FK
        self._nodeGlobalMatrices = np.zeros((num_nodes, 4, 4), dtype=np.float32)
        for idx, node in enumerate(self._nodes):
            self._nodeGlobalMatrices[idx] = (
                node.LocalMatrix if node.Parent is None
                else Transform.Multiply(self._nodeGlobalMatrices[node.Parent], node.LocalMatrix)
            )

        # Maps FBX node object id to our node index
        fbx_node_id_to_idx: Dict[int, int] = {}
        for i, (fbx_node, _) in enumerate(flat_nodes):
            fbx_node_id_to_idx[id(fbx_node)] = i

        # Use the expanded skeleton set (cluster bones + all ancestors)
        skin_joint_node_indices = sorted(skeleton_node_indices)
        skin_joint_idx_set = skeleton_node_indices

        # Map from FBX bone node id to joint index (for skin weight extraction)
        fbx_bone_id_to_joint_idx: Dict[int, int] = {}
        for ji, ni in enumerate(skin_joint_node_indices):
            fbx_bone_id_to_joint_idx[id(flat_nodes[ni][0])] = ji

        # Build name-based joint lookup for skin extraction
        bone_name_to_joint_idx: Dict[str, int] = {}
        for ji, ni in enumerate(skin_joint_node_indices):
            bone_name = self._nodeNames[ni]
            bone_name_to_joint_idx[bone_name] = ji

        # --- Extract meshes ---
        self._meshes: List[Mesh] = []
        for fbx_node, _ in flat_nodes:
            mesh = fbx_node.GetMesh()
            if mesh is None:
                continue

            num_verts = mesh.GetControlPointsCount()
            ctrl_points = mesh.GetControlPoints()
            vertices = np.zeros((num_verts, 3), dtype=np.float32)
            for vi in range(num_verts):
                v = ctrl_points[vi]
                vertices[vi] = [v[0] * unit_scale, v[1] * unit_scale, v[2] * unit_scale]

            num_polys = mesh.GetPolygonCount()
            triangles = np.zeros(num_polys * 3, dtype=np.int64)
            for pi in range(num_polys):
                for pvi in range(3):
                    triangles[pi * 3 + pvi] = mesh.GetPolygonVertex(pi, pvi)

            normals = _extract_normals(mesh, num_verts)
            skin_idx, skin_w = _extract_skin(mesh, num_verts, bone_name_to_joint_idx)

            actually_skinned = np.any(skin_w > 0)
            if actually_skinned:
                max_bone_idx = skin_idx.max()
                print(f"[FBX]   Mesh '{fbx_node.GetName()}': {num_verts} verts, {num_polys} tris, skinned, "
                      f"max_bone_idx={max_bone_idx}, weight_range=[{skin_w[skin_w>0].min():.4f}, {skin_w.max():.4f}], "
                      f"verts_range=[{vertices.min():.4f}, {vertices.max():.4f}]")
            else:
                print(f"[FBX]   Mesh '{fbx_node.GetName()}': {num_verts} verts, {num_polys} tris, NOT skinned")

            self._meshes.append(Mesh(
                name=fbx_node.GetName(),
                vertices=vertices,
                normals=normals,
                triangles=triangles,
                skin_indices=skin_idx if actually_skinned else np.zeros((0, 4), dtype=np.int64),
                skin_weights=skin_w if actually_skinned else np.zeros((0, 4), dtype=np.float32),
            ))

        print(f"[FBX] Extracted {len(self._meshes)} meshes")

        # --- Skin ---
        self._skin: Optional[Skin] = None
        if len(skin_joint_node_indices) > 0:
            bind_pose_matrices = self._nodeGlobalMatrices[skin_joint_node_indices]
            joints = np.array(skin_joint_node_indices, dtype=np.int64)
            self._skin = Skin(joints=joints, bind_pose_matrices=bind_pose_matrices)
            print(f"[FBX] Extracted skin with {len(skin_joint_node_indices)} joints")

        # --- Animation ---
        self._animation: Optional[Animation] = None
        num_stacks = scene.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimStack.ClassId))

        if num_stacks > 0:
            anim_stack = scene.GetSrcObject(FbxCriteria.ObjectType(FbxAnimStack.ClassId), 0)
            scene.SetCurrentAnimationStack(anim_stack)

            ts = anim_stack.GetLocalTimeSpan()
            t_start = ts.GetStart().GetSecondDouble()
            t_end = ts.GetStop().GetSecondDouble()
            duration = t_end - t_start
            framerate = _detect_framerate(scene, anim_stack, flat_nodes)

            if duration > 0:
                num_frames = max(int(round(duration * framerate)) + 1, 2)
                timestamps = np.linspace(t_start, t_end, num_frames)
                print(f"[FBX] {num_nodes} nodes, {num_frames} frames, {framerate} fps, {duration:.2f}s")

                local_matrices = _extract_animation(
                    flat_nodes, scene, anim_stack, timestamps, unit_scale
                )
                # FK
                global_matrices = np.zeros((num_frames, num_nodes, 4, 4), dtype=np.float32)
                for ni, node in enumerate(self._nodes):
                    global_matrices[:, ni] = (
                        local_matrices[:, ni] if node.Parent is None
                        else Transform.Multiply(global_matrices[:, node.Parent], local_matrices[:, ni])
                    )

                self._animation = Animation(framerate, local_matrices, global_matrices)
                print(f"[FBX] Done")

        manager.Destroy()

    # --- ModelImporter interface ---

    @property
    def Filename(self) -> str:
        return os.path.splitext(os.path.basename(self._path))[0]

    @property
    def JointNames(self) -> List[str]:
        if self._skin is None:
            return self._nodeNames
        return [self._nodeNames[i] for i in self._skin.Joints]

    @property
    def JointParents(self) -> List[str]:
        if self._skin is None:
            return self._nodeParentNames
        return [self._nodeParentNames[i] for i in self._skin.Joints]

    @property
    def JointMatrices(self) -> np.ndarray:
        if self._skin is None:
            return self._nodeGlobalMatrices
        return self._nodeGlobalMatrices[self._skin.Joints]

    @property
    def Meshes(self) -> List[Mesh]:
        return self._meshes

    @property
    def Skin(self) -> Optional[Skin]:
        return self._skin

    @property
    def SkinnedMesh(self) -> Mesh:
        """Combines all skinned meshes into a single mesh for rendering."""
        vertices_all = []
        normals_all = []
        triangles_all = []
        skin_indices_all = []
        skin_weights_all = []

        vertex_offset = 0
        for mesh in self._meshes:
            if not mesh.HasSkinning:
                continue

            vertices_all.append(mesh.Vertices)
            normals_all.append(mesh.Normals)

            triangles = mesh.Triangles + vertex_offset
            triangles_all.append(triangles)

            skin_indices_all.append(mesh.SkinIndices)
            skin_weights_all.append(mesh.SkinWeights)

            vertex_offset += mesh.Vertices.shape[0]

        if len(vertices_all) == 0:
            return Mesh(
                name=self._path,
                vertices=np.zeros((0, 3), dtype=np.float32),
                normals=np.zeros((0, 3), dtype=np.float32),
                triangles=np.zeros((0,), dtype=np.int64),
                skin_indices=np.zeros((0, 4), dtype=np.int64),
                skin_weights=np.zeros((0, 4), dtype=np.float32),
            )

        return Mesh(
            name=self._path,
            vertices=Tensor.Create(np.concatenate(vertices_all, axis=0)),
            normals=Tensor.Create(np.concatenate(normals_all, axis=0)),
            triangles=Tensor.Create(np.concatenate(triangles_all, axis=0)),
            skin_indices=Tensor.Create(np.concatenate(skin_indices_all, axis=0)),
            skin_weights=Tensor.Create(np.concatenate(skin_weights_all, axis=0)),
        )

    @classmethod
    @lru_cache(maxsize=1)
    def Create(cls, fbx_path: str) -> "FBX":
        return cls(fbx_path)

    def FindParent(self, name, candidates):
        node = next(x for x in self._nodes if x.Name == name)
        pivot = node
        while pivot.Parent is not None:
            pivot = self._nodes[pivot.Parent]
            if pivot.Name in candidates:
                return pivot
        return None

    def LoadMotion(self, names=None, floor=None):
        assert self._animation is not None, f"No animation in {self._path}"
        anim = self._animation

        if names is None:
            hierarchy = Hierarchy(self._nodeNames, self._nodeParentNames)
            frames = anim.GlobalTransformations
        else:
            parents = [self.FindParent(name, names) for name in names]
            parentNames = [p.Name if p is not None else None for p in parents]
            hierarchy = Hierarchy(names, parentNames)
            name_to_index = {node.Name: i for i, node in enumerate(self._nodes)}
            indices = [name_to_index[name] for name in hierarchy.BoneNames if name in name_to_index]
            frames = anim.GlobalTransformations[:, indices]

        if floor is not None:
            floor_node = next((x for x in self._nodes if x.Name == floor), None)
            if floor_node is None:
                print(f"Floor '{floor}' not found. Available: {self._nodeNames}")
            else:
                offset = anim.GlobalTransformations[:, floor_node.Index]
                frames = Transform.TransformationTo(frames, offset.reshape(-1, 1, 4, 4))

        return Motion(
            name=self.Filename,
            hierarchy=hierarchy,
            frames=frames,
            framerate=anim.Framerate,
        )

    def Debug(self):
        print(f"=== FBX: {self._path} ===")
        print(f"Nodes: {len(self._nodes)}")
        print(f"Meshes: {len(self._meshes)}")
        if self._skin:
            print(f"Skin: {len(self.JointNames)} joints")
        if self._animation:
            a = self._animation
            print(f"Animation: {a.Framerate} fps, {a.GlobalTransformations.shape[0]} frames")
