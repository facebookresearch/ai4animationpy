# Copyright (c) Meta Platforms, Inc. and affiliates.
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class Skin:
    def __init__(self, inverse_bind_mats=None, joints=None, bind_pose_matrices=None):
        """Initialize skin data.

        Args:
            inverse_bind_mats: Pre-computed inverse bind matrices (GLB provides these directly)
            joints: Array of joint indices into the node hierarchy
            bind_pose_matrices: Rest-pose global matrices for joints (FBX provides these).
        """
        self.Joints = joints

        if inverse_bind_mats is not None:
            self.Inverse_bind_matrices = inverse_bind_mats
        elif bind_pose_matrices is not None:
            inverse_mats = np.linalg.inv(bind_pose_matrices)
            self.Inverse_bind_matrices = np.transpose(inverse_mats, axes=(0, 2, 1))
        else:
            raise ValueError("Either inverse_bind_mats or bind_pose_matrices must be provided")


class Mesh:
    def __init__(self, name, vertices, normals, triangles, skin_indices, skin_weights):
        self.Name = name
        self.Vertices = vertices
        self.VertexCount = self.Vertices.shape[0]
        self.Normals = normals
        self.Triangles = triangles
        self.TriangleCount = int(triangles.shape[0] / 3)
        self.SkinIndices = skin_indices
        self.SkinWeights = skin_weights
        self.HasSkinning = (
            self.SkinIndices is not None
            and self.SkinWeights is not None
            and len(self.SkinIndices) > 0
            and len(self.SkinWeights) > 0
        )


class ModelImporter(ABC):
    """Abstract base class for 3D model importers (GLB, FBX, etc.)
    """

    @property
    @abstractmethod
    def JointNames(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def JointParents(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def JointMatrices(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def Meshes(self) -> List[Mesh]:
        ...

    @property
    @abstractmethod
    def Skin(self) -> Optional[Skin]:
        ...
