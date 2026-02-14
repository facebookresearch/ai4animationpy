# Copyright (c) Meta Platforms, Inc. and affiliates.
__version__ = "1.0.0"

# Subpackages (import modules/packages, not classes yet to avoid circular imports)
# Core utility modules (no dependencies on other package modules)
from . import AI, Animation, Components, IK, Import, Math, Time, Utility

# AI modules
from .AI import DataSampler, FeedTensor, ONNXNetwork, Plotting, ReadTensor, Stats
from .AI.Networks import (
    Autoencoder,
    CodebookMatching,
    CodebookMatchingRegularized,
    CodebookMatchingRegularizedZ,
    ConditionalFlow,
    Flow,
    MLP,
)
from .AI.Optimizers.AdamWR import AdamW, CyclicScheduler

# Core classes (imported after subpackages to avoid circular dependencies)
from .AI4Animation import AI4Animation
from .Animation.ContactModule import ContactModule
from .Animation.Dataset import Dataset
from .Animation.GuidanceModule import GuidanceModule
from .Animation.Module import Module

# Animation classes
from .Animation.Motion import Hierarchy, Motion
from .Animation.MotionModule import MotionModule
from .Animation.RootModule import RootModule
from .Animation.TimeSeries import TimeSeries
from .Animation.TrackingModule import TrackingModule
from .AssetManager import AssetManager

# Component classes
from .Components.Actor import Actor
from .Components.Component import Component
from .Components.MeshRenderer import MeshRenderer
from .Components.MotionEditor import MotionEditor
from .Entity import Entity

# IK classes
from .IK.FABRIK import FABRIK

# Import classes
from .Import.GLBImporter import GLB

# Math classes - re-export for convenience
from .Math import Quaternion, Rotation, Tensor, Transform, Vector3
from .Profiler import Profiler
from .Scene import Scene

# Define what gets exported with "from ai4animation import *"
__all__ = [
    # Version
    "__version__",
    # Core
    "AI4Animation",
    "Scene",
    "Entity",
    "Dataset",
    "ONNXNetwork",
    "FeedTensor",
    "ReadTensor",
    "Time",
    "Utility",
    "Profiler",
    "AssetManager",
    # Subpackages
    "Math",
    "Animation",
    "AI",
    "IK",
    "Import",
    "Components",
    # Animation
    "Motion",
    "Hierarchy",
    "TimeSeries",
    "Module",
    "RootModule",
    "MotionModule",
    "ContactModule",
    "GuidanceModule",
    "TrackingModule",
    # Components
    "Component",
    "Actor",
    "MotionEditor",
    "Dataset",
    # IK
    "FABRIK",
    # Import
    "GLB",
    # AI
    "DataSampler",
    "Stats",
    "Plotting",
    "MLP",
    "Autoencoder",
    "Flow",
    "ConditionalFlow",
    "CodebookMatching",
    "CodebookMatchingRegularized",
    "CodebookMatchingRegularizedZ",
    "AdamW",
    "CyclicScheduler",
    # Math
    "Tensor",
    "Transform",
    "Vector3",
    "Rotation",
    "Quaternion",
]
