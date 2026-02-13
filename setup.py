# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ai4animation",
    version="1.0.0",
    description="AI4Animation Python Framework - Neural network-based character animation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Paul Starke, Sebastian Starke",
    packages=find_packages() + ["AnimRig", "Trinity", "Manifold"],  # @fb-only
    package_dir={
        "ai4animation": "ai4animation",
        "AnimRig": "Assets/AnimRig",  # @fb-only
        "Trinity": "Assets/Trinity",  # @fb-only
        "Manifold": "Assets/Manifold",  # @fb-only
    },
    entry_points={
        "console_scripts": [
            "convert=ai4animation.Import.BatchConverter:main",
        ],
    },
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "onnxruntime-gpu>=1.18.0",
        "raylib>=4.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.10.3",
        "scikit-learn>=1.7.1",
        "einops>=0.8.1",
        "pygltflib==1.16.5",
        "pyscreenrec==0.6",
        "tqdm",
        "pyyaml",
        "onnx==1.19.1",
    ],
)
