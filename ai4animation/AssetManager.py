# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Asset path resolver for locating project resource files."""

import os
import sys
import types
from pathlib import Path


class AssetManager:
    Root = None

    @classmethod
    def _register_assets_package(cls):
        """Expose the Assets directory as the `ai4animation.Assets` subpackage.

        Instead of injecting asset folders into the global ``sys.path`` (which
        pollutes the top-level namespace and makes ``import TrinityV4`` work
        from anywhere), we attach the asset directories to the ``__path__`` of
        a virtual ``ai4animation.Assets`` package. This makes imports such as
        ``from ai4animation.Assets import TrinityV4`` work via the regular
        package machinery, scoped to ``ai4animation.Assets`` only.
        """
        if cls.Root is None:
            return

        assets_path = str(cls.Root)
        if not os.path.exists(assets_path):
            return

        # Collect the assets root plus its immediate subdirectories so that
        # files nested one level deep (e.g. Definitions/TrinityV4.py) are
        # importable as ``ai4animation.Assets.TrinityV4``.
        paths = [assets_path]
        for item in os.listdir(assets_path):
            item_path = os.path.join(assets_path, item)
            if os.path.isdir(item_path) and not item.startswith((".", "__")):
                paths.append(item_path)

        pkg_name = "ai4animation.Assets"
        mod = sys.modules.get(pkg_name)
        if mod is None:
            mod = types.ModuleType(pkg_name)
            mod.__package__ = pkg_name
            mod.__path__ = []
            sys.modules[pkg_name] = mod
            parent = sys.modules.get("ai4animation")
            if parent is not None:
                parent.Assets = mod

        existing = list(getattr(mod, "__path__", []))
        for p in paths:
            if p not in existing:
                existing.append(p)
        mod.__path__ = existing

    @classmethod
    def SetRoot(cls, path):
        cls.Root = Path(path).resolve()
        cls._register_assets_package()

    @classmethod
    def GetPath(cls, relative_path):
        """
        Get the full path to an asset file.

        Args:
            relative_path: Either an absolute path or a path relative to assets root

        Returns:
            Absolute path to the asset file as a string

        Example:
            AssetManager.GetPath("MyFolder/SubFolder")
            AssetManager.GetPath("v3.glb")
            AssetManager.GetPath("Assets/v3.glb")
            AssetManager.GetPath("C:/absolute/path/model.glb")
        """

        # Handle list of paths
        if isinstance(relative_path, list):
            return [cls.GetPath(p) for p in relative_path]

        # If already absolute path, return it
        if os.path.isabs(relative_path):
            return relative_path

        if cls.Root is None:
            # Default: AI4AnimationPy/Assets
            module_dir = Path(__file__).resolve().parent
            cls.Root = module_dir.parent / "Assets"
            cls._register_assets_package()

        # Try as asset name first (e.g., "v3.glb")
        asset_path = cls.Root / relative_path
        if asset_path.is_file() or asset_path.is_dir():
            return str(asset_path)

        # Try stripping "Assets/" prefix if present
        if relative_path.startswith(("Assets/", "Assets\\")):
            stripped_path = (
                relative_path.split(os.sep, 1)[1]
                if os.sep in relative_path
                else relative_path.split("/", 1)[1]
            )
            asset_path = cls.Root / stripped_path
            if asset_path.is_file() or asset_path.is_dir():
                return str(asset_path)

        path = str(cls.Root / relative_path)
        if not os.path.isfile(path) and not os.path.isdir(path):
            raise FileNotFoundError(f"Asset path or directory not found: {path}")
        return path

    @classmethod
    def Reset(cls):
        cls.Root = None

    @classmethod
    def GetRoot(cls):
        if cls.Root is None:
            # Trigger auto-detection
            cls.GetPath("")
        return cls.Root


AssetManager.SetRoot(Path(__file__).resolve().parent.parent / "Assets")


def __getattr__(name):
    """Delegate module-level attribute access to the AssetManager class.

    This ensures ai4animation.AssetManager.GetPath(...) works whether
    AssetManager resolves to the module or the class.
    """
    return getattr(AssetManager, name)
