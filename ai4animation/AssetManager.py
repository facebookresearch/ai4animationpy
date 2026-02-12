import os
import sys
from pathlib import Path


class AssetManager:
    Root = None

    @classmethod
    def _add_assets_to_path(cls):
        if cls.Root is not None:
            assets_path = str(cls.Root)
            if assets_path not in sys.path:
                sys.path.insert(0, assets_path)

            # Automatically add all subdirectories in Assets to path
            if os.path.exists(assets_path):
                for item in os.listdir(assets_path):
                    item_path = os.path.join(assets_path, item)
                    # Add directories (excluding __pycache__ and hidden folders)
                    if os.path.isdir(item_path) and not item.startswith((".", "__")):
                        if item_path not in sys.path:
                            sys.path.insert(0, item_path)

    @classmethod
    def SetRoot(cls, path):
        cls.Root = Path(path).resolve()
        cls._add_assets_to_path()

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
        # If already absolute path, return it
        if os.path.isabs(relative_path):
            return relative_path

        if cls.Root is None:
            # Default: AI4AnimationPy/Assets
            module_dir = Path(__file__).resolve().parent
            cls.Root = module_dir.parent / "Assets"
            cls._add_assets_to_path()

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
        if not os.path.isfile(path) or not os.path.dir(path):
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
