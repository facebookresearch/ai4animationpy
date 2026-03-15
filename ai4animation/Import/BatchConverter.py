# Copyright (c) Meta Platforms, Inc. and affiliates.
import multiprocessing
import os
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import List, Optional

from ai4animation import Utility
from ai4animation.Animation.Motion import Motion
from tqdm import tqdm


class BatchConverter:
    """Batch processor for converting GLB and FBX files to NPZ motion data"""

    SUPPORTED_EXTENSIONS = (".glb", ".fbx")

    def __init__(
        self,
        input_directory: str,
        output_directory: str = None,
        max_workers: Optional[int] = None,
    ):
        self.input_directory = input_directory
        self.output_directory = (
            output_directory if output_directory else input_directory
        )
        self.max_workers = max_workers if max_workers else multiprocessing.cpu_count()

        if not os.path.exists(input_directory):
            raise FileNotFoundError(f"Input directory not found: {input_directory}")

    def Run(self, bone_names, floor) -> List[str]:
        files = self.FindFiles()
        if not files:
            print(f"No GLB or FBX files found in {self.input_directory}")
            return []

        output_paths = []
        failed_files = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = {
                executor.submit(
                    self.ProcessFile,
                    (
                        file,
                        self.input_directory,
                        self.output_directory,
                        bone_names,
                        floor,
                    ),
                ): file
                for file in files
            }

            with tqdm(
                total=len(files), unit="file", desc="[Converting files]"
            ) as pbar:
                for future in as_completed(tasks):
                    filename, output_path, success, error_msg = future.result()
                    if success:
                        output_paths.append(output_path)
                        pbar.set_postfix({"Process": filename})
                    else:
                        failed_files.append((filename, error_msg))
                        pbar.set_postfix({"Process": f"ERROR: {filename}"})

                    pbar.update(1)

        # Print failures
        if failed_files:
            print(f"\n{len(failed_files)} file(s) failed to process:")
            for filename, error in failed_files:
                print(f"  - {filename}: {error}")

        return output_paths

    def ProcessFile(self, args):
        filename, input_directory, output_directory, bone_names, floor = args
        try:
            filepath = os.path.join(input_directory, filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".glb":
                motion = Motion.LoadFromGLB(filepath, bone_names, floor)
            elif ext == ".fbx":
                motion = Motion.LoadFromFBX(filepath, bone_names, floor)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            # Preserve subfolder structure
            relative_dir = os.path.dirname(filename)
            target_output_dir = os.path.join(output_directory, relative_dir)
            os.makedirs(target_output_dir, exist_ok=True)

            output_path = motion.SaveToNPZ(
                os.path.join(
                    target_output_dir,
                    os.path.splitext(os.path.basename(filename))[0],
                )
            )
            return (filename, output_path, True, None)
        except Exception as e:
            return (filename, None, False, str(e))

    def FindFiles(self) -> List[str]:
        """Find all supported files (GLB and FBX) in the input directory."""
        found_files = []

        for root, _, files in os.walk(self.input_directory):
            for file in files:
                if file.lower().endswith(self.SUPPORTED_EXTENSIONS):
                    # Get relative path from input directory
                    relative_path = os.path.relpath(
                        os.path.join(root, file), self.input_directory
                    )
                    found_files.append(relative_path)
        return sorted(found_files)

    def FindGLBs(self) -> List[str]:
        """Find GLB files only (for backwards compatibility)."""
        return [f for f in self.FindFiles() if f.lower().endswith(".glb")]

    def FindFBXs(self) -> List[str]:
        """Find FBX files only."""
        return [f for f in self.FindFiles() if f.lower().endswith(".fbx")]


def Run(
    input_dir: str, output_dir: str = None, bone_names=None, floor=None
) -> List[str]:
    converter = BatchConverter(
        input_dir, output_dir, max_workers=Utility.GetNumWorkers()
    )
    return converter.Run(bone_names, floor)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch convert GLB and FBX files to NPZ motion data", prog="convert"
    )
    parser.add_argument(
        "--input_dir", required=True, help="Input directory containing GLB or FBX files"
    )
    parser.add_argument(
        "--output_dir", help="Output directory for NPZ files (default: input_dir/NPZ)"
    )
    parser.add_argument(
        "--skeleton",
        choices=["Cranberry"],
        default="Cranberry",
        required=False,
        help="Bone names to store (default: Cranberry)",
    )

    args = parser.parse_args()

    # output directory
    output_dir = (
        args.output_dir if args.output_dir else os.path.join(args.input_dir, "NPZ")
    )
    # os.makedirs(output_dir, exist_ok=True)

    # bone names
    bone_names = None
    floor = None
    if args.skeleton == "Cranberry":
        bone_names = [
            "b_root",
            "b_l_upleg",
            "b_l_leg",
            "b_l_talocrural",
            "b_l_subtalar",
            "b_l_ball",
            "b_r_upleg",
            "b_r_leg",
            "b_r_talocrural",
            "b_r_subtalar",
            "b_r_ball",
            "b_spine0",
            "b_spine1",
            "b_spine2",
            "b_spine3",
            "b_neck0",
            "b_head",
            "b_l_shoulder",
            "p_l_scap",
            "b_l_arm",
            "b_l_forearm",
            "b_l_wrist_twist",
            "b_l_wrist",
            "b_r_shoulder",
            "p_r_scap",
            "b_r_arm",
            "b_r_forearm",
            "b_r_wrist_twist",
            "b_r_wrist",
        ]

    Run(args.input_dir, output_dir, bone_names=bone_names, floor=floor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
