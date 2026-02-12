import multiprocessing
import os
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import List, Optional

from ai4animation import Utility
from ai4animation.Animation.Motion import Motion
from tqdm import tqdm


class BatchConverter:
    """Batch processor for converting GLB files to NPZ motion data"""

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
        glb_files = self.FindGLBs()
        if not glb_files:
            print(f"No GLB files found in {self.input_directory}")
            return []

        output_paths = []
        failed_files = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = {
                executor.submit(
                    self.ProcessGLB,
                    (
                        glb_file,
                        self.input_directory,
                        self.output_directory,
                        bone_names,
                        floor,
                    ),
                ): glb_file
                for glb_file in glb_files
            }

            with tqdm(
                total=len(glb_files), unit="file", desc="[Converting GLB files]"
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

    def ProcessGLB(self, args):
        glb_filename, input_directory, output_directory, bone_names, floor = args
        try:
            glb_path = os.path.join(input_directory, glb_filename)
            motion = Motion.LoadFromGLB(glb_path, bone_names, floor)

            # Preserve subfolder structure
            relative_dir = os.path.dirname(glb_filename)
            target_output_dir = os.path.join(output_directory, relative_dir)
            os.makedirs(target_output_dir, exist_ok=True)

            output_path = motion.SaveToNPZ(
                os.path.join(
                    target_output_dir,
                    os.path.splitext(os.path.basename(glb_filename))[0],
                )
            )
            return (glb_filename, output_path, True, None)
        except Exception as e:
            return (glb_filename, None, False, str(e))

    def FindGLBs(self) -> List[str]:
        glb_files = []

        for root, _, files in os.walk(self.input_directory):
            for file in files:
                if file.lower().endswith(".glb"):
                    # Get relative path from input directory
                    relative_path = os.path.relpath(
                        os.path.join(root, file), self.input_directory
                    )
                    glb_files.append(relative_path)
        return sorted(glb_files)


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
        description="Batch convert GLB files to NPZ motion data", prog="convert"
    )
    parser.add_argument(
        "--input_dir", required=True, help="Input directory containing GLB files"
    )
    parser.add_argument(
        "--output_dir", help="Output directory for NPZ files (default: input_dir/NPZ)"
    )
    parser.add_argument(
        "--skeleton",
        choices=["Trinity3", "Trinity4", "AnimRig"],
        default="Trinity3",
        required=False,
        help="Bone names to store (default: Trinity3)",
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
    if args.skeleton == "Trinity3":
        bone_names = [
            "b_root",
            "b_l_upleg",
            "b_l_leg",
            "b_l_talocrural",
            "b_l_ball",
            "b_r_upleg",
            "b_r_leg",
            "b_r_talocrural",
            "b_r_ball",
            "b_spine0",
            "b_spine1",
            "b_spine2",
            "b_spine3",
            "b_neck0",
            "b_head",
            "b_l_shoulder",
            "b_l_arm",
            "b_l_forearm",
            "b_l_wrist",
            "b_r_shoulder",
            "b_r_arm",
            "b_r_forearm",
            "b_r_wrist",
            "b_l_thumb0",
            "b_l_thumb1",
            "b_l_thumb2",
            "b_l_thumb3",
            "b_l_index1",
            "b_l_index2",
            "b_l_index3",
            "b_l_middle1",
            "b_l_middle2",
            "b_l_middle3",
            "b_l_ring1",
            "b_l_ring2",
            "b_l_ring3",
            "b_l_pinky1",
            "b_l_pinky2",
            "b_l_pinky3",
            "b_r_thumb0",
            "b_r_thumb1",
            "b_r_thumb2",
            "b_r_thumb3",
            "b_r_index1",
            "b_r_index2",
            "b_r_index3",
            "b_r_middle1",
            "b_r_middle2",
            "b_r_middle3",
            "b_r_ring1",
            "b_r_ring2",
            "b_r_ring3",
            "b_r_pinky1",
            "b_r_pinky2",
            "b_r_pinky3",
        ]
        # floor = "_FlooredA3"
    elif args.skeleton == "Trinity4":
        bone_names = [
            "root",
            "l_upleg",
            "l_lowleg",
            "l_talocrural",
            "l_ball",
            "r_upleg",
            "r_lowleg",
            "r_talocrural",
            "r_ball",
            "c_spine0",
            "c_spine1",
            "c_spine2",
            "c_spine3",
            "c_neck",
            "c_head",
            "l_clavicle",
            "l_uparm",
            "l_lowarm",
            "l_wrist",
            "r_clavicle",
            "r_uparm",
            "r_lowarm",
            "r_wrist",
            "l_thumb0",
            "l_thumb1",
            "l_thumb2",
            "l_thumb3",
            "l_index1",
            "l_index2",
            "l_index3",
            "l_middle1",
            "l_middle2",
            "l_middle3",
            "l_ring1",
            "l_ring2",
            "l_ring3",
            "l_pinky1",
            "l_pinky2",
            "l_pinky3",
            "r_thumb0",
            "r_thumb1",
            "r_thumb2",
            "r_thumb3",
            "r_index1",
            "r_index2",
            "r_index3",
            "r_middle1",
            "r_middle2",
            "r_middle3",
            "r_ring1",
            "r_ring2",
            "r_ring3",
            "r_pinky1",
            "r_pinky2",
            "r_pinky3",
        ]
    elif args.skeleton == "AnimRig":
        bone_names = [
            "pelvis_joint",
            "hip_left_joint",
            "knee_left_joint",
            "footAnkle_left_joint",
            "footBall_left_joint",
            "hip_right_joint",
            "knee_right_joint",
            "footAnkle_right_joint",
            "footBall_right_joint",
            "spineLower_joint",
            "spineMiddle_joint",
            "spineUpper_joint",
            "chest_joint",
            "neck_joint",
            "head_joint",
            "clavicle_left_joint",
            "shoulder_left_joint",
            "elbow_left_joint",
            "handWrist_left_joint",
            "clavicle_right_joint",
            "shoulder_right_joint",
            "elbow_right_joint",
            "handWrist_right_joint",
        ]

    Run(args.input_dir, output_dir, bone_names=bone_names, floor=floor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
