import importlib.util
import sys

# buck run fbcode//ai4animationpy:demos -- DemoName
# buck run fbcode//ai4animationpy:demos -- list

DEMOS = {
    "Empty": "Demos.Empty.Program",
    "Actor": "Demos.Actor.Program",
    "ECS": "Demos.ECS.Program",
    "GLB_Loading": "Demos.GLBLoading.Program",
    "IK": "Demos.InverseKinematics.Program",
    "MotionEditor": "Demos.MotionEditor.Program",
    "Autoencoder": "Demos.AI.Autoencoder.Program",
    "SequencePrediction": "Demos.AI.SequencePrediction.Program",
    "MotionTracking": "Demos.MotionTracking.Program",
    "Locomotion": "Demos.Locomotion.Program",
    "3PT": "Demos.ThreePointTracking.Program",
}


def list_demos() -> None:
    print("Available demos:")
    for demo_name in sorted(DEMOS.keys()):
        print(f"  - {demo_name}")


def Run(demo_name: str) -> int:
    if demo_name not in DEMOS:
        print(f"Error: Unknown demo '{demo_name}'")
        print()
        list_demos()
        return 1

    module_path = DEMOS[demo_name]
    print(f"Running demo: {demo_name}")
    print(f"Module: {module_path}")
    print("-" * 60)

    try:
        # Import and run the module
        module = importlib.import_module(module_path)

        if module is None or not hasattr(module, "main"):
            print(f"Error: Module '{module_path}' does not have a 'main' function")
            return 1

        module.main()

        return 0
    except Exception as e:
        print(f"Error running demo '{demo_name}': {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: RunDemo <demo_name>")
        print()
        list_demos()
        return 1

    demo_name = sys.argv[1]

    if demo_name in ("--list", "-l", "list"):
        list_demos()
        return 0

    return Run(demo_name)


if __name__ == "__main__":
    sys.exit(main())
