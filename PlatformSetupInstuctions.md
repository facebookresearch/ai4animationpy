## Usage
To setup an environment, run the conda setup for your platform below

### Windows
```
conda create -n AI4AnimationPY python=3.12
conda activate AI4AnimationPY
pip install msvc-runtime==14.40.33807
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install nvidia-cudnn-cu12==9.3.0.75 nvidia-cuda-runtime-cu12==12.5.82 nvidia-cufft-cu12==11.2.3.61
pip install onnxruntime-gpu==1.19.0
pip install -e ${HOME}/fbsource/fbcode/ai4animationpy --use-pep517
```

## Unsupported Platforms
The platforms below are not actively supported by the core development team, not guaranteed to be stable and may not work with the latest version of the code.
They are provided as a convenience for users who would like to try to run the code on these platforms.

### Linux
```
conda create -y -n AI4AnimationPY python=3.12 pip
conda activate AI4AnimationPY
pip install torch torchvision torchaudio onnx raylib numpy scipy matplotlib scikit-learn einops pygltflib pyscreenrec tqdm pyyaml ipython
pip install onnxruntime-gpu
pip install -e ${HOME}/fbsource/fbcode/ai4animationpy --no-dependencies
```

### OSX
```
conda create -y -n AI4AnimationPY python=3.12 pip
conda activate AI4AnimationPY
pip install torch torchvision torchaudio onnx raylib numpy scipy matplotlib scikit-learn einops pygltflib pyscreenrec tqdm pyyaml ipython
pip install onnxruntime
pip install -e ${HOME}/fbsource/fbcode/ai4animationpy --no-dependencies
```
