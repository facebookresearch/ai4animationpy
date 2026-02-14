# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import onnxruntime as ort
import torch

from ai4animation.Math import *
from functools import lru_cache

_LRU_CACHE_MAX_SIZE: int = 4


class ONNXNetwork:
    def __init__(self, path):
        self.Path = path
        # TODO:  model needs a tensorrt optimization for GPU
        providers = ["CUDAExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # sess_options.set_deterministic_compute(True)

        self.Session = ort.InferenceSession(
            path, sess_options=sess_options, providers=providers
        )
        print(torch.version.cuda)
        print(ort.get_device())

        self.Inputs = {}
        self.Outputs = {}
        self.InputShapes = {}
        self.OutputShapes = {}

        self.InputsOrt = {}
        self.reset_inputs()
        self.reset_outputs()

    @classmethod
    @lru_cache(maxsize=_LRU_CACHE_MAX_SIZE)
    def Create(cls, model_path: str) -> "ONNXNetwork":
        return cls(model_path)

    def reset_inputs(self) -> None:
        from ai4animation.AI.FeedTensor import FeedTensor

        for t in self.Session.get_inputs():
            has_dynamic_dims = any(isinstance(dim, str) for dim in t.shape)
            if has_dynamic_dims:
                # print("Dynamic Input Tensor:", t.name, t.shape)
                self.InputShapes[t.name] = self._create_default_shape(t.shape)
            else:
                # print("Static Input Tensor:", t.name, t.shape)
                self.InputShapes[t.name] = t.shape
            self.Inputs[t.name] = FeedTensor(t.name, self.InputShapes[t.name])
            self.InputsOrt[t.name] = ort.OrtValue.ortvalue_from_numpy(
                self.Inputs[t.name].GetValues()
            )

    def reset_outputs(self) -> None:
        from ai4animation.AI.ReadTensor import ReadTensor

        for t in self.Session.get_outputs():
            has_dynamic_dims = any(isinstance(dim, str) for dim in t.shape)

            if has_dynamic_dims:
                print("Dynamic Output Tensor:", t.name, t.shape)
                self.OutputShapes[t.name] = self._create_default_shape(t.shape)
            else:
                print("Static Output Tensor:", t.name, t.shape)
                self.OutputShapes[t.name] = t.shape
            self.Outputs[t.name] = ReadTensor(t.name, self.OutputShapes[t.name])

    def _create_default_shape(self, shape_template):
        default_shape = []
        for dim in shape_template:
            if isinstance(dim, str):
                default_shape.append(1)  # Default size for dynamic dimensions
            else:
                default_shape.append(dim)
        return tuple(default_shape)

    def Run(self):
        for input in self.Inputs.values():
            assert input.Pivot == input.GetValues().shape[-1], (
                f"Input {input.Name} in network {self.Path} has not received"
                f" all features ({input.Pivot} / {input.GetValues().shape[-1]})"
            )

        self.Reset()

        prediction = self.Session.run(
            list(self.Outputs.keys()),
            # {k: v.Values for k, v in self.Inputs.items()}
            self.InputsOrt,
        )

        for key in self.Outputs.keys():
            index = list(self.Outputs).index(key)
            self.Outputs[key].SetData(prediction[index].flatten())

    def Reset(self):
        for value in self.Inputs.values():
            value.Reset()

        for value in self.Outputs.values():
            value.Reset()
