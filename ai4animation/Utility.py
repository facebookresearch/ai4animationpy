import importlib.util
import os
import pathlib
import random
import secrets
import string
import sys

import numpy as np
import torch


def ToBytes(value):
    return bytes("None" if value is None else value, "utf-8")


def FromBytes(value):
    return value.decode("utf-8")


def Opacity(color, value):
    return (color[0], color[1], color[2], int(value * color[3]))


def Normalize(value, valueMin, valueMax, resultMin, resultMax):
    if valueMax - valueMin != 0.0:
        return (value - valueMin) / (valueMax - valueMin) * (
            resultMax - resultMin
        ) + resultMin
    else:
        print("Not possible to perform normalization.")
        return value


def Ratio(current, start, end):
    if start == end:
        return 1.0
    return Clamp((current - start) / (end - start), 0.0, 1.0)


def Clamp(value, min, max):
    if value < min:
        return min
    if value > max:
        return max
    return value


def ClampArray(values, min, max):
    for i in range(len(values)):
        values[i] = Clamp(values[i], min, max)
    return values


def SymmetryIndices(joint_names):
    def TryAssign(value: str, bone: int):
        if value in name_to_idx:
            symmetry[bone] = name_to_idx[value]
            return True
        else:
            return False

    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    symmetry = list(range(len(joint_names)))
    for i, boneName in enumerate(joint_names):
        if boneName is None:
            continue
        if "_l_" in boneName:
            if TryAssign(boneName.replace("_l_", "_r_"), i):
                continue

        if "_r_" in boneName:
            if TryAssign(boneName.replace("_r_", "_l_"), i):
                continue

        if "_left_" in boneName:
            if TryAssign(boneName.replace("_left_", "_right_"), i):
                continue

        if "_right_" in boneName:
            if TryAssign(boneName.replace("_right_", "_left_"), i):
                continue

        symmetry[i] = i
    # for i, boneName in enumerate(joint_names):
    #    print(boneName, name_to_idx[boneName], joint_names[symmetry[i]])
    return symmetry


def gensym(length=32, prefix="gensym_"):
    """
    generates a fairly unique symbol, used to make a module name,
    used as a helper function for load_module

    :return: generated symbol
    """
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    symbol = "".join([secrets.choice(alphabet) for i in range(length)])

    return prefix + symbol


def LoadModule(source, module_name=None):
    """
    reads file source and loads it as a module

    :param source: file to load
    :param module_name: name of module to register in sys.modules
    :return: loaded module
    """

    if module_name is None:
        module_name = gensym()

    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def MakeDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)


def GetDirectory(file):
    return str(pathlib.Path(file).parent)


def SetSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def SaveONNX(
    path,
    model,
    input_size,
    input_names,
    output_names,
    dynamic_shapes=None,
    dynamo=False,
):
    model = model.cpu() if torch.cuda.is_available() else model
    if dynamo:
        torch.onnx.export(
            model,
            input_size,
            path,
            opset_version=18,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            external_data=False,
            dynamo=dynamo,
        )
    else:
        torch.onnx.export(
            model,
            input_size,
            path,
            opset_version=18,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_shapes,
            external_data=False,
            dynamo=dynamo,
        )
    model = model.cuda() if torch.cuda.is_available() else model


def GetNumWorkers():
    return os.cpu_count() // 4


# @staticmethod
# def RegisterVariable(instance, id, fn):
#     try:
#         return getattr(instance, id) #Let's see if this gets too expensive
#     except:
#         value = fn()
#         setattr(instance, id, value)
#         return value

# @staticmethod
# def HasVariable(instance, id):
#     return hasattr(instance, id)

# print("Absolute Time:", self.Time, rl.GetTime()) #Get time is the absolute system timestamp
# print("Mismatch:", rl.GetTime() - self.Time) #The mismatch stays almost constant
