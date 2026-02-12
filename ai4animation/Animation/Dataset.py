import os

from ai4animation.Animation.Motion import Motion


class Dataset:
    def __init__(self, directory, modules, max_files=None):
        self.Directory = directory
        self.Modules = modules
        self.Pool = []
        # Find all NPZ files in the directory and subdirectories
        for root, _dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".npz"):
                    if max_files is None or len(self.Pool) < max_files:
                        self.Pool.append(os.path.join(root, file))
        self.Pool.sort()
        self.Filter()

    def GetName(self, file):
        return os.path.splitext(os.path.basename(file))[0]

    def GetMotionIndex(self, motion):
        if motion.Name in self.NameToIndex:
            return self.NameToIndex[motion.Name]
        else:
            return None

    def LoadMotion(self, index):
        motion = Motion.LoadFromNPZ(self.Files[index])
        for module in self.Modules:
            motion.AddModule(module)
        return motion

    def Filter(self, id=None):
        self.Files = []
        for item in self.Pool:
            if id is None or id in self.GetName(item):
                self.Files.append(item)
        self.NameToIndex = {}
        for i in range(len(self.Files)):
            self.NameToIndex[self.GetName(self.Files[i])] = i

    def __len__(self):
        return len(self.Files)
