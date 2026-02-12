from abc import ABC, abstractmethod

from ai4animation.AI4Animation import AI4Animation


class Module(ABC):
    Visualize = {}

    def __init__(self, motion):
        self.Motion = motion
        if not type(self) in Module.Visualize:
            Module.Visualize[type(self)] = False
        try:
            if AI4Animation.Standalone is not None:
                self.Standalone()
        except:
            pass

    def GetName(self):
        return __name__

    def Standalone(self):
        pass

    def Callback(self, editor):
        pass

    def GUI(self, editor):
        pass

    def Draw(self, editor):
        pass

    def ToggleVisualize(self):
        Module.Visualize[type(self)] = not Module.Visualize[type(self)]

    @staticmethod
    def GetVisualizeStates(types):
        return [Module.Visualize[type(t)] for t in types]
