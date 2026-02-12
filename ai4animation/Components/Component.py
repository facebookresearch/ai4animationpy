from abc import ABC, abstractmethod

from ai4animation import AI4Animation


class Component(ABC):
    def __init__(self, entity, params):
        self.Entity = entity
        self.Start(params)
        if AI4Animation.AI4Animation.Standalone is not None:
            self.Standalone()

    def Start(self, params):
        pass

    def Standalone(self):
        pass

    def Update(self):
        pass

    def Standalone(self):
        pass

    def Draw(self):
        pass

    def GUI(self):
        pass
