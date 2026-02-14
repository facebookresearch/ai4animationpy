# Copyright (c) Meta Platforms, Inc. and affiliates.
from ai4animation import Utility
from ai4animation.Math import Tensor


class TimeSeries:
    def __init__(self, start: float, end: float, samples: int):
        self.Start = start
        self.End = end
        self.Samples = [
            Sample(i, Utility.Normalize(i, 0, samples - 1, start, end))
            for i in range(samples)
        ]

    @property
    def SampleCount(self) -> int:
        return len(self.Samples)

    @property
    def Window(self) -> float:
        return self.End - self.Start

    @property
    def DeltaTime(self) -> float:
        return self.Window / (self.SampleCount - 1)

    @property
    def MaximumFrequency(self) -> float:
        return 0.5 * self.SampleCount / self.Window

    @property
    def FirstSample(self):
        return self.Samples[0]

    @property
    def LastSample(self):
        return self.Samples[-1]

    @property
    def Timestamps(self):
        return Tensor.LinSpace(self.Start, self.End, self.SampleCount)

    def GetSample(self, timestamp):
        if timestamp < self.Start or timestamp > self.End:
            print(
                "Given timestamp was "
                + str(timestamp)
                + " but must be within "
                + str(self.Start)
                + " and "
                + str(self.End)
                + "."
            )
        timestamp = Tensor.Clamp(timestamp, self.Start, self.End)
        return self.Samples[
            int(
                Utility.Normalize(
                    timestamp, self.Start, self.End, 0, self.SampleCount - 1
                )
            )
        ]

    # def GetSampleIndices(self, startIndex, endIndex):
    #     return Tensor.Arange(startIndex, endIndex, 1)

    def SimulateTimestamps(self, timestamp=0.0):
        return Tensor.LinSpace(
            timestamp + self.Start, timestamp + self.End, self.SampleCount
        )

    def Draw(self):
        pass

    def GUI(self):
        pass


class Sample:
    def __init__(self, index, timestamp):
        self.Index = index
        self.Timestamp = timestamp
