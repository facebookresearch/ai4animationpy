import os
import time
from collections import deque
from concurrent.futures import as_completed, ThreadPoolExecutor

import numpy as np
from ai4animation import Utility
from tqdm import tqdm


class DataSampler:
    def __init__(self, dataset, framerate, batch_size, function):
        self.Dataset = dataset
        self.Framerate = framerate
        self.BatchSize = batch_size
        self.Function = function

        self.NumWorkers = Utility.GetNumWorkers()
        print(
            f"Auto-detected num workers={self.NumWorkers} ({os.cpu_count()} CPU cores)"
        )

        print(
            "Generating data sampler for",
            len(self.Dataset),
            "files at",
            self.Framerate,
            "FPS",
        )

        self.Motions = [None] * len(self.Dataset)
        with ThreadPoolExecutor(max_workers=self.NumWorkers) as executor:
            # Submit all tasks with their indices
            future_to_index = {
                executor.submit(self.Dataset.LoadMotion, i): i
                for i in range(len(self.Dataset))
            }

            with tqdm(
                total=len(self.Dataset), desc="Loading motions", unit="file"
            ) as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    self.Motions[index] = future.result()
                    pbar.update(1)

        self.Timestamps = [m.GetTimestamps(self.Framerate) for m in self.Motions]

        self.SampleCount = sum([len(t) for t in self.Timestamps])
        print("Training Samples:", self.SampleCount)

        # self.BatchCount = (self.SampleCount // self.BatchSize) + 1
        # print("Batch Count:", self.BatchCount)

    # Creates batch of lists [[Motion=1, Timestamp=1]]
    def SampleBatchesAcrossMotions(self):
        samples = []
        for i, m in enumerate(self.Motions):
            for t in self.Timestamps[i]:
                samples.append((m, t))
        np.random.shuffle(samples)

        batches = [
            self.DataBatch(self.Function, samples[i : i + self.BatchSize])
            for i in range(0, self.SampleCount, self.BatchSize)
        ]
        print("Training Batches:", len(batches))
        return batches

    # Creates batch of tuples [(Motion=1, [Timestamps=N])]
    def SampleBatchesWithinMotions(self, current_epoch, total_epochs):
        probabilities = [
            len(self.Timestamps[i]) / self.SampleCount for i in range(len(self.Motions))
        ]
        batches = []
        for i in range(0, self.SampleCount, self.BatchSize):
            items = min(self.SampleCount - i, self.BatchSize)
            index = np.random.choice(np.arange(len(self.Motions)), 1, probabilities)[0]
            batches.append(
                self.DataBatch(
                    self.Function,
                    (
                        self.Motions[index],
                        np.random.choice(self.Timestamps[index], items),
                    ),
                )
            )
        return self._Iterator(batches, current_epoch, total_epochs)

    # Creates batch of tuples [(Motion=1, Timestamps=N)]
    def SampleBatchesAsMotions(self):
        batches = []
        for i, m in enumerate(self.Motions):
            batches.append(
                self.DataBatch(self.Function, (self.Motions[i], self.Timestamps[i]))
            )
        return self._Iterator(batches, 1, 1)

    def GetToySample(self):
        return self.DataBatch(
            self.Function, (self.Motions[0], np.array([0.0]))
        ).Retrieve()

    def _Iterator(self, batches, current_epoch=None, total_epochs=None):
        pbar = tqdm(
            total=len(batches),
            desc=f"Epoch {current_epoch}/{total_epochs}",
            unit="batch",
            ncols=140,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        with ThreadPoolExecutor(max_workers=self.NumWorkers) as executor:
            futures_queue = deque()

            batch_idx = 0

            for _ in range(len(batches)):
                # Submit next batches while available
                while len(futures_queue) < self.NumWorkers and batch_idx < len(batches):
                    future = executor.submit(batches[batch_idx].Retrieve)
                    futures_queue.append((batch_idx, future))
                    batch_idx += 1

                # Get the result from the oldest submitted batch
                _, future = futures_queue.popleft()

                yield future.result()

                pbar.update(1)

        pbar.close()

    class DataBatch:
        def __init__(self, fn, args):
            self.Fn = fn
            self.Args = args

        def Retrieve(self):
            return self.Fn(self.Args)
