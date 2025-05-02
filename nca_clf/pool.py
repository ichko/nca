from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Sample:
    batch: np.ndarray
    index: np.ndarray


class NCAPool:
    def __init__(self, data_gen, pool_size):
        self.pool = []
        self.data_gen = data_gen
        for i in range(pool_size):
            batch = next(data_gen)
            self.pool.append(batch)

    def sample(self):
        index = np.random.randint(0, len(self.pool))
        return self.pool[index]

    def update(self, sample, new_seeds):
        self.pool[sample.index] = new_seeds
