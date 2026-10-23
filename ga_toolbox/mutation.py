from __future__ import annotations
from interfaces import Mutation, Chromosome

class BitFlipMutation(Mutation):
    def __call__(self, chromosome: Chromosome) -> Chromosome:
        mutated = list(chromosome)

        for i in range(len(mutated)):
            if self.rng.random() < self.rate:
                mutated[i] = 1 - mutated[i]

        return mutated

class IntegerMutation(Mutation):
    def __init__(self, rate=0.1, low=0, high=10, rng=None):
        super().__init__(rate, rng)
        self.low = low
        self.high = high

    def __call__(self, chromosome):
        mutated = list(chromosome)

        for i in range(len(mutated)):
            if self.rng.random() < self.rate:
                mutated[i] = self.rng.randint(self.low, self.high)

        return mutated


class UniformFloatMutation(Mutation):
    def __init__(self, rate=0.1, low=0.0, high=1.0, rng=None):
        super().__init__(rate, rng)
        self.low = low
        self.high = high

    def __call__(self, chromosome):
        mutated = list(chromosome)

        for i in range(len(mutated)):
            if self.rng.random() < self.rate:
                x = mutated[i]

                if isinstance(self.low, list):
                    lb = self.low[i]
                else:
                    lb = self.low

                if isinstance(self.high, list):
                    ub = self.high[i]
                else:
                    ub = self.high

                delta_l = x - lb
                delta_u = ub - x

                r1 = self.rng.uniform(0, 1)

                if r1 <= 0.5:
                    if delta_l > 0:
                        r2 = self.rng.uniform(0, delta_l)
                        mutated[i] = x - r2
                else:
                    if delta_u > 0:
                        r2 = self.rng.uniform(0, delta_u)
                        mutated[i] = x + r2

        return mutated


class InversionMutation(Mutation):
    def __call__(self, chromosome):
        mutated = list(chromosome)

        if self.rng.random() < self.rate:
            if len(mutated) > 2:
                indices = self.rng.sample(range(len(mutated)), 2)
                indices.sort()  # Put them in order
                i = indices[0]
                j = indices[1]

                sub_segment = mutated[i: j + 1]
                sub_segment.reverse()
                mutated[i: j + 1] = sub_segment

        return mutated

class SwapMutation(Mutation):
    def __call__(self, chromosome):
        mutated = list(chromosome)

        if self.rng.random() < self.rate:
            if len(mutated) > 1:
                indices = self.rng.sample(range(len(mutated)), 2)
                i = indices[0]
                j = indices[1]

                temp = mutated[i]
                mutated[i] = mutated[j]
                mutated[j] = temp

        return mutated