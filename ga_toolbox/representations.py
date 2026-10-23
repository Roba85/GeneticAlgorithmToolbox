from __future__ import annotations
from interfaces import Representation, Chromosome, is_permutation
from typing import cast, Sequence


class BinaryRepresentation(Representation):

    def __init__(self, length: int, rng = None):
        super().__init__(rng)
        if length <= 0:
            raise ValueError("Invalid chromosome length!")
        self.length = length

    def random_Chromosome(self) -> Chromosome:
        chromosome = []
        for i in range (self.length):
            bit = self.rng.choice([0, 1])
            chromosome.append(bit)
        return chromosome

    def is_valid(self, chromosome: Chromosome) -> bool:
        if len(chromosome) != self.length:
            return False

        for gene in chromosome:
            if gene not in [0, 1]:
                return False
        return True


class IntegerRepresentation(Representation):

    def __init__(self, length: int, low: int, high: int, rng = None):
        super().__init__(rng)
        if length <= 0:
            raise ValueError("Invalid chromosome length!")
        if low > high:
            raise ValueError("Lower bound can't be greater than upper bound!")
        self.length = length
        self.low = low
        self.high = high

    def random_Chromosome(self) -> Chromosome:
        chromosome = []
        for i in range (self.length):
            val = self.rng.randint(self.low, self.high)
            chromosome.append(val)
        return chromosome

    def is_valid(self, chromosome):
        if len(chromosome) != self.length:
            return False
        for gene in chromosome:
            if not isinstance(gene, int):
                return False
            if gene < self.low or gene > self.high:
                return False
        return True


class FloatRepresentation(Representation):

    def __init__(self, length: int, low: float, high: float, rng = None):
        super().__init__(rng)
        if length <= 0:
            raise ValueError("Chromosome length must be greater than 0")
        if low > high:
            raise ValueError("Low bound cannot be greater than high bound")
        self.length = length
        self.low = low
        self.high = high

    def random_Chromosome(self):
        chromosome = []
        for i in range(self.length):
            val = self.rng.uniform(self.low, self.high)
            chromosome.append(val)
        return chromosome

    def is_valid(self, chromosome):
        if len(chromosome) != self.length:
            return False
        for gene in chromosome:
            if not isinstance(gene, (int, float)):
                return False
            if gene < self.low or gene > self.high:
                return False
        return True


class PermutationRepresentation(Representation):
    def __init__(self, n, rng=None):
        super().__init__(rng)
        self.n = n

    def random_Chromosome(self):
        chromosome = []
        for i in range(self.n):
            chromosome.append(i)

        self.rng.shuffle(chromosome)
        return chromosome

    def is_valid(self, chromosome):
        for i in range(len(chromosome)):
            if not isinstance(chromosome[i], int):
                return False
        return is_permutation(cast(Sequence[int], chromosome), self.n)
