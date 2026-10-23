from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Tuple, Union

# Type aliases
Gene = Union[int, float]
Chromosome = List[Gene]
Population = List[Chromosome]
Fitnesses = List[float]

# User-supplied problem functions
FitnessFn = Callable[[Chromosome], float]
FeasibleFn = Callable[[Chromosome], bool]


class Operator(ABC):
    """Common base: stores an optional random generator for reproducibility."""

    def __init__(self, rng: Optional[random.Random] = None):
        # If no rng is given, fall back to the global `random` module.
        self.rng = rng if rng is not None else random


class Representation(Operator):

    @abstractmethod
    def random_Chromosome(self) -> Chromosome:
        """Return ONE new random chromosome."""

    def random_population(self, size: int) -> Population:
        return [self.random_Chromosome() for _ in range(size)]

    # Overridden by children classes
    def is_valid(self, chromosome: Chromosome) -> bool:
        return True

class Selection(Operator):

    @abstractmethod
    def __call__(
        self,
        population: Population,
        fitnesses: Fitnesses,
        k: int,
        maximize: bool = True,
    ) -> Population:
        """Returns selected individuals (copies)"""


class Crossover(Operator):

    def __init__(self, rate: float = 0.9, rng: Optional[random.Random] = None):
        super().__init__(rng)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("crossover rate must be in [0, 1]")
        self.rate = rate

    def __call__(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        if self.rng.random() < self.rate:
            return self._crossover(parent1, parent2)
        return list(parent1), list(parent2)

    @abstractmethod
    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Always perform the crossover and return two NEW children."""


class Mutation(Operator):

    def __init__(self, rate: float = 0.1, rng: Optional[random.Random] = None):
        super().__init__(rng)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("mutation rate must be in [0, 1]")
        self.rate = rate

    @abstractmethod
    def __call__(self, Chromosome: Chromosome) -> Chromosome:
        """Return a NEW (possibly unchanged) chromosome of the same length."""

class Replacement(Operator):

    @abstractmethod
    def __call__(
        self,
        population: Population,
        pop_fitnesses: Fitnesses,
        offspring: Population,
        off_fitnesses: Fitnesses,
        maximize: bool = True,
    ) -> Tuple[Population, Fitnesses]:
        """Return (new_population, new_fitnesses).

        The new population must have the same size as `population`, and
        new_fitnesses[i] must belong to new_population[i].
        """


def is_permutation(Chromosome: Sequence[int], n: int) -> bool:
    """True if `Chromosome` contains each of 0..n-1 exactly once."""
    return len(Chromosome) == n and sorted(Chromosome) == list(range(n))