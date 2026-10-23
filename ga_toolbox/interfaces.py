"""
interfaces.py - the contract shared by every member of the team.

RULES (read before writing any operator):
  1. A chromosome ("Chromosome") is ALWAYS a plain Python list. The engine never
     looks inside it, so one engine works for binary, integer and float Chromosome.
  2. Operators NEVER modify their inputs in place. Always return new lists.
  3. Each operator is a small class: parameters go in __init__, and the engine
     only calls __call__. A plain function with the same signature also works.
  4. Fitness: the engine handles both directions through the `maximize` flag.
     Operators that compare fitness values must respect it.
  5. For reproducibility, use `self.rng` (a random.Random or the random module)
     instead of calling random.xxx() directly.

"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Tuple, Union

# --------------------------------------------------------------------------
# Type aliases
# --------------------------------------------------------------------------
Gene = Union[int, float]
Chromosome = List[Gene]                      # one chromosome
Population = List[Chromosome]
Fitnesses = List[float]

# User-supplied problem functions (passed to the engine)
FitnessFn = Callable[[Chromosome], float]    # Chromosome -> fitness value
FeasibleFn = Callable[[Chromosome], bool]    # Chromosome -> True if feasible


# --------------------------------------------------------------------------
# Base class giving every operator a shared random source
# --------------------------------------------------------------------------
class Operator(ABC):
    """Common base: stores an optional random generator for reproducibility."""

    def __init__(self, rng: Optional[random.Random] = None):
        # If no rng is given, fall back to the global `random` module.
        self.rng = rng if rng is not None else random


# --------------------------------------------------------------------------
# Representation (Member B)
# --------------------------------------------------------------------------
class Representation(Operator):
    """Knows how to create valid random chromosomes of one gene type.

    Examples: BinaryRepresentation(length), IntegerRepresentation(length, low, high),
    PermutationRepresentation(n), FloatRepresentation(length, low, high).
    """

    @abstractmethod
    def random_Chromosome(self) -> Chromosome:
        """Return ONE new random chromosome."""

    def random_population(self, size: int) -> Population:
        """Return `size` random chromosomes. Subclasses rarely need to override."""
        return [self.random_Chromosome() for _ in range(size)]

    def is_valid(self, Chromosome: Chromosome) -> bool:
        """Check that Chromosome respect the representation's own constraints
        (length, value range, permutation property...). Used by tests and by
        the engine's start-up validation. Default: accept everything."""
        return True


# --------------------------------------------------------------------------
# Selection (Member C)
# --------------------------------------------------------------------------
class Selection(Operator):
    """Chooses parents from the current population."""

    @abstractmethod
    def __call__(
        self,
        population: Population,
        fitnesses: Fitnesses,
        k: int,
        maximize: bool = True,
    ) -> Population:
        """Return `k` selected parents (copies, so later operators cannot
        corrupt the population). The same individual may appear more than once."""


# --------------------------------------------------------------------------
# Crossover (Member C)
# --------------------------------------------------------------------------
class Crossover(Operator):
    """Combines two parents into two children.

    `rate` is the probability that crossover is applied to a pair. When it is
    not applied, the children are plain copies of the parents. This is handled
    here once, so subclasses only implement `_crossover`.
    """

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


# --------------------------------------------------------------------------
# Mutation (Member B)
# --------------------------------------------------------------------------
class Mutation(Operator):
    """Randomly alters a chromosome.

    `rate` meaning is defined by each subclass and must be documented in it:
      - per-gene   (e.g. BitFlip: each bit flips with probability `rate`)
      - per-individual (e.g. Swap/Inversion: the whole chromosome is mutated
        with probability `rate`)
    """

    def __init__(self, rate: float = 0.1, rng: Optional[random.Random] = None):
        super().__init__(rng)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("mutation rate must be in [0, 1]")
        self.rate = rate

    @abstractmethod
    def __call__(self, Chromosome: Chromosome) -> Chromosome:
        """Return a NEW (possibly unchanged) chromosome of the same length."""


# --------------------------------------------------------------------------
# Replacement (Member C)
# --------------------------------------------------------------------------
class Replacement(Operator):
    """Builds the next generation from the old population and the offspring."""

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


# --------------------------------------------------------------------------
# Helper for operator tests (everyone may use it)
# --------------------------------------------------------------------------
def is_permutation(Chromosome: Sequence[int], n: int) -> bool:
    """True if `Chromosome` contains each of 0..n-1 exactly once."""
    return len(Chromosome) == n and sorted(Chromosome) == list(range(n))