from typing import Optional
import random

from interfaces import Crossover, Chromosome


class MultiPointCrossover(Crossover):

    def __init__(
        self,
        n_points: int = 2,
        rate: float = 0.9,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(rate, rng)

        if n_points < 1:
            raise ValueError("n_points must be at least 1")

        self.n_points = n_points

    def _crossover(
        self,
        parent1: Chromosome,
        parent2: Chromosome,
    ) -> tuple[Chromosome, Chromosome]:

        if len(parent1) != len(parent2):
            raise ValueError("Parents must have equal lengths")

        n = len(parent1)

        if self.n_points > n - 1:
            raise ValueError(
                "n_points must be less than chromosome length"
            )

        cuts = sorted(
            self.rng.sample(range(1, n), self.n_points)
        )

        boundaries = [0] + cuts + [n]

        child1 = []
        child2 = []

        for segment in range(len(boundaries) - 1):
            start = boundaries[segment]
            end = boundaries[segment + 1]

            if segment % 2 == 0:
                child1.extend(parent1[start:end])
                child2.extend(parent2[start:end])
            else:
                child1.extend(parent2[start:end])
                child2.extend(parent1[start:end])

        return child1, child2


class OrderCrossover(Crossover):

    def _crossover(
        self,
        parent1: Chromosome,
        parent2: Chromosome,
    ) -> tuple[Chromosome, Chromosome]:

        if len(parent1) != len(parent2):
            raise ValueError("Parents must have equal lengths")

        if sorted(parent1) != sorted(parent2):
            raise ValueError(
                "Parents must contain the same genes"
            )

        n = len(parent1)

        if n < 2:
            return list(parent1), list(parent2)

        start, end = sorted(self.rng.sample(range(n), 2))
        end += 1  

        def make_child(
            segment_parent: Chromosome,
            order_parent: Chromosome,
        ) -> Chromosome:

            child = [None] * n

            child[start:end] = segment_parent[start:end]

            used = set(segment_parent[start:end])

            genes = order_parent[end:] + order_parent[:end]
            genes = [gene for gene in genes if gene not in used]

            positions = list(range(end, n)) + list(range(start))

            for position, gene in zip(positions, genes):
                child[position] = gene

            return child

        child1 = make_child(parent1, parent2)
        child2 = make_child(parent2, parent1)

        return child1, child2


