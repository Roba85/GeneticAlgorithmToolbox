import random
from typing import Optional

from itertools import accumulate
from interfaces import Selection, Population, Fitnesses

class TournamentSelection(Selection):
    def __init__(
            self, 
            tournament_size: int = 2, 
            rng: Optional[random.Random] = None
        ):

        super().__init__(rng)

        if tournament_size < 1:
            raise ValueError("tournament_size must be at least 1")

        self.tournament_size = tournament_size

    def __call__(
            self, 
            population: Population, 
            fitnesses: Fitnesses, 
            k: int, 
            maximize: bool = True
        ) -> Population:

        if len(population) != len(fitnesses):
            raise ValueError(
                "Population and fitnesses must have the same length"
            )

        if not population:
            raise ValueError("Population cannot be empty")

        if k < 0:
            raise ValueError("k cannot be negative")

        selected = []

        for _ in range(k):

            # Select tournament_size candidates to choose the best form 
            candidates = [
                self.rng.randrange(len(population))
                for _ in range(self.tournament_size)
            ]

            if maximize:
                winner = max(candidates, key=lambda i: fitnesses[i])
            else:
                winner = min(candidates, key=lambda i: fitnesses[i])

            selected.append(list(population[winner]))

        return selected


class RouletteWheelSelection(Selection):
    def __init__(
            self, 
            rng: Optional[random.Random] = None):
        super().__init__(rng)

    def __call__(
            self, 
            population: Population, 
            fitnesses: Fitnesses,
            k: int, 
            maximize: bool = True
        ) -> Population:

        if len(population) != len(fitnesses):
            raise ValueError(
                "Population and fitnesses must have the same length"
            )

        if not population:
            raise ValueError("Population cannot be empty")

        if k <= 0:
            raise ValueError("k cannot be negative")


        if maximize:
            min_fitness = min(fitnesses)
            weights = [
                fitness - min_fitness
                for fitness in fitnesses
            ]
        else:
            max_fitness = max(fitnesses)
            weights = [
                max_fitness - fitness
                for fitness in fitnesses
            ]

        total_weight = sum(weights)
        cumulative_fitness = list(accumulate(weights))

        selected = []

        for _ in range(k):
            spin = self.rng.random() * total_weight

            for i in range(len(cumulative_fitness)):
                if spin < cumulative_fitness[i]:
                    selected.append(list(population[i]))
                    break

        return selected


