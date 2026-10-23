import random
from typing import Optional

from interfaces import Population, Replacement, Fitnesses


class GenerationalReplacement(Replacement):
    def __init__(
        self, 
        rng: Optional[random.Random] = None
    ):
        super().__init__(rng)
    
    def __call__(
        self,
        population: Population,
        pop_fitnesses: Fitnesses,
        offspring: Population,
        off_fitnesses: Fitnesses,
        maximize: bool = True,
    ) -> tuple[Population, Fitnesses]:
        return offspring, off_fitnesses

class ElitistReplacement(Replacement):
    def __call__(
        self,
        population: Population,
        pop_fitnesses: Fitnesses,
        offspring: Population,
        off_fitnesses: Fitnesses,
        maximize: bool = True,
    ) -> tuple[Population, Fitnesses]:

        combined = list(zip(population, pop_fitnesses)) + list(zip(offspring, off_fitnesses))

        combined.sort(key=lambda item: item[1], reverse=maximize)

        selected = combined[:len(population)]

        next_population = [chromosome for chromosome, _ in selected]
        next_fitnesses = [fitness for _, fitness in selected]

        return next_population, next_fitnesses
