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

# def main():
#     population = [[1], [2], [3]]
#     pop_fitnesses = [10, 20, 30]

#     offspring = [[4], [5], [6]]
#     off_fitnesses = [25, 5, 35]

#     # Generational replacement
#     replacement = GenerationalReplacement()
#     print(replacement(population, pop_fitnesses, offspring, off_fitnesses))
#     # ([[4], [5], [6]], [25, 5, 35])

#     # Elitist replacement (maximize)
#     replacement = ElitistReplacement()
#     print(replacement(population, pop_fitnesses, offspring, off_fitnesses))
#     # ([[6], [3], [4]], [35, 30, 25])

# main()