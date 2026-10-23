from selection import TournamentSelection, RouletteWheelSelection
from replacement import ElitistReplacement
from crossover import OrderCrossover
from mutation import BitFlipMutation
from representations import BinaryRepresentation
from engine import GAConfig, GeneticAlgorithm
from interfaces import FitnessFn, FeasibleFn

import random

# Will be completed after all is finished

def default_ga_binary(
        feasibilityFn: FeasibleFn, 
        fitnessFn: FitnessFn,
        rng: random.Random,
        population: int = 100, 
        gen: int = 200, 
        maximize: bool = True, 
        p_crossover: float = 0.7,
        p_mutation: float = 0.1,
        length: int = 8,
        k_tournament: int | None = 0,
    ) -> GeneticAlgorithm | None:
    config = GAConfig(
        pop_size=population,
        generations=gen,
        maximize=maximize,
        k_group_tournament = k_tournament if k_tournament is not None else 0
    )
    if p_mutation < 0 or p_mutation > 1 or p_crossover < 0 or p_crossover > 1:
        print("probability can't be out of range 0-1")
        return None

    return GeneticAlgorithm(
        config=config,
        selection_method=RouletteWheelSelection(rng),
        replacement= ElitistReplacement(rng),
        crossover_method=OrderCrossover(p_crossover, rng),
        mutation_method=BitFlipMutation(p_mutation, rng),
        representation=BinaryRepresentation(length, rng),
        feasibility=feasibilityFn,
        fitness=fitnessFn
    )

# def main():
#     ga = default_ga_binary(
#         fitnessFn=lambda x: x.count(1),
#         feasibilityFn=lambda x: True,
#         rng= random.Random(0),
#         population=8,
#         gen=8,
#         maximize=True
#     )
#     if ga is not None:
#         ga.run_GA()

# main()