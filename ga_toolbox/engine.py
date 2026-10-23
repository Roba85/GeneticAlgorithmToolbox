from dataclasses import dataclass, field
import math
from typing import Optional, List, Tuple
from interfaces import Representation, Selection, Crossover, Mutation, FeasibleFn, FitnessFn, Chromosome, Population, Replacement

@dataclass
class GAConfig:
    pop_size: int = 100
    generations: int = 200
    maximize: bool = True     
    p_crossover: float = 0.7
    p_mutation: float = 0.2
    k_group_tournament: int = 0
    log_every: int = 10
    log_verbose: bool = False

class GeneticAlgorithm:
    def __init__(
            self,
            config : GAConfig,
            representation: Representation,
            replacement: Replacement,
            selection_method: Selection,
            crossover_method: Crossover,
            mutation_method: Mutation,
            feasibility: FeasibleFn,
            fitness: FitnessFn
        ) -> None:
        self.config = config
        self.representation = representation
        self.replacement_method = replacement
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.feasibility_fn = feasibility
        self.fitness_fn = fitness
        self.best_fitness = 0.0
        self.best_chromosome = Chromosome


    def run_GA(self):
        initial_population = self.representation.random_population(
                size=self.config.pop_size
            )
        # Ensuring population is feasible
        fitness = []

        fitness_avg = 0.0
        for i in range(self.config.pop_size):
            while not self.feasibility_fn(initial_population[i]):
                initial_population[i] = self.representation.random_Chromosome()

            fitness.append(self.fitness_fn(initial_population[i]))
            if fitness[i] > self.best_fitness:
                self.best_chromosome = initial_population[i]
                self.best_fitness = fitness[i]

        fitness_avg = math.fsum(fitness) / self.config.pop_size
        self.print_generation(initial_population, fitness, fitness_avg, 0)

        logger = 1
        for gen in range(self.config.generations):
            new_population = self.selection_method(
                population=initial_population, 
                fitnesses=fitness,
                maximize=self.config.maximize,
                k=self.config.k_group_tournament
            )

            # crossover & mutation
            new_fitness = []
            for i in range(0, len(new_population) - 1, 2):
                res1, res2 = self.crossover_method(
                    parent1=new_population[i], 
                    parent2=new_population[i + 1]
                )
                res1 = self.mutation_method(res1)
                res2 = self.mutation_method(res2)

                while not self.feasibility_fn(res1) or not self.feasibility_fn(res2):
                    res1, res2 = self.crossover_method(
                                parent1=new_population[i], 
                                parent2=new_population[i + 1]
                            )
                    res1 = self.mutation_method(res1)
                    res2 = self.mutation_method(res2)

                new_population[i] = res1
                new_population[i + 1] = res2
                new_fitness.append(self.fitness_fn(res1))
                new_fitness.append(self.fitness_fn(res2))


            # Replacement
            initial_population, fitness = self.replacement_method(
                population=initial_population,
                pop_fitnesses=fitness,
                offspring=new_population,
                off_fitnesses=new_fitness,
                maximize=self.config.maximize
            )
            fitness_avg = math.fsum(fitness) / len(fitness)
            if logger == self.config.log_every:
                self.print_generation(initial_population, fitness, fitness_avg, gen)
                logger = 0
            logger += 1


    def print_generation(self, population : Population, fitness : List[float], avg_fitness: float, gen_number: int):
        print('=' * 10)
        print(f"Generation Number: {gen_number}")
        print(f"Average Fitness: {avg_fitness}")
        print('=' * 10)
        print("Population Details:")
        if self.config.log_verbose:
            for i in range(len(population)):
                print(f"Chromosome: {population[i]}")
                print(f"Fitness: {fitness[i]}")
        print('=' * 10)