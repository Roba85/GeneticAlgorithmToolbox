from dataclasses import dataclass, field
import math
from interfaces import Representation, Selection, Crossover, Mutation, FeasibleFn, FitnessFn, Chromosome, Population, Replacement

@dataclass
class GAConfig:
    pop_size: int = 100
    generations: int = 200
    maximize: bool = True
    stop_after_x_static_avg: int = 5
    repair_limit: int = 100
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
        self.best_fitness = 0.0 if self.config.maximize else float('inf')
        self.best_chromosome = None
        self.last_best_change_at = 0


    def run_GA(self):
        initial_population = self.representation.random_population(
                size=self.config.pop_size
            )
        # Ensuring population is feasible
        fitness = []
        same_fitness_for = 0
        convergence = False
        last_gen = 0

        fitness_avg = 0.0
        for i in range(self.config.pop_size):
            repairs = 0
            while not self.feasibility_fn(initial_population[i]) and repairs < self.config.repair_limit:
                initial_population[i] = self.representation.random_Chromosome()
                repairs += 1
            if repairs == self.config.repair_limit:
                print(f"Failed to fix invalid chromosomes after {repairs} tries")
                return

            fitness.append(self.fitness_fn(initial_population[i]))
            self.update_fitness(initial_population[i], fitness[i], 0)

        fitness_avg = math.fsum(fitness) / self.config.pop_size
        self.print_generation(initial_population, fitness, fitness_avg, 0)

        logger = 1
        for gen in range(1, self.config.generations + 1):
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

                repairs = 0
                while not self.feasibility_fn(res1) or not self.feasibility_fn(res2) and repairs < self.config.repair_limit:
                    res1, res2 = self.crossover_method(
                                parent1=new_population[i], 
                                parent2=new_population[i + 1]
                            )
                    res1 = self.mutation_method(res1)
                    res2 = self.mutation_method(res2)
                    repairs += 1
                if repairs == self.config.repair_limit:
                    print(f"Failed to fix invalid chromosomes after {repairs} tries")
                    return

                new_population[i] = res1
                new_population[i + 1] = res2
                new_fitness.append(self.fitness_fn(res1))
                new_fitness.append(self.fitness_fn(res2))

            if len(new_population) % 2 != 0:
                new_fitness.append(self.fitness_fn(new_population[-1]))


            # Replacement
            initial_population, fitness = self.replacement_method(
                population=initial_population,
                pop_fitnesses=fitness,
                offspring=new_population,
                off_fitnesses=new_fitness,
                maximize=self.config.maximize
            )
            new_fit_avg = math.fsum(fitness) / len(fitness)

            for i in range(len(initial_population)):
                self.update_fitness(initial_population[i], fitness[i], gen)

            if logger == self.config.log_every:
                self.print_generation(initial_population, fitness, new_fit_avg, gen)
                logger = 0
            logger += 1

            if math.isclose(new_fit_avg, fitness_avg, rel_tol=1e-3, abs_tol=1e-3):
                same_fitness_for += 1
            else:
                same_fitness_for = 0
            fitness_avg = new_fit_avg

            last_gen = gen
            if self.config.stop_after_x_static_avg == same_fitness_for:
                convergence = True
                break

        # self.print_sol(
        #     last_gen,
        #     convergence
        # )
        return (self.best_chromosome, self.best_fitness)

            
    def update_fitness(self, chromosome: Chromosome, fit: float, gen: int):
        if self.config.maximize:
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_chromosome = chromosome 
                self.last_best_change_at = gen
        else:
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_chromosome = chromosome 
                self.last_best_change_at = gen



    def print_generation(self, population : Population, fitness : list[float], avg_fitness: float, gen_number: int):
        print('=' * 10)
        print(f"Generation Number: {gen_number}")
        print(f"Average Fitness: {avg_fitness}")
        if self.config.log_verbose:
            print('=' * 10)
            print("Population Details:")
            for i in range(len(population)):
                print(f"Chromosome: {population[i]}")
                print(f"Fitness: {fitness[i]}")
        print('=' * 10)

    def print_sol(self, last_gen: int, convergence: bool = True):
        # print('=' * 10)
        print(f"Stopping at generation {last_gen}")
        print(f"Stopped due to convergence: {convergence}")
        print(f"Solution found at generation {self.last_best_change_at}")
        print(f"Chromosome: {self.best_chromosome}")
        print(f"Fitness: {self.best_fitness}")
