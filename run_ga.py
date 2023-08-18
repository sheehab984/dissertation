import pygad
import numpy as np
import pandas as pd

from strategy1 import load_strategy_1, strategy1_fitness_function


def initialize_population(num_genes, sol_per_pop):
    """
    Initialize a population of chromosomes with genes such that the sum of genes in each chromosome is 1.

    Parameters:
    - num_genes (int): Number of genes in each chromosome.
    - sol_per_pop (int): Number of solutions (chromosomes) in the population.

    Returns:
    - numpy.ndarray: Initialized population of chromosomes.
    """
    population = np.random.rand(sol_per_pop, num_genes)
    population = population / population.sum(axis=1, keepdims=True)
    return population


def run_ga(strategy_fitness_function):
    """
    Run the Genetic Algorithm (GA) using pygad library.

    Returns:
    - tuple: Best solution chromosome, its fitness value, and its index.
    """
    # Parameters
    num_genes = 10
    num_solutions = 100
    num_generations = 18
    crossover_probability = 0.95
    mutation_probability = 0.05
    tournament_size = 2

    # Read the data from CSV
    df = pd.read_csv("data/stock_data.csv")

    thresholds = (
        np.array([0.098, 0.22, 0.48, 0.72, 0.98, 1.22, 1.55, 1.70, 2, 2.55])
        / 100
    )

    stock_decision_by_thresholds = load_strategy_1(df, thresholds)

    def fitness_func(solution, solution_idx):
        return strategy_fitness_function(
            df, solution, stock_decision_by_thresholds
        )

    # Create an instance of the GA class
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=2,
        fitness_func=fitness_func,
        sol_per_pop=num_solutions,
        num_genes=num_genes,
        gene_type=np.float32,
        init_range_low=0,
        init_range_high=1,
        parent_selection_type="tournament",
        K_tournament=tournament_size,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        mutation_type="random",
        keep_parents=1,
        initial_population=initialize_population(num_genes, num_solutions),
    )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    return solution, solution_fitness, solution_idx


if __name__ == "__main__":
    solution, solution_fitness, solution_idx = run_ga(
        strategy1_fitness_function
    )
    print("Best solution is:", solution)
    print("Fitness of the best solution is:", solution_fitness)