import sys
import pygad
import numpy as np
import pandas as pd
import itertools

from strategy1 import load_strategy_1, strategy1_fitness_function
from strategy2 import load_strategy_2, strategy2_fitness_function
from strategy3 import load_strategy_3, strategy3_fitness_function

# Backup the original stdout
original_stdout = sys.stdout

# Open a file in write mode
log_file = open("logfile.txt", "w")

# Redirect stdout to the file
sys.stdout = log_file


def split_func(df):
    # Define the split ratios
    train_ratio = 0.8
    valid_ratio_from_train = 0.2

    # Calculate the split indices
    total_rows = len(df)
    train_split_idx = int(total_rows * train_ratio)
    valid_split_idx = int(train_split_idx * (1 - valid_ratio_from_train))

    # Split the data
    train_df = df.iloc[:valid_split_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_split_idx:train_split_idx].reset_index(drop=True)
    test_df = df.iloc[train_split_idx:].reset_index(drop=True)

    return train_df, valid_df, test_df


def normalize_population(population):
    """
    Normalize a population of chromosomes such that the sum of genes in each chromosome is 1.

    Parameters:
    - population (numpy.ndarray): Population of chromosomes.

    Returns:
    - numpy.ndarray: Normalized population of chromosomes.
    """
    return population / population.sum(axis=1, keepdims=True)


def on_crossover(ga_instance, offspring_crossover):
    return normalize_population(offspring_crossover)


def on_mutation(ga_instance, offspring_mutation):
    return normalize_population(offspring_mutation)


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
    return normalize_population(population)


def run_ga(params, loader_function):
    """
    Run the Genetic Algorithm (GA) using pygad library.

    Returns:
    - tuple: Best solution chromosome, its fitness value, and its index.
    """
    # Parameters
    num_genes = 10
    num_solutions = params["num_solutions"]
    num_generations = params["num_generations"]
    crossover_probability = params["crossover_probability"]
    mutation_probability = 1 - params["crossover_probability"]
    tournament_size = 2

    fitness_func = loader_function()

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
        on_crossover=on_crossover,
        on_mutation=on_mutation,
        mutation_probability=mutation_probability,
        mutation_type="random",
        keep_parents=1,
        initial_population=initialize_population(num_genes, num_solutions),
        parallel_processing=8,
    )

    ga_instance.run()

    return ga_instance.best_solution()


def loader_function_strategy_1() -> callable:
    """
    Load strategy 1 data and return a fitness function for evaluating solutions.

    Returns:
    - callable: A fitness function that evaluates the fitness of a solution based on strategy 1.
    """

    # Read the data from CSV
    df = pd.read_csv("data/stock_data.csv")

    train_df, valid_df, test_df = split_func(df)

    # Define thresholds
    thresholds = (
        np.array([0.098, 0.22, 0.48, 0.72, 0.98, 1.22, 1.55, 1.70, 2, 2.55])
        / 100
    )
    # Load strategy 1 decisions
    stock_decision_by_thresholds_train = load_strategy_1(
        df=train_df,
        thresholds=thresholds,
        filename="data/strategy1_train_data.pkl",
        export_excel=True,
    )

    # Load strategy 1 decisions
    stock_decision_by_thresholds_validation = load_strategy_1(
        df=train_df,
        thresholds=thresholds,
        filename="data/strategy1_valid_data.pkl",
        export_excel=True,
    )

    def fitness_func(
        ga_instance: pygad.GA, solution: list, solution_idx: int
    ) -> float:
        """
        Fitness function for evaluating a given solution.

        Parameters:
        - solution (list): The solution to evaluate.
        - solution_idx (int): Index of the solution.

        Returns:
        - float: Fitness value of the solution.
        """

        # Use the solution to generate trading signals and calculate returns for the training set
        train_fitness = strategy1_fitness_function(
            df, solution, stock_decision_by_thresholds_train
        )

        # Optimize the strategy on the training set (this step can vary based on the problem)
        # For simplicity, we'll assume the strategy is optimized if it has a positive Sharpe Ratio on the training set
        if train_fitness <= 0:
            return -np.inf  # This will make the GA avoid such solutions

        # Now, evaluate the strategy on the validation set
        validation_fitness = strategy1_fitness_function(
            df, solution, stock_decision_by_thresholds_validation
        )

        return validation_fitness

    return fitness_func


def loader_function_strategy_2() -> callable:
    """
    Load strategy 1 data and return a fitness function for evaluating solutions.

    Returns:
    - callable: A fitness function that evaluates the fitness of a solution based on strategy 1.
    """

    # Read the data from CSV
    df = pd.read_csv("data/stock_data.csv")

    # Define thresholds
    thresholds = (
        np.array([0.098, 0.22, 0.48, 0.72, 0.98, 1.22, 1.55, 1.70, 2, 2.55])
        / 100
    )

    # Load strategy 2 decisions
    stock_decision_by_thresholds = load_strategy_2(df, thresholds)

    def fitness_func(
        ga_instance: pygad.GA, solution: list, solution_idx: int
    ) -> float:
        """
        Fitness function for evaluating a given solution.

        Parameters:
        - solution (list): The solution to evaluate.
        - solution_idx (int): Index of the solution.

        Returns:
        - float: Fitness value of the solution.
        """
        return strategy2_fitness_function(
            df, solution, stock_decision_by_thresholds
        )

    return fitness_func


def loader_function_strategy_3() -> callable:
    """
    Load strategy 1 data and return a fitness function for evaluating solutions.

    Returns:
    - callable: A fitness function that evaluates the fitness of a solution based on strategy 1.
    """

    # Read the data from CSV
    df = pd.read_csv("data/stock_data.csv")

    # Define thresholds
    thresholds = np.array([0.098, 0.22, 0.48, 0.72, 0.98]) / 100

    # Load strategy 2 decisions
    stock_decision_by_thresholds = load_strategy_3(df, thresholds, True)

    def fitness_func(
        ga_instance: pygad.GA, solution: list, solution_idx: int
    ) -> float:
        """
        Fitness function for evaluating a given solution.

        Parameters:
        - solution (list): The solution to evaluate.
        - solution_idx (int): Index of the solution.

        Returns:
        - float: Fitness value of the solution.
        """
        return strategy3_fitness_function(
            df, solution, stock_decision_by_thresholds
        )

    return fitness_func


if __name__ == "__main__":
    param_grid = {
        "num_solutions": [20, 50, 70, 100, 150, 200, 300],
        "num_generations": [15, 18, 25, 30, 35, 45],
        "crossover_probability": [0.75, 0.85, 0.95, 0.99],
    }

    # Generate all combinations of hyperparameters
    all_params = [
        dict(zip(param_grid.keys(), values))
        for values in itertools.product(*param_grid.values())
    ]

    best_score = float("-inf")
    best_params = None

    for params in all_params:
        solution, solution_fitness, _ = run_ga(
            params, loader_function_strategy_1
        )
        if solution_fitness > best_score:
            best_score = solution_fitness
            best_params = params

    print("Best Score:", best_score)
    print("Best Hyperparameters:", best_params)
    print("Weights", solution)


# Restore the original stdout
sys.stdout = original_stdout
log_file.close()
