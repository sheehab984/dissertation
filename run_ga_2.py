import datetime
import pygad
import numpy as np
import pandas as pd
import itertools
import logging

from strategy2 import load_strategy_2, strategy2_fitness_function


# Configure logging settings
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app2_train.log",
    filemode="w",
)  # 'w' will overwrite the log file each time the script runs. Use 'a' to append.

# Create a logger object
logger = logging.getLogger()


def split_func(df):
    # Define the split ratios
    train_ratio = 0.8

    # Calculate the split indices
    total_rows = len(df)
    train_split_idx = int(total_rows * train_ratio)

    # Split the data
    train_df = df.iloc[:train_split_idx].reset_index(drop=True)
    test_df = df.iloc[train_split_idx:].reset_index(drop=True)

    return train_df, test_df


def normalize_population(population):
    """
    Normalize a population of chromosomes such that the sum of genes in each chromosome is 1.

    Parameters:
    - population (numpy.ndarray): Population of chromosomes.

    Returns:
    - numpy.ndarray: Normalized population of chromosomes.
    """
    return np.abs(population) / np.abs(population).sum(axis=1, keepdims=True)


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


def on_generation(ga_instance):
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("output/strategy2_train_run.txt", "a") as f:
        f.write(f"Generation completed at {current_timestamp}.\n")

    ga_instance.logger.info(
        "Generation = {generation}".format(
            generation=ga_instance.generations_completed
        )
    )
    ga_instance.logger.info(
        "Fitness    = {fitness}".format(
            fitness=ga_instance.best_solution(
                pop_fitness=ga_instance.last_generation_fitness
            )[1]
        )
    )


def run_ga(params, loader_function):
    """
    Run the Genetic Algorithm (GA) using pygad library.

    Returns:
    - tuple: Best solution chromosome, its fitness value, and its index.
    """
    # Parameters
    num_genes = params["num_genes"]
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
        logger=logger,
        on_generation=on_generation,
        parallel_processing=50,
    )

    ga_instance.run()

    return ga_instance.best_solution()


def loader_function_strategy_2() -> callable:
    """
    Load strategy 1 data and return a fitness function for evaluating solutions.

    Returns:
    - callable: A fitness function that evaluates the fitness of a solution based on strategy 1.
    """

    # Read the data from CSV
    # Read the data from CSV
    df = pd.read_csv("data/stock_data.csv")

    train_df, test_df = split_func(df)

    # Define thresholds
    thresholds = (
        np.array([0.098, 0.22, 0.48, 0.72, 0.98, 1.22, 1.55, 1.70, 2, 2.55])
        / 100
    )

    # Load strategy 2 decisions
    stock_decision_by_thresholds_train = load_strategy_2(
        df=train_df,
        thresholds=thresholds,
        pkl_filename="data/strategy2_train_data.pkl",
        excel_filename="output/strategy2_train.xlsx",
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
        print("Running fitness function for solution " + str(solution_idx))
        print("Weights are " + str(solution))

        # Use the solution to generate trading signals and calculate returns for the training set
        RoR, volatility, sharpe_ratio = strategy2_fitness_function(
            df, solution, stock_decision_by_thresholds_train
        )

        print(
            "Train fitness function for solution "
            + str(sharpe_ratio)
            + "\t"
            + str(volatility)
            + "\t"
            + str(RoR)
        )

        return sharpe_ratio

    return fitness_func


if __name__ == "__main__":
    param_grid = {
        "num_genes": [10],
        "num_solutions": [100],
        "num_generations": [18],
        "crossover_probability": [0.95],
    }
    for i in range(50):
        # Get the current timestamp
        current_timestamp = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open("output/strategy2_train_run.txt", "a") as f:
            f.write(
                f"--------Starting a new GA instance at {current_timestamp}.------------\n"
            )
            f.write(
                f"Starting running GA for strategy 2 with parameters. Run {i + 1}\n"
            )
        all_params = [
            dict(zip(param_grid.keys(), values))
            for values in itertools.product(*param_grid.values())
        ]

        solution, solution_fitness, _ = run_ga(
            all_params[0], loader_function_strategy_2
        )
        with open("output/strategy2_train_run.txt", "a") as f:
            f.write(
                str(i)
                + "\t"
                + str(solution)
                + "\t"
                + str(solution_fitness)
                + "\n"
            )
            f.write(
                f"----------Finished running GA for strategy 2. Run {i + 1}-------------\n"
            )
