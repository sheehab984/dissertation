import pygad
import numpy as np
import pandas as pd
import datetime
from strategy1_v1 import compute_threshold_dc_summaries, run_strategy_1


def calculate_metrics(returns, risk_free_rate=0.01):
    """
    Calculate RoR, Risk, and Sharpe Ratio from a return array.

    Parameters:
    - returns (list or np.array): Array of returns.
    - risk_free_rate (float): Risk-free rate. Default is 0.01 (1%).

    Returns:
    - RoR, Risk, Sharpe Ratio
    """

    # Convert returns to numpy array for easier calculations
    returns = np.array(returns)

    # Calculate RoR
    RoR = np.mean(returns)

    # Calculate Risk (Volatility)
    Risk = np.std(returns)

    # Calculate Sharpe Ratio
    Sharpe_Ratio = (RoR - risk_free_rate) / Risk

    return RoR, Risk, Sharpe_Ratio


# Read the data from CSV
df = pd.read_csv("data/stock_data.csv")

THETA_THRESHOLDS = (
    np.array([0.098, 0.22, 0.48, 0.72, 0.98, 1.22, 1.55, 1.70, 2, 2.55]) / 100
)

THRESHOLDS_DC_SUMMARIES_CACHE = {col: pd.DataFrame() for col in df.columns[1:]}


# Mock Sharpe Ratio computation
def compute_sharpe_ratio(weights):
    def add_to_file(s, file):
        file.write(s + "\n")

    with open("output.txt", "a") as file:
        returns = [0] * df.columns[1:].shape[0]
        # Get the current date and time
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        add_to_file(
            f"{current_time}: This will be appended to the file with a timestamp.",
            file=file,
        )
        print(
            f"{current_time}: This will be appended to the file with a timestamp."
        )
        print("-------------------------------------------------------")
        add_to_file(
            "-------------------------------------------------------", file=file
        )
        print(f"New Weights are: {weights}")
        add_to_file(f"New Weights are: {weights}", file=file)

        for idx, stock in enumerate(df.columns[1:]):
            print(f"Running the algorithm on : {stock}")
            add_to_file(f"Running the algorithm on : {stock}", file=file)

            run = run_strategy_1(
                THRESHOLDS_DC_SUMMARIES_CACHE[stock],
                THETA_THRESHOLDS,
                weights,
                df[stock],
            )
            _, _, Sharpe_Ratio = calculate_metrics(run["returns"], 0.025)

            print(f"Sharpe Ratio for {stock} is: {Sharpe_Ratio}")
            add_to_file(
                f"Sharpe Ratio for {stock} is: {Sharpe_Ratio}", file=file
            )

            returns[idx] = Sharpe_Ratio

        print(f"Fitness is: {np.sum(returns)}")
        add_to_file(f"Fitness is: {np.sum(returns)}", file=file)
        return np.sum(returns)


# The fitness function
def fitness_func(ga_instance, solution, solution_idx):
    # The objective is to maximize the sum of squares
    return compute_sharpe_ratio(solution)


def main():
    for idx, stock in enumerate(df.columns[1:]):
        prices = df[stock]
        # Split the data into an initial 80% train and 20% test
        num_rows = len(df)
        train_idx = int(0.8 * num_rows)
        train_df_initial = prices[:train_idx]
        test_df = prices[train_idx:]

        # Split the initial train_df into 80% train and 20% validation
        val_idx = int(0.8 * len(train_df_initial))
        train_df = train_df_initial[:val_idx]
        val_df = train_df_initial[val_idx:]

        threshold_dc_summaries = compute_threshold_dc_summaries(
            train_df, THETA_THRESHOLDS
        )
        THRESHOLDS_DC_SUMMARIES_CACHE[stock] = threshold_dc_summaries

    # Create an instance of the GA class
    ga_instance = pygad.GA(
        num_generations=18,
        num_parents_mating=2,
        fitness_func=fitness_func,
        sol_per_pop=100,
        num_genes=10,
        gene_type=np.float32,
        init_range_low=0,
        init_range_high=1,
        parent_selection_type="tournament",
        K_tournament=2,
        crossover_type="single_point",
        crossover_probability=0.95,
        mutation_type="random",
        mutation_probability=0.05,
        #    parallel_processing=["process", 4],
    )
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Best solution is:", solution)
    print("Fitness of the best solution is:", solution_fitness)


main()
