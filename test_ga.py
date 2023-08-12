from multiple_threshold_DC import calculate_metrics, run_strategy_1
import pygad
import numpy as np
import pandas as pd
import datetime
from multiprocessing import Pool

# run = run_strategy_1(THETA_THRESHOLDS, THETA_WEIGHTS, ALL_STOCK)
# print(f"Rate of Return: {RoR:.2%}")
# print(f"Risk (Volatility): {Risk:.2%}")
# print(f"Sharpe Ratio: {Sharpe_Ratio:.2f}")
# timestamps = df["Date"]
# ALL_STOCK = df["ALL"]
# upturn, downturn, p_ext = calculate_dc(ALL_STOCK, 0.098 / 100)
# compute_sharpe_ratio()
# # Global Variables
# all_overshoot_with_osv_best = calculate_dc_indicators(
#     ALL_STOCK, upturn, downturn, p_ext, 0.098 / 100
# )

THETA_THRESHOLDS = (
    np.array([0.098, 0.22, 0.48, 0.72, 0.98, 1.22, 1.55, 1.70, 2, 2.55]) / 100
)


# Read the data from CSV
df = pd.read_csv("stock_data.csv")


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
        print("-------------------------------------------------------")
        add_to_file(
            "-------------------------------------------------------", file=file
        )
        print(f"New Weights are: {weights}")
        add_to_file(f"New Weights are: {weights}", file=file)

        for idx, stock in enumerate(df.columns[1:]):
            print(f"Running the algorithm on : {stock}")
            add_to_file(f"Running the algorithm on : {stock}", file=file)

            run = run_strategy_1(THETA_THRESHOLDS, weights, df[stock])
            _, _, Sharpe_Ratio = calculate_metrics(run["returns"], 0.025)

            print(f"Sharpe Ratio for {stock} is: {Sharpe_Ratio}")
            add_to_file(f"Sharpe Ratio for {stock} is: {Sharpe_Ratio}", file=file)

            returns[idx] = Sharpe_Ratio

        print(f"Fitness is: {np.sum(returns)}")
        add_to_file(f"Fitness is: {np.sum(returns)}", file=file)
        return np.sum(returns)


# The fitness function
def fitness_func(ga_instance, solution, solution_idx):
    # The objective is to maximize the sum of squares
    return compute_sharpe_ratio(solution)

def evaluate_population(pool, population):
    return pool.map(fitness_func, population)


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
    parallel_processing=["process", 4]
)


ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution is:", solution)
print("Fitness of the best solution is:", solution_fitness)
