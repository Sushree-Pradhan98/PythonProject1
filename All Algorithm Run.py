import os
from tabulate import tabulate
from Greedy_Algorithm import GreedyCVRP
from Random_Search import RandomCVRP
from Tabu_Search import TabuSearchCVRP
from Simulated_Annealing import SimulatedAnnealingCVRP
from Genetic_Algorithm import GeneticAlgorithmCVRP

# Change this to your specific file
DATA_FILE = "/Users/shreejoy/PycharmProjects/PythonProject1/Data/A-n32-k5.vrp"


def run_algorithm(algorithm_class, file_path):
    """Runs a CVRP algorithm and returns the total distance of the solution."""
    algorithm = algorithm_class(file_path)
    solution = algorithm.solve()

    # Extract total distance from the solution
    total_distance = solution.get_total_distance()  # Ensure this method exists in all algorithms
    return total_distance


def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: File {DATA_FILE} not found!")
        return

    print(f"Running all CVRP algorithms on: {DATA_FILE}\n")

    algorithms = {
        "Greedy": GreedyCVRP,
        "Genetic": GeneticAlgorithmCVRP,
        "Simulated Annealing": SimulatedAnnealingCVRP,
        "Random": RandomCVRP,
        "Tabu Search": TabuSearchCVRP,
    }

    results = []

    for name, algorithm_class in algorithms.items():
        total_distance = run_algorithm(algorithm_class, DATA_FILE)
        results.append((name, total_distance))

    # Print results in a table format
    print("\n===== Results (Total Distance) =====")
    print(tabulate(results, headers=["Algorithm", "Total Distance"], tablefmt="grid"))


if __name__ == "__main__":
    main()
