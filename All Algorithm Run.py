import os
import time
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from Genetic_Algorithm import GeneticAlgorithmCVRP, load_cvrp_data as load_ga
from Greedy_Algorithm import GreedyCVRP, load_cvrp_data as load_greedy
from Random_Search import RandomCVRP, load_cvrp_data as load_random
from Simulated_Annealing import SimulatedAnnealingCVRP, load_cvrp_data as load_sa
from Tabu_Search import TabuSearchCVRP, load_cvrp_data as load_tabu

# Global cache for optimal values
OPTIMAL_CACHE = {
    # Augerat set A
    "A-n32-k5": 784, "A-n33-k5": 661, "A-n33-k6": 742, "A-n34-k5": 778,
    # Augerat set B
    "B-n31-k5": 672, "B-n34-k5": 788, "B-n35-k5": 955, "B-n38-k6": 805,
    # Christofides and Eilon
    "E-n22-k4": 375, "E-n23-k3": 569, "E-n30-k3": 534, "E-n33-k4": 835,
    # Golden et al.
    "M-n101-k10": 820, "M-n121-k7": 1034, "M-n151-k12": 1015
}


def calculate_optimal(locations, demands, vehicle_capacity):
    """Calculate optimal solution using OR-Tools."""

    def create_distance_matrix(locs):
        return [
            [int(np.hypot(loc1[0] - loc2[0], loc1[1] - loc2[1]))
             for loc2 in locs]
            for loc1 in locs
        ]

    # Create routing index manager
    distance_matrix = create_distance_matrix(locations)
    manager = pywrapcp.RoutingIndexManager(
        len(distance_matrix),
        len([d for d in demands if d > 0]),  # Number of vehicles
        0  # Depot index
    )

    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    # Define distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [vehicle_capacity] * len([d for d in demands if d > 0]),
        True,  # start cumul to zero
        'Capacity'
    )

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(10)  # Reduced for faster results

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        return solution.ObjectiveValue()
    return None


def get_stats(solver, runs=10, method_name=""):
    distances = []
    for _ in range(runs):
        if method_name == "random":
            solution = solver.random_solution()
            distances.append(solver.total_distance(solution))
        elif method_name == "greedy":
            solution = solver.greedy_solution()
            distances.append(solver.total_distance(solution))
        elif method_name == "ga":
            solution = solver.genetic_algorithm()
            distances.append(solver.total_distance(solution))
        elif method_name == "sa":
            _, distance = solver.simulated_annealing()
            distances.append(distance)
        elif method_name == "tabu":
            _, distance = solver.tabu_search()
            distances.append(distance)
    return {
        "best": round(np.min(distances), 1),
        "worst": round(np.max(distances), 1),
        "avg": round(np.mean(distances), 1),
        "std": round(np.std(distances), 1),
    }


def run_all_algorithms(file_path):
    row = {}
    base_name = os.path.basename(file_path).split(".")[0]
    row["Instance"] = base_name

    # Load data once
    loc, dem, cap = load_random(file_path)

    # Get or calculate optimal
    if base_name not in OPTIMAL_CACHE:
        print(f"Calculating optimal for {base_name}...")
        optimal = calculate_optimal(loc, dem, cap)
        OPTIMAL_CACHE[base_name] = optimal if optimal is not None else "N/A"

    row["Optimal"] = OPTIMAL_CACHE[base_name]

    # Run all algorithms
    # Random Search
    random_solver = RandomCVRP(loc, dem, cap)
    rand_stats = get_stats(random_solver, runs=10, method_name="random")
    row.update({f"Random_{k}": v for k, v in rand_stats.items()})

    # Greedy
    greedy_solver = GreedyCVRP(loc, dem, cap)
    greedy_stats = get_stats(greedy_solver, runs=1, method_name="greedy")
    row["Greedy"] = greedy_stats["best"]

    # Genetic Algorithm
    ga_solver = GeneticAlgorithmCVRP(loc, dem, cap)
    ga_stats = get_stats(ga_solver, runs=10, method_name="ga")
    row.update({f"GA_{k}": v for k, v in ga_stats.items()})

    # Tabu Search
    tabu_solver = TabuSearchCVRP(loc, dem, cap)
    tabu_stats = get_stats(tabu_solver, runs=10, method_name="tabu")
    row.update({f"Tabu_{k}": v for k, v in tabu_stats.items()})

    # Simulated Annealing
    sa_solver = SimulatedAnnealingCVRP(loc, dem, cap)
    sa_stats = get_stats(sa_solver, runs=10, method_name="sa")
    row.update({f"SA_{k}": v for k, v in sa_stats.items()})

    return row


def main():
    data_dir = "Data"
    vrp_files = [f for f in os.listdir(data_dir) if f.endswith(".vrp")]

    if not vrp_files:
        print("No .vrp files found in the Data directory!")
        return

    print("\nFound these VRP files:")
    for file in vrp_files:
        print(f"- {file}")

    all_results = []
    for file_name in vrp_files:
        file_path = os.path.join(data_dir, file_name)
        print(f"\nProcessing {file_name}...")
        start_time = time.time()
        row = run_all_algorithms(file_path)
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")
        all_results.append(row)

    df = pd.DataFrame(all_results)
    df.set_index("Instance", inplace=True)

    # Save to Excel and CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    excel_path = f"results_{timestamp}.xlsx"
    csv_path = f"results_{timestamp}.csv"

    df.to_excel(excel_path)
    df.to_csv(csv_path)

    print("\n=== Final Results ===")
    print(df.to_string())
    print(f"\nResults saved to:\n- {excel_path}\n- {csv_path}")

    # Save optimal cache for future use
    with open("optimal_cache.txt", "w") as f:
        for k, v in sorted(OPTIMAL_CACHE.items()):
            f.write(f"'{k}': {v},\n")


if __name__ == "__main__":
    main()