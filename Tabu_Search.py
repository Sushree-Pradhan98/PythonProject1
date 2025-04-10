import numpy as np
import random
from scipy.spatial import distance_matrix


def load_cvrp_data(file_path):
    locations = []
    demands = []
    vehicle_capacity = 0

    with open(file_path, 'r') as file:
        section = None
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'NODE_COORD_SECTION':
                section = 'nodes'
                continue
            elif parts[0] == 'DEMAND_SECTION':
                section = 'demands'
                continue
            elif parts[0] == 'DEPOT_SECTION':
                section = 'depot'
                continue
            elif parts[0] == 'CAPACITY':
                vehicle_capacity = int(parts[-1])
                continue
            elif parts[0] == 'EOF':
                break

            if section == 'nodes':
                locations.append((int(parts[1]), int(parts[2])))
            elif section == 'demands':
                demands.append(int(parts[1]))

    return locations, demands, vehicle_capacity


class TabuSearchCVRP:
    def __init__(self, locations, demands, vehicle_capacity, tabu_tenure=10, max_iterations=100):
        self.locations = np.array(locations)
        self.demands = np.array(demands)
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix(self.locations, self.locations)
        self.num_customers = len(locations) - 1  # Exclude depot
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations

    def total_distance(self, solution):
        total_dist = 0
        for route in solution:
            if route:
                total_dist += self.distance_matrix[0, route[0]]
                for i in range(len(route) - 1):
                    total_dist += self.distance_matrix[route[i], route[i + 1]]
                total_dist += self.distance_matrix[route[-1], 0]
        return total_dist

    def initial_solution(self):
        customers = list(range(1, self.num_customers + 1))
        random.shuffle(customers)
        solution = []
        while customers:
            route = []
            capacity = self.vehicle_capacity
            while customers:
                next_customer = customers[0]
                if self.demands[next_customer] <= capacity:
                    route.append(next_customer)
                    capacity -= self.demands[next_customer]
                    customers.pop(0)
                else:
                    break
            solution.append(route)
        return solution

    def neighborhood(self, solution):
        neighbors = []
        for i in range(len(solution)):
            if not solution[i]:
                continue
            for j in range(len(solution[i])):
                for k in range(len(solution)):
                    if i != k:
                        new_solution = [list(route) for route in solution]
                        if new_solution[i]:
                            moved_customer = new_solution[i].pop(j)
                            new_solution[k].append(moved_customer)
                            neighbors.append(new_solution)
        return neighbors if neighbors else [solution]

    def tabu_search(self):
        best_solution = self.initial_solution()
        best_distance = self.total_distance(best_solution)
        current_solution = best_solution
        tabu_list = []

        for _ in range(self.max_iterations):
            neighborhood = self.neighborhood(current_solution)
            if not neighborhood:
                continue

            neighborhood = sorted(neighborhood, key=lambda sol: self.total_distance(sol))
            for candidate in neighborhood:
                if candidate not in tabu_list:
                    current_solution = candidate
                    dist = self.total_distance(candidate)
                    if dist < best_distance:
                        best_solution = candidate
                        best_distance = dist
                    tabu_list.append(candidate)
                    if len(tabu_list) > self.tabu_tenure:
                        tabu_list.pop(0)
                    break

        return best_solution, best_distance


def run_multiple_times(file_path, num_runs=10):
    locations, demands, capacity = load_cvrp_data(file_path)
    distances = []
    solutions = []

    for _ in range(num_runs):
        solver = TabuSearchCVRP(locations, demands, capacity)
        solution, dist = solver.tabu_search()
        distances.append(dist)
        solutions.append(solution)

    best_distance = min(distances)
    worst_distance = max(distances)
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    best_route = solutions[distances.index(best_distance)]

    print("\n--- Tabu Search Results ---")
    print(f"Best Distance:  {best_distance:.2f}")
    print(f"Worst Distance: {worst_distance:.2f}")
    print(f"Avg Distance:   {avg_distance:.2f}")
    print(f"Std Deviation:  {std_distance:.2f}")
    print(f"Best Route:     {best_route}")


# Example usage
if __name__ == "__main__":
    vrp_file = "Data/A-n32-k5.vrp"  # Change to your actual VRP file
    run_multiple_times(vrp_file, num_runs=10)
