import numpy as np
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
                vehicle_capacity = int(parts[-1])  # Take the last element instead of parts[1]
                continue
            elif parts[0] == 'EOF':
                break

            if section == 'nodes':
                locations.append((int(parts[1]), int(parts[2])))
            elif section == 'demands':
                demands.append(int(parts[1]))

    return locations, demands, vehicle_capacity


class GreedyCVRP:
    def __init__(self, locations, demands, vehicle_capacity):
        self.locations = np.array(locations)
        self.demands = np.array(demands)
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix(self.locations, self.locations)
        self.num_customers = len(locations) - 1  # Exclude depot

    def total_distance(self, solution):
        total_dist = 0
        for route in solution:
            if route:
                total_dist += self.distance_matrix[0, route[0]]
                for i in range(len(route) - 1):
                    total_dist += self.distance_matrix[route[i], route[i + 1]]
                total_dist += self.distance_matrix[route[-1], 0]
        return total_dist

    def greedy_solution(self):
        customers = set(range(1, self.num_customers + 1))
        solution = []

        while customers:
            route = []
            capacity = self.vehicle_capacity
            current_location = 0  # Start from depot

            while customers:
                next_customer = min(customers, key=lambda c: self.distance_matrix[current_location, c])
                if self.demands[next_customer] <= capacity:
                    route.append(next_customer)
                    capacity -= self.demands[next_customer]
                    current_location = next_customer
                    customers.remove(next_customer)
                else:
                    break
            solution.append(route)

        return self.optimize_routes(solution)

    def optimize_routes(self, solution):
        """ Optimize routes by reordering them to minimize total distance."""
        for route in solution:
            if len(route) > 2:
                route.sort(key=lambda c: self.distance_matrix[0, c])  # Sort by distance from depot
        return solution


# Load data from file
file_path = "/Users/shreejoy/PycharmProjects/PythonProject1/Data/A-n32-k5.vrp"
locations, demands, capacity = load_cvrp_data(file_path)

# Solve using Greedy Algorithm
greedy_solver = GreedyCVRP(locations, demands, capacity)
best_greedy_solution = greedy_solver.greedy_solution()

