import numpy as np
import random
from scipy.spatial import distance_matrix

def load_cvrp_data(file_path):
    locations = []
    demands = []
    vehicle_capacity = 0
    num_customers = 0  # To track the correct size of demands list

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
                vehicle_capacity = int(parts[-1])  # Extract the last element (which is the capacity value)
                continue
            elif parts[0] == 'EOF':
                break

            if section == 'nodes':
                locations.append((int(parts[1]), int(parts[2])))
            elif section == 'demands':
                if len(demands) == 0:  # Ensure demands array size matches number of customers
                    num_customers = len(locations)
                    demands = [0] * num_customers  # Initialize demands list
                demands[int(parts[0]) - 1] = int(parts[1])  # Correct indexing

    return locations, demands, vehicle_capacity


class RandomCVRP:
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
                total_dist += self.distance_matrix[0, route[0]]  # Depot to first customer
                for i in range(len(route) - 1):
                    total_dist += self.distance_matrix[route[i], route[i + 1]]  # Customer to next customer
                total_dist += self.distance_matrix[route[-1], 0]  # Last customer to depot
        return total_dist

    def random_solution(self):
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


# Load data from file
file_path = "/Users/shreejoy/PycharmProjects/PythonProject1/Data/A-n39-k5.vrp"
locations, demands, capacity = load_cvrp_data(file_path)

# Solve using RandomCVRP
random_solver = RandomCVRP(locations, demands, capacity)
random_solution = random_solver.random_solution()

