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


class GeneticAlgorithmCVRP:
    def __init__(self, locations, demands, vehicle_capacity, population_size=10, mutation_rate=0.1, generations=100):
        self.locations = np.array(locations)
        self.demands = np.array(demands)
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix(self.locations, self.locations)
        self.num_customers = len(locations) - 1  # Exclude depot
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def total_distance(self, solution):
        total_dist = 0
        for route in solution:
            if route:
                total_dist += self.distance_matrix[0, route[0]]  # Depot to first customer
                for i in range(len(route) - 1):
                    total_dist += self.distance_matrix[route[i], route[i + 1]]  # Customer to next customer
                total_dist += self.distance_matrix[route[-1], 0]  # Last customer to depot
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

    def population(self):
        return [self.initial_solution() for _ in range(self.population_size)]

    def crossover(self, parent1, parent2):
        size = len(parent1)
        point = random.randint(1, size - 1)

        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        return child1, child2

    def mutation(self, solution):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(solution)), 2)
            if solution[i] and solution[j]:
                swap_idx1 = random.randint(0, len(solution[i]) - 1)
                swap_idx2 = random.randint(0, len(solution[j]) - 1)
                solution[i][swap_idx1], solution[j][swap_idx2] = solution[j][swap_idx2], solution[i][swap_idx1]
        return solution

    def selection(self, population):
        population.sort(key=lambda sol: self.total_distance(sol))
        return population[:2]

    def genetic_algorithm(self):
        population = self.population()

        for gen in range(self.generations):
            next_population = []
            while len(next_population) < self.population_size:
                parent1, parent2 = self.selection(population)
                child1, child2 = self.crossover(parent1, parent2)
                next_population.append(self.mutation(child1))
                next_population.append(self.mutation(child2))

            population = next_population

        best_solution = min(population, key=lambda sol: self.total_distance(sol))
        return best_solution


# ------------------------------
# Run Genetic Algorithm Multiple Times and Analyze
# ------------------------------
def run_multiple_times(file_path, num_runs=10):
    locations, demands, capacity = load_cvrp_data(file_path)
    distances = []
    solutions = []

    for _ in range(num_runs):
        ga_solver = GeneticAlgorithmCVRP(locations, demands, capacity)
        solution = ga_solver.genetic_algorithm()
        dist = ga_solver.total_distance(solution)
        distances.append(dist)
        solutions.append(solution)

    best_distance = min(distances)
    worst_distance = max(distances)
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    best_route = solutions[distances.index(best_distance)]

    print("\n--- Genetic Algorithm Results ---")
    print(f"Best Distance:  {best_distance:.2f}")
    print(f"Worst Distance: {worst_distance:.2f}")
    print(f"Avg Distance:   {avg_distance:.2f}")
    print(f"Std Deviation:  {std_distance:.2f}")
    print(f"Best Route:     {best_route}")


# Example usage
if __name__ == "__main__":
    vrp_file = "Data/A-n32-k5.vrp"  # Replace with your VRP file path
    run_multiple_times(vrp_file, num_runs=10)
