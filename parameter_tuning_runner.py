import os
import numpy as np
import pandas as pd
from datetime import datetime
from Tabu_Search import TabuSearchCVRP, load_cvrp_data as load_tabu
from Simulated_Annealing import SimulatedAnnealingCVRP, load_cvrp_data as load_sa
from Genetic_Algorithm import GeneticAlgorithmCVRP, load_cvrp_data as load_ga


class ParameterTuner:
    def __init__(self, easy_path, hard_path):
        self.instances = {
            "Easy Instance (A-n32-k5)": easy_path,
            "Hard Instance (A-n60-k9)": hard_path
        }
        self.RUNS_PER_CONFIG = 10  # Number of runs per configuration

    def summarize(self, distances):
        """Calculate statistics for a set of distances"""
        valid_distances = [d for d in distances if d != float('inf')]
        if not valid_distances:
            return {
                "best": np.nan,
                "worst": np.nan,
                "avg": np.nan,
                "std": np.nan,
                "success_rate": 0.0
            }
        return {
            "best": round(float(np.min(valid_distances)),  # Round to nearest integer
                          "worst": round(float(np.max(valid_distances))),
        "avg": round(float(np.mean(valid_distances))),
        "std": round(float(np.std(valid_distances))) if len(valid_distances) > 1 else 0.0,
        "success_rate": len(valid_distances) / len(distances)
        }

        def run_trials(self, algo_name, solver_class, loader, param_sets):
            results = {}
            for instance_name, path in self.instances.items():
                loc, dem, cap = loader(path)

                print(f"\n🔧 Tuning {algo_name.upper()} on {instance_name}...")

                for config_name, params in param_sets.items():
                    distances = []

                    print(f"⚙️ Testing config {config_name}: {params}")

                    for _ in range(self.RUNS_PER_CONFIG):
                        try:
                            solver = solver_class(loc, dem, cap)

                            if algo_name == "tabu":
                                _, distance = solver.tabu_search(
                                    tabu_tenure=params.get('tabu_tenure', 10),
                                    max_iter=params.get('max_iter', 100)
                                )
                            elif algo_name == "sa":
                                _, distance = solver.simulated_annealing(
                                    temp=params.get('temp', 1000),
                                    cooling_rate=params.get('cooling_rate', 0.95),
                                    max_iter=params.get('max_iter', 100)
                                )
                            elif algo_name == "ga":
                                solution = solver.genetic_algorithm(
                                    pop_size=params.get('pop_size', 50),
                                    mut_rate=params.get('mut_rate', 0.1),
                                    n_gen=params.get('n_gen', 100)
                                )
                                distance = solver.total_distance(solution)

                            distances.append(distance)
                        except Exception as e:
                            print(f"⚠️ Error in trial: {str(e)}")
                            distances.append(float('inf'))

                    stats = self.summarize(distances)
                    key = f"{algo_name.upper()}-{instance_name}-{config_name}"
                    results[key] = stats

                    print(f"✅ Results: Best={stats['best']}, Avg={stats['avg']}, "
                          f"Std={stats['std']}, Success={stats['success_rate'] * 100:.0f}%")

            return results

        def run_all(self):
            # Enhanced parameter configurations
            tabu_configs = {
                "short": {"tabu_tenure": 5, "max_iter": 200},
                "medium": {"tabu_tenure": 10, "max_iter": 500},
                "long": {"tabu_tenure": 15, "max_iter": 1000}
            }

            sa_configs = {
                "fast_cool": {"temp": 500, "cooling_rate": 0.85, "max_iter": 200},
                "balanced": {"temp": 1000, "cooling_rate": 0.95, "max_iter": 500},
                "slow_cool": {"temp": 2000, "cooling_rate": 0.99, "max_iter": 1000}
            }

            ga_configs = {
                "small_pop": {"pop_size": 30, "mut_rate": 0.05, "n_gen": 200},
                "standard": {"pop_size": 50, "mut_rate": 0.1, "n_gen": 500},
                "large_pop": {"pop_size": 100, "mut_rate": 0.2, "n_gen": 1000},
                "high_mut": {"pop_size": 50, "mut_rate": 0.3, "n_gen": 300}
            }

            results = {}

            print("\n=== Starting Parameter Tuning ===")

            # Run all algorithms with progress tracking
            print("\n🔍 Running Tabu Search trials...")
            results.update(self.run_trials("tabu", TabuSearchCVRP, load_tabu, tabu_configs))

            print("\n🔍 Running Simulated Annealing trials...")
            results.update(self.run_trials("sa", SimulatedAnnealingCVRP, load_sa, sa_configs))

            print("\n🔍 Running Genetic Algorithm trials...")
            results.update(self.run_trials("ga", GeneticAlgorithmCVRP, load_ga, ga_configs))

            return results

    if __name__ == "__main__":
        # File paths - adjust as needed
        easy_file = "Data/A-n32-k5.vrp"
        hard_file = "Data/A-n60-k9.vrp"

        print("🚀 Starting parameter tuning experiment...")
        start_time = datetime.now()

        tuner = ParameterTuner(easy_file, hard_file)
        tuning_results = tuner.run_all()

        # Process and save results
        df = pd.DataFrame.from_dict(tuning_results, orient='index')
        df.index.name = "Configuration"

        # Sort by algorithm and performance
        df = df.sort_index()

        # Generate timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save to Excel with formatting
        excel_path = f"parameter_tuning_results_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results')
            # Add summary sheet
            df.describe().to_excel(writer, sheet_name='Summary')

        # Save to CSV
        csv_path = f"parameter_tuning_results_{timestamp}.csv"
        df.to_csv(csv_path)

        # Calculate duration
        duration = datetime.now() - start_time

        # Print final summary
        print("\n=== Parameter Tuning Complete ===")
        print(f"⏱️  Total duration: {duration}")
        print(f"📊 Results summary:")
        print(df.round(2).to_string())
        print(f"\n💾 Results saved to:")
        print(f"- {os.path.abspath(excel_path)}")
        print(f"- {os.path.abspath(csv_path)}")