import numpy as np
import prognosticcriteria
import random
import pandas as pd

#initiealize lambda vales
lambda_vales = [random.uniform(0, 1) for _ in range(3)]

# Define the objective function to maximize
def objective_function(weights, lambda_values, degradation_data):
    # Calculate Mon(H1:m), Tre(H1:m), Rob(H1:m) based on degradation_data
    mon_value = prognosticcriteria.Mo(degradation_data)
    tre_value = prognosticcriteria.Tr(degradation_data)
    pr_value = prognosticcriteria.Pr(degradation_data)

    # Weighted sum of the three properties
    j_value = lambda_values[0] * mon_value + lambda_values[1] * tre_value + lambda_values[2] * pr_value

    # Additionally, incorporate the constraint hm = 1 if necessary

    return j_value


def mutation(population, lambda_values, degradation_data):
    mutated_population = []
    for individual in population:
        # Find the best, suboptimal, and worst individuals in the population
        sorted_population = sorted(population,
                                   key=lambda x: objective_function(x[:-2], lambda_values, degradation_data),
                                   reverse=True)
        omega_j1, omega_j2, omega_j3 = sorted_population[0], sorted_population[1], sorted_population[-1]

        # Calculate mutation factor
        mutation_factor = np.random.uniform(0.1, 0.9)

        # Construct new vector ρv_i
        mutated_individual = omega_j1 + mutation_factor * (omega_j2 - omega_j3)

        mutated_population.append(mutated_individual)

    return mutated_population
def crossover(mutated_individual, individual):
    # Initialize trial vector
    trial_vector = np.copy(individual)

    # Perform crossover
    crossover_factor = np.random.uniform(0, 0.9)
    krand = np.random.randint(len(individual) - 2)  # Random index for crossover
    for k in range(len(individual) - 2):
        if np.random.rand() <= crossover_factor or k == krand:
            trial_vector[k] = mutated_individual[k]
        if crossover_factor < np.random.rand() <= 0.9:
            trial_vector[k] = trial_vector[k]
        #else:
            #trial_vector[k] = omega_j1

    return trial_vector

# SADE Algorithm
def sade_optimization(degradation_data, lambda_values, max_iterations):
    # Initialize population
    population_size = 50
    population = np.random.uniform(low=0, high=1, size=(population_size, len(degradation_data) + 2))

    # Initialize mutation and crossover factors
    mutation_factor = np.random.uniform(0.1, 0.9)
    crossover_factor = np.random.uniform(0, 0.9)

    # Cumulative improvement times for each individual
    cumulative_improvement_times = [0] * population_size

    # Main optimization loop
    for iteration in range(max_iterations):
        for idx, individual in enumerate(population):
            # Perform mutation
            mutated_individual = mutation(population, lambda_values, degradation_data)

            # Perform crossover
            trial_vector = crossover(mutated_individual, individual)

            # Evaluate trial vector
            trial_vector_fitness = objective_function(trial_vector[:-2], lambda_values, degradation_data)
            current_individual_fitness = objective_function(individual[:-2], lambda_values, degradation_data)

            # Select better individual
            if trial_vector_fitness > current_individual_fitness:
                population[idx] = trial_vector
                cumulative_improvement_times[idx] += 1
            else:
                cumulative_improvement_times[idx] = 0

        # Update mutation and crossover factors
        best_individual_idx = np.argmax([objective_function(individual[:-2], lambda_values, degradation_data) for individual in population])
        best_individual = population[best_individual_idx]
        mutation_factor = mutation_factor + (1 - cumulative_improvement_times[best_individual_idx] / max(cumulative_improvement_times)) * (best_individual[0] - mutation_factor)
        crossover_factor = crossover_factor + (1 - cumulative_improvement_times[best_individual_idx] / max(cumulative_improvement_times)) * (best_individual[1] - crossover_factor)

        # Check termination criteria
        if max(cumulative_improvement_times) >= 5:  # Example termination criterion
            break

    # Return optimal solution
    optimal_solution = population[np.argmax([objective_function(individual[:-2], lambda_values, degradation_data) for individual in population])]
    return optimal_solution



# Example usage
degradation_data = np.array([...])  # Insert your degradation data here
lambda_values = [0.3, 0.3, 0.4]  # Example lambda values
max_iterations = 1000

degradation_data = pd.read_csv(r"C:\Users\geort\Desktop\Universty\PZT-CSV-L1-04\L104_2019_12_11_15_29_20\State_1_2019_12_11_15_29_20\050kHz-Features.csv")
optimal_solution = sade_optimization(degradation_data, lambda_values, max_iterations)
print("Optimal weights:", optimal_solution)