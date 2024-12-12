import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random

# 1. Define fuzzy variables
diversity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'diversity')
convergence = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'convergence')
mutation_rate = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'mutation_rate')
crossover_rate = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'crossover_rate')

# 2. Define membership functions for fuzzy variables
diversity['low'] = fuzz.trimf(diversity.universe, [0, 0, 0.5])
diversity['high'] = fuzz.trimf(diversity.universe, [0.5, 1, 1])
convergence['slow'] = fuzz.trimf(convergence.universe, [0, 0, 0.5])
convergence['fast'] = fuzz.trimf(convergence.universe, [0.5, 1, 1])
mutation_rate['low'] = fuzz.trimf(mutation_rate.universe, [0, 0, 0.5])
mutation_rate['high'] = fuzz.trimf(mutation_rate.universe, [0.5, 1, 1])
crossover_rate['low'] = fuzz.trimf(crossover_rate.universe, [0, 0, 0.5])
crossover_rate['high'] = fuzz.trimf(crossover_rate.universe, [0.5, 1, 1])

# 3. Define fuzzy rules with additional coverage
rule1 = ctrl.Rule(diversity['low'] & convergence['fast'], (mutation_rate['low'], crossover_rate['high']))
rule2 = ctrl.Rule(diversity['high'] & convergence['slow'], (mutation_rate['high'], crossover_rate['low']))
rule3 = ctrl.Rule(diversity['low'] & convergence['slow'], (mutation_rate['high'], crossover_rate['low']))
rule4 = ctrl.Rule(diversity['high'] & convergence['fast'], (mutation_rate['low'], crossover_rate['high']))
rule5 = ctrl.Rule(diversity['low'] & convergence['slow'], (mutation_rate['high'], crossover_rate['high']))
rule6 = ctrl.Rule(diversity['high'] & convergence['slow'], (mutation_rate['low'], crossover_rate['low']))

# Control system creation and simulation
fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

# 4. Define a simple fitness function (depends on the problem)
def fitness(solution):
    return sum(solution)

# 5. Initialize population
def create_population(size, solution_length):
    return [np.random.randint(0, 2, solution_length) for _ in range(size)]

# Genetic algorithm parameters
population_size = 50
generations = 20
solution_length = 10

# GA main loop
population = create_population(population_size, solution_length)
for gen in range(generations):
    fitness_values = [fitness(individual) for individual in population]
    avg_fitness = np.mean(fitness_values)
    diversity_value = np.std(fitness_values) / avg_fitness if avg_fitness > 0 else 0
    convergence_value = (generations - gen) / generations
    fuzzy_sim.input['diversity'] = diversity_value
    fuzzy_sim.input['convergence'] = convergence_value
    fuzzy_sim.compute()
    mutation_chance = fuzzy_sim.output.get('mutation_rate', 0.1)
    crossover_chance = fuzzy_sim.output.get('crossover_rate', 0.7)
    print(f"Generation {gen+1}, Diversity: {diversity_value:.2f}, Convergence: {convergence_value:.2f}")
    print(f"Mutation Rate: {mutation_chance:.2f}, Crossover Rate: {crossover_chance:.2f}")
    selected_parents = random.choices(population, k=population_size)
    next_population = []
    for i in range(0, population_size, 2):
        parent1, parent2 = selected_parents[i], selected_parents[(i+1) % population_size]
        if random.random() < crossover_chance:
            crossover_point = random.randint(1, solution_length - 1)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            next_population.extend([child1, child2])
        else:
            next_population.extend([parent1, parent2])
    for individual in next_population :
        for gene in range(solution_length):
            if random.random() < mutation_chance:
                individual[gene] = 1 - individual[gene]
    population = next_population

# Final output after all generations
best_solution = max(population, key=fitness)
print("Best solution found:", best_solution)
print("Best solution fitness:", fitness(best_solution))