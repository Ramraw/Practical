import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from deap import base, creator, tools, algorithms

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Define the evaluation function
def evaluate(individual):
    # Unpack hyperparameters from the individual
    n_estimators, max_depth = individual
    # Create a Random Forest model with the given hyperparameters
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    # Fit the model on the training data
    model.fit(X_train, y_train)
    # Return the accuracy score as the fitness value
    return model.score(X_test, y_test),

# Set up the genetic algorithm
creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # Maximize fitness
creator.create('Individual', list, fitness=creator.FitnessMax)  # Define individual as a list

toolbox = base.Toolbox()
# Define hyperparameter ranges
toolbox.register('n_estimators', np.random.randint, 10, 200)  # Random integers for n_estimators
toolbox.register('max_depth', np.random.randint, 1, 20)  # Random integers for max_depth
# Create an individual with the defined hyperparameters
toolbox.register('individual', tools.initCycle, creator.Individual, (toolbox.n_estimators, toolbox.max_depth), n=1)
# Create a population of individuals
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate)  # Register the evaluation function
toolbox.register('mate', tools.cxTwoPoint)  # Crossover operator
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)  # Mutation operator
toolbox.register('select', tools.selTournament, tournsize=3)  # Selection operator

# Run the genetic algorithm
population = toolbox.population(n=50)  # Create an initial population of 50 individuals
for gen in range(10):  # Run for a specified number of generations
    # Evaluate the individuals
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit  # Assign fitness values to individuals
    # Select the next generation individuals
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))  # Clone selected individuals
    # Apply crossover and mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.rand() < 0.5:  # Randomly decide whether to mate
            toolbox.mate(child1, child2)  # Perform crossover
            del child1.fitness.values  # Clear fitness values to re-evaluate
            del child2.fitness.values
    for mutant in offspring:
        if np.random.rand() < 0.2:  # Randomly decide whether to mutate
            toolbox.mutate(mutant)  # Perform mutation
            del mutant.fitness.values  # Clear fitness values to re-evaluate
    # Replace the old population by the offspring
    population[:] = offspring

# Best individual
best_individual = tools.selBest(population, 1)[0]  # Select the best individual
print('Best hyperparameters:', best_individual)  # Print the best hyperparameters