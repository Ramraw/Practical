import random
import string

# Function to generate a population of random strings
def gen_pop(length, pop_len):
    pop = []
    for _ in range(pop_len):
        # Generate a random string of specified length using lowercase letters
        pop.append(''.join(random.choices(string.ascii_lowercase, k=length)))
    return pop

# Function to calculate the fitness of a string compared to the target word
def fitness(input_str, target_word):
    score = 0
    # Compare each character in the input string with the target word
    for i in range(len(input_str)):
        if input_str[i] == target_word[i]:
            score += 1  # Increment score for each matching character
    # Return the fitness score as a fraction of the total length
    return score / len(input_str)

# Function to evaluate the fitness of each member in the population
def eval(input_str, population):
    fit_pop = {}
    for individual in population:
        fit_pop[individual] = fitness(input_str, individual)  # Store fitness scores
    return fit_pop

# Function to select a breeding population based on fitness scores
def BPop(fit_pop):
    population = list(fit_pop.keys())
    fit = list(fit_pop.values())
    # Select members based on their fitness scores
    members = random.choices(population, fit, k=20)
    b_pop = {}
    for member in members:
        b_pop[member] = fit_pop[member]  # Store selected members and their fitness
    return b_pop

# Function to create a new population through crossover of selected parents
def create_new_population(breeding_population):
    new_population = []
    BPMem = list(breeding_population.keys())
    # Iterate through the breeding population in pairs
    for i in range(0, len(breeding_population), 2):
        parent1 = BPMem[i]
        parent2 = BPMem[i + 1 if i + 1 < len(breeding_population) else 0]
        child1, child2 = '', ''
        # Perform crossover to create two children
        for j in range(len(parent1)):
            if random.random() < 0.5:
                child1 += parent1[j]  # Take character from parent1
                child2 += parent2[j]  # Take character from parent2
            else:
                child1 += parent2[j]  # Take character from parent2
                child2 += parent1[j]  # Take character from parent1
        new_population.append(child1)  # Add child1 to the new population
        new_population.append(child2)  # Add child2 to the new population
    return new_population

# Function to check if the stopping condition is met
def StopC(new_pop, threshold=0.8):
    max_score = max(list(new_pop.values()))  # Get the maximum fitness score
    return max_score >= threshold  # Return True if the score meets the threshold

# Main function to run the genetic algorithm
def GenAlgo():
    input_word = input("Enter Word: ")  # Get the target word from the user
    population = gen_pop(len(input_word), 100)  # Generate initial population
    pop = population
    generation = 0

    while True:
        generation += 1
        pop = eval(input_word, pop)  # Evaluate the fitness of the population
        # Check if the stopping condition is met
        if StopC(pop, threshold=0.8):
            members = list(pop.keys())
            fit = list(pop.values())
            print(f"Word guessed or threshold met in generation {generation}")
            best_match = members[fit.index(max(fit))]  # Find the best matching word
            print(f"Best matching word: {best_match} with fitness {max(fit)}")
            break  # Exit the loop if the condition is met
        breeding_population = BPop(pop)  # Select breeding population
        pop = create_new_population(breeding_population)  # Create new population

# Run the genetic algorithm
GenAlgo()