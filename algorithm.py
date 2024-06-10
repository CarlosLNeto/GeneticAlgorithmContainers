import numpy as np
import random

# Definir constantes
NUM_CONTAINERS = 30
POPULATION_SIZE = 100
GENERATIONS = 1000
MUTATION_RATE = 0.01
ELITISM_RATE = 0.05
CONTAINERS_PER_FLOOR = [12, 12, 6]
PORT_ROWS = 3
PORT_COLS = 5


# Função de fitness
def fitness(solution):
    stability = calculate_stability(solution)
    movements = calculate_movements(solution)
    floating_penalty = calculate_floating_penalty(solution)
    return 1 / (stability + movements + floating_penalty + 1)


def calculate_stability(solution):
    floor_weights = [0] * len(CONTAINERS_PER_FLOOR)
    for position in solution:
        if position < 12:
            floor_weights[0] += 1
        elif position < 24:
            floor_weights[1] += 1
        else:
            floor_weights[2] += 1
    variance = np.var(floor_weights)
    return variance


def calculate_movements(solution):
    initial_position = list(range(NUM_CONTAINERS))
    movements = 0
    for i, container in enumerate(solution):
        if initial_position[i] != container:
            movements += 1
    return movements


def calculate_floating_penalty(solution):
    penalty = 0
    floor_indices = [
        sum(CONTAINERS_PER_FLOOR[:i]) for i in range(len(CONTAINERS_PER_FLOOR) + 1)
    ]
    for floor in range(1, len(CONTAINERS_PER_FLOOR)):
        for i in range(floor_indices[floor], floor_indices[floor + 1]):
            if solution[i] < floor_indices[floor - 1]:
                penalty += 1
    return penalty


# Inicializar a população
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = list(np.random.permutation(NUM_CONTAINERS))
        population.append(individual)
    return population


# Seleção por roleta
def roulette_wheel_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    selected_indices = np.random.choice(len(population), 2, p=selection_probs)
    return population[selected_indices[0]], population[selected_indices[1]]


# Operador de cruzamento de ciclo
def crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = [None] * size, [None] * size
    cycle_start = random.randint(0, size - 1)

    idx = cycle_start
    while True:
        child1[idx] = parent1[idx]
        child2[idx] = parent2[idx]
        idx = parent1.index(parent2[idx])
        if idx == cycle_start:
            break

    for i in range(size):
        if child1[i] is None:
            child1[i] = parent2[i]
        if child2[i] is None:
            child2[i] = parent1[i]

    return child1, child2


# Operador de mutação
def mutate(individual):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(NUM_CONTAINERS), 2)
        individual[i], individual[j] = individual[j], individual[i]


# Algoritmo Genético com Elitismo e Seleção por Roleta
def genetic_algorithm():
    population = initialize_population()
    best_solution = None
    best_fitness = float("-inf")

    elitism_count = int(POPULATION_SIZE * ELITISM_RATE)

    for generation in range(GENERATIONS):
        fitnesses = [fitness(individual) for individual in population]

        # Preservar os melhores indivíduos (elitismo)
        new_population = [
            population[idx] for idx in np.argsort(fitnesses)[-elitism_count:]
        ]

        # Gerar nova população através de seleção, cruzamento e mutação
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = roulette_wheel_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])

        population = new_population[:POPULATION_SIZE]

        current_best_fitness = max(fitnesses)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[fitnesses.index(best_fitness)]

        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness


# Coordenadas de origem e destino
def get_coordinates(port_position, barge_position):
    # Coordenadas no porto
    port_floor = 1 if port_position < 15 else 0
    port_row = (port_position % 15) // PORT_COLS
    port_col = (port_position % 15) % PORT_COLS

    # Coordenadas na balsa
    barge_floor = 0
    remaining = barge_position
    for floor, count in enumerate(CONTAINERS_PER_FLOOR):
        if remaining < count:
            barge_floor = floor
            break
        remaining -= count
    barge_slot = remaining
    barge_row = barge_slot // 4  # Assumindo 4 colunas por andar na balsa
    barge_col = barge_slot % 4
    return (port_floor, port_row, port_col), (barge_floor, barge_row, barge_col)


# Executar o algoritmo genético
best_solution, best_fitness = genetic_algorithm()

# Mostrar coordenadas de origem e destino na ordem da melhor solução encontrada
for container in best_solution:
    port_coord, barge_coord = get_coordinates(container, best_solution.index(container))
    print(f"Container {container}: Porto {port_coord} -> Balsa {barge_coord}")

print("Melhor solução encontrada:", best_solution)
print("Fitness da melhor solução encontrada:", best_fitness)
