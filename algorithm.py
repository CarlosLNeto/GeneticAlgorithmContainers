import numpy as np
import random

# Definir constantes
NUM_CONTAINERS = 30
POPULATION_SIZE = 100
GENERATIONS = 1000
MUTATION_RATE = 0.01
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
    floor_weights = [
        sum(
            1
            for i in range(
                sum(CONTAINERS_PER_FLOOR[:floor]),
                sum(CONTAINERS_PER_FLOOR[: floor + 1]),
            )
        )
        for floor in range(len(CONTAINERS_PER_FLOOR))
    ]
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
        second_floor_containers = list(np.random.permutation(15))
        first_floor_containers = list(np.random.permutation(15) + 15)
        individual = second_floor_containers + first_floor_containers
        population.append(individual)
    return population


# Seleção por torneio
def tournament_selection(population, fitnesses):
    selected = []
    for _ in range(2):
        participants = random.sample(list(zip(population, fitnesses)), 3)
        selected.append(max(participants, key=lambda x: x[1])[0])
    return selected


# Operador de cruzamento de um ponto
def crossover(parent1, parent2):
    point = random.randint(1, 14)  # Garantir cruzamento dentro do segundo andar
    child1 = parent1[:point] + [x for x in parent2 if x not in parent1[:point]]
    child2 = parent2[:point] + [x for x in parent1 if x not in parent2[:point]]
    return child1, child2


# Operador de mutação
def mutate(individual):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(NUM_CONTAINERS), 2)
        if (
            i < 15 and j < 15 or i >= 15 and j >= 15
        ):  # Garantir mutação dentro do mesmo andar
            individual[i], individual[j] = individual[j], individual[i]


# Algoritmo Genético
def genetic_algorithm():
    population = initialize_population()
    best_solution = None
    best_fitness = float("-inf")

    for generation in range(GENERATIONS):
        fitnesses = [fitness(individual) for individual in population]

        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])

        population = new_population

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
    port_coord, barge_coord = get_coordinates(container, best_solution[container])
    print(f"Container {container}: Porto {port_coord} -> Balsa {barge_coord}")

print("Melhor solução encontrada:", best_solution)
print("Fitness da melhor solução encontrada:", best_fitness)
