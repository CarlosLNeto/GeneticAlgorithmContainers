import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Definir constantes
NUM_CONTAINERS = 30
POPULATION_SIZE = 100
GENERATIONS = 1000
MUTATION_RATE = 0.01
ELITISM_RATE = 0.05
CONTAINERS_PER_FLOOR = [12, 12, 6]
PORT_ROWS = 5
PORT_COLS = 3
FLOOR_SHAPE = [(4, 3), (4, 3), (4, 3)]  # Configuração para 4x3


# Função de fitness
def fitness(solution):
    stability = calculate_stability(solution)
    movements = calculate_movements(solution)
    floor_violation = calculate_floor_violation_penalty(solution)
    return 1 / (stability + movements + floor_violation)


def calculate_stability(solution):
    stability_penalty = 0

    def center_of_mass(floor_containers):
        if len(floor_containers) == 0:
            return (0, 0)
        rows, cols = FLOOR_SHAPE[0]  # Assumindo que todos os andares têm a mesma forma
        x_coords = [c // cols for c in floor_containers]
        y_coords = [c % cols for c in floor_containers]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    def distance_from_center_of_mass(center, rows, cols):
        center_x, center_y = center
        ship_center_x, ship_center_y = rows / 2, cols / 2
        return (
            (center_x - ship_center_x) ** 2 + (center_y - ship_center_y) ** 2
        ) ** 0.5

    floor_indices = [
        sum(CONTAINERS_PER_FLOOR[:i]) for i in range(len(CONTAINERS_PER_FLOOR) + 1)
    ]

    for i, (start, end) in enumerate(zip(floor_indices[:-1], floor_indices[1:])):
        floor_containers = [solution.index(j) for j in range(start, end)]
        center = center_of_mass(floor_containers)
        rows, cols = FLOOR_SHAPE[i]
        stability_penalty += distance_from_center_of_mass(center, rows, cols)

    return stability_penalty


def calculate_movements(solution):
    movements = 0
    for port_position, barge_position in enumerate(solution):
        port_coords, barge_coords = get_coordinates(port_position, barge_position)
        movements += manhattan_distance(port_coords, barge_coords)
        movements += abs(port_coords[0] - barge_coords[0])  # Movimentos de subida
        movements += 1  # Movimento de descida
    return movements


def manhattan_distance(coord1, coord2):
    return abs(coord1[1] - coord2[1]) + abs(coord1[2] - coord2[2])


def calculate_floor_violation_penalty(solution):
    penalty = 0
    floor_indices = [
        sum(CONTAINERS_PER_FLOOR[:i]) for i in range(len(CONTAINERS_PER_FLOOR) + 1)
    ]
    for i in range(15, 30):  # Containers do primeiro andar (port positions 15 a 29)
        if (
            solution.index(i) < 15
        ):  # Se estiver entre os primeiros 15 containers no barco
            penalty += 1000  # Penalidade alta para quebra da regra
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


def genetic_algorithm():
    population = initialize_population()
    best_solution = None
    best_fitness = float("-inf")

    elitism_count = int(POPULATION_SIZE * ELITISM_RATE)

    best_fitnesses = []
    generation_best_solutions = []

    fig, ax = plt.subplots()
    ax.set_xlim(50, GENERATIONS)
    ax.set_ylim(
        0.007, 0.012
    )  # Ajustar os limites do eixo y para focar no intervalo relevante
    ax.set_yticks(
        np.arange(0.007, 0.013, 0.001)
    )  # Ajustar para incrementos de 0,01 # Ajustar os limites do eixo y para focar no intervalo relevante
    (line,) = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        nonlocal population, best_solution, best_fitness

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

        best_fitnesses.append(best_fitness)
        generation_best_solutions.append(best_solution)

        line.set_data(range(len(best_fitnesses)), best_fitnesses)
        return (line,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(GENERATIONS),
        init_func=init,
        blit=True,
        repeat=False,
        interval=2,
    )

    plt.xlabel("Gerações")
    plt.ylabel("Melhor Fitness")
    plt.title("Evolução do Algoritmo genético")
    plt.show()

    return best_solution, best_fitness


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
    if barge_floor == 0:
        order = [
            (1, 1),
            (1, 2),
            (1, 0),
            (2, 1),
            (2, 2),
            (2, 0),
            (0, 1),
            (0, 2),
            (0, 0),
            (3, 1),
            (3, 2),
            (3, 0),
        ]
    elif barge_floor == 1:
        order = [
            (1, 1),
            (1, 2),
            (1, 0),
            (2, 1),
            (2, 2),
            (2, 0),
            (0, 1),
            (0, 2),
            (0, 0),
            (3, 1),
            (3, 2),
            (3, 0),
        ]
    else:
        order = [(1, 1), (1, 2), (1, 0), (2, 1), (2, 2), (2, 0)]

    if barge_slot >= len(order):
        raise IndexError(f"Invalid barge slot: {barge_slot}")

    return (port_floor, port_row, port_col), (barge_floor, *order[barge_slot])


# Função para exibir as matrizes do píer
def display_pier_positions():
    pier_floor_1 = np.zeros((PORT_ROWS, PORT_COLS), dtype=int)  # Primeiro andar do píer
    pier_floor_2 = np.zeros((PORT_ROWS, PORT_COLS), dtype=int)  # Segundo andar do píer

    for pos in range(30):
        row = (pos % 15) // PORT_COLS
        col = (pos % 15) % PORT_COLS
        if pos < 15:
            pier_floor_2[row, col] = pos + 1
        else:
            pier_floor_1[row, col] = pos + 1

    print("Posições no píer (Segundo andar):")
    print(pier_floor_2)
    print()
    print("Posições no píer (Primeiro andar):")
    print(pier_floor_1)
    print()


# Executar o algoritmo genético
best_solution, best_fitness = genetic_algorithm()

# Verificar se uma solução foi encontrada
if best_solution is not None:
    # Mostrar coordenadas de origem e destino na ordem da melhor solução encontrada
    for container in best_solution:
        port_coord, barge_coord = get_coordinates(
            container, best_solution.index(container)
        )
        print(f"Container {container + 1}: Porto {port_coord} -> Balsa {barge_coord}")

    for i in range(len(best_solution)):
        best_solution[i] = best_solution[i] + 1

    print("Melhor solução encontrada:", best_solution)
    print("Fitness da melhor solução encontrada:", best_fitness)
else:
    print("Nenhuma solução encontrada.")

# Exibir as posições dos contêineres no píer antes do transporte
display_pier_positions()
