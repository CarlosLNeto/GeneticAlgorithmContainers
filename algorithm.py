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
    floating_penalty = calculate_floating_penalty(solution)
    return 1 / (stability + movements + floating_penalty + 1)


def calculate_stability(solution):
    left_right_balance = [0] * len(CONTAINERS_PER_FLOOR)
    front_back_balance = [0] * len(CONTAINERS_PER_FLOOR)

    for position in solution:
        floor = 0
        if position < 12:
            floor = 0
        elif position < 24:
            floor = 1
        else:
            floor = 2

        rows, cols = FLOOR_SHAPE[floor]
        row = (position % (rows * cols)) // cols
        col = (position % (rows * cols)) % cols

        if cols > 1:
            if col < cols // 2:
                left_right_balance[floor] += 1
            else:
                left_right_balance[floor] -= 1

        if rows > 1:
            if row < rows // 2:
                front_back_balance[floor] += 1
            else:
                front_back_balance[floor] -= 1

    left_right_penalty = sum(abs(balance) for balance in left_right_balance)
    front_back_penalty = sum(abs(balance) for balance in front_back_balance)

    return left_right_penalty + front_back_penalty


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


# Função principal do Algoritmo Genético com a nova lógica de estabilidade
def genetic_algorithm():
    population = initialize_population()
    best_solution = None
    best_fitness = float("-inf")

    elitism_count = int(POPULATION_SIZE * ELITISM_RATE)

    best_fitnesses = []
    generation_best_solutions = []

    fig, ax = plt.subplots()
    ax.set_xlim(0, GENERATIONS)
    ax.set_ylim(
        0.02, 0.1
    )  # Ajustar os limites do eixo y para focar no intervalo relevante
    ax.set_yticks(np.arange(0, 0.11, 0.01))  # Ajustar para incrementos de 0,01
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

    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Genetic Algorithm Evolution")
    plt.show()

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
    barge_row = barge_slot // 3  # Assumindo 3 colunas por andar na balsa
    barge_col = barge_slot % 3
    return (port_floor, port_row, port_col), (barge_floor, barge_row, barge_col)


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


# Função para exibir as matrizes dos andares do navio
def display_ship_floors(solution):
    floors = []
    for rows, cols in FLOOR_SHAPE:
        floors.append(np.zeros((rows, cols), dtype=int))

    for i, container in enumerate(solution):
        barge_position = solution.index(container)
        barge_floor = 0
        remaining = barge_position
        for floor, count in enumerate(CONTAINERS_PER_FLOOR):
            if remaining < count:
                barge_floor = floor
                break
            remaining -= count
        barge_slot = remaining
        barge_row = barge_slot // floors[barge_floor].shape[1]
        barge_col = barge_slot % floors[barge_floor].shape[1]
        floors[barge_floor][barge_row, barge_col] = i + 1  # Ordem de transporte

    for floor_idx, floor in enumerate(floors):
        print(f"Andar {floor_idx + 1}:")
        print(floor)
        print()


# Executar o algoritmo genético
best_solution, best_fitness = genetic_algorithm()

# Mostrar coordenadas de origem e destino na ordem da melhor solução encontrada
for container in best_solution:
    port_coord, barge_coord = get_coordinates(container, best_solution.index(container))
    print(f"Container {container}: Porto {port_coord} -> Balsa {barge_coord}")

print("Melhor solução encontrada:", best_solution)
print("Fitness da melhor solução encontrada:", best_fitness)

# Exibir as posições dos contêineres no píer antes do transporte
display_pier_positions()

# Exibir as matrizes dos andares do navio
display_ship_floors(best_solution)
