import random
from container import Container, Balsa


def generate_initial_population(size):
    population = []
    for _ in range(size):
        balsa = Balsa()
        containers = [Container(i) for i in range(30)]
        random.shuffle(containers)
        for i, container in enumerate(containers):
            if i < 12:
                balsa.add_container(container, 1)
            elif i < 24:
                balsa.add_container(container, 2)
            else:
                balsa.add_container(container, 3)
        population.append(balsa)
    return population


def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [fitness / total_fitness for fitness in fitnesses]
    parent1 = random.choices(population, weights=selection_probs, k=1)[0]
    parent2 = random.choices(population, weights=selection_probs, k=1)[0]
    return parent1, parent2


def crossover(parent1, parent2):
    child = Balsa()
    for layer in child.layers:
        parent_layer_containers = parent1.layers[layer] + parent2.layers[layer]
        random.shuffle(parent_layer_containers)
        child.layers[layer] = parent_layer_containers[: len(parent1.layers[layer])]
    return child


def mutate(balsa):
    for layer in balsa.layers:
        if len(balsa.layers[layer]) > 1:
            i, j = random.sample(range(len(balsa.layers[layer])), 2)
            balsa.layers[layer][i], balsa.layers[layer][j] = (
                balsa.layers[layer][j],
                balsa.layers[layer][i],
            )
