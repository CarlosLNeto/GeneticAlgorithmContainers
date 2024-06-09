import random
from fitness import calculate_fitness
from utils import generate_initial_population, select_parents, crossover, mutate
from container import Container, Balsa


def genetic_algorithm(population_size, generations):
    population = generate_initial_population(population_size)
    loading_sequences = [generate_loading_sequence() for _ in range(population_size)]

    for generation in range(generations):
        fitnesses = [
            calculate_fitness(balsa, seq)
            for balsa, seq in zip(population, loading_sequences)
        ]
        new_population = []
        new_loading_sequences = []

        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2), crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])

            seq1, seq2 = crossover_sequences(
                loading_sequences[population.index(parent1)],
                loading_sequences[population.index(parent2)],
            )
            new_loading_sequences.extend([seq1, seq2])

        population = new_population
        loading_sequences = new_loading_sequences

    best_balsa = max(
        population,
        key=lambda b: calculate_fitness(b, loading_sequences[population.index(b)]),
    )
    best_sequence = loading_sequences[population.index(best_balsa)]
    return best_balsa, best_sequence


def generate_loading_sequence():
    containers = [Container(i) for i in range(30)]
    random.shuffle(containers)
    sequence = []
    for i, container in enumerate(containers):
        layer = 1 if i < 12 else 2 if i < 24 else 3
        sequence.append((container, layer))
    return sequence


def crossover_sequences(seq1, seq2):
    mid = len(seq1) // 2
    new_seq1 = seq1[:mid] + seq2[mid:]
    new_seq2 = seq2[:mid] + seq1[mid:]
    return new_seq1, new_seq2
