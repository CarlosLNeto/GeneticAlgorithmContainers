from genetic_algorithm import genetic_algorithm
from fitness import calculate_fitness


def main():
    population_size = 100
    generations = 50
    best_solution, best_sequence = genetic_algorithm(population_size, generations)
    best_fitness = calculate_fitness(best_solution, best_sequence)

    print("Best solution found:")
    for layer in best_solution.layers:
        print(
            f"Layer {layer}: {[container.id for container in best_solution.layers[layer]]}"
        )

    print("Best loading sequence:")
    for step in best_sequence:
        container, layer = step
        print(f"Container {container.id} -> Layer {layer}")

    print(f"Best fitness score: {best_fitness}")


if __name__ == "__main__":
    main()
