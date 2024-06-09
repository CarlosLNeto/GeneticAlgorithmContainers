def calculate_fitness(balsa, loading_sequence):
    stability_score = calculate_stability(balsa)
    movement_penalty = calculate_movements(balsa)
    placement_penalty = calculate_placement_penalty(balsa)
    order_penalty = calculate_loading_order_penalty(balsa, loading_sequence)
    total_fitness = (
        stability_score - movement_penalty - placement_penalty - order_penalty
    )
    return total_fitness


def calculate_stability(balsa):
    ideal_weight = len(balsa.get_containers()) / 3
    layer_weights = [len(balsa.layers[layer]) for layer in balsa.layers]
    stability_score = -sum(abs(weight - ideal_weight) for weight in layer_weights)
    return stability_score


def calculate_movements(balsa):
    total_movements = len(balsa.get_containers())
    return total_movements * 1  # Cada movimento adiciona 1 ao movimento penalidade


def calculate_placement_penalty(balsa):
    penalty = 0
    for layer in [2, 3]:
        for container in balsa.layers[layer]:
            position_in_previous_layer = any(
                c.id == container.id for c in balsa.layers[layer - 1]
            )
            if not position_in_previous_layer:
                penalty += 10  # Penalidade ajustada para 10

    for container in balsa.layers[2]:
        if not any(c.id == container.id for c in balsa.layers[1]):
            penalty += 10
    for container in balsa.layers[3]:
        if not any(c.id == container.id for c in balsa.layers[2]):
            penalty += 10

    return penalty


def calculate_loading_order_penalty(balsa, loading_sequence):
    if not balsa.check_loading_order(loading_sequence):
        return 50  # Penalidade ajustada para 50
    return 0
