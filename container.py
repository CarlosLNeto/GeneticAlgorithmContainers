class Container:
    def __init__(self, id):
        self.id = id


class Balsa:
    def __init__(self):
        self.layers = {1: [], 2: [], 3: []}

    def add_container(self, container, layer):
        self.layers[layer].append(container)

    def is_valid(self):
        for container in self.layers[2]:
            if not any(c.id == container.id for c in self.layers[1]):
                return False
        for container in self.layers[3]:
            if not any(c.id == container.id for c in self.layers[2]):
                return False
        return True

    def get_containers(self):
        return [container for layer in self.layers.values() for container in layer]

    def check_loading_order(self, loading_sequence):
        loaded = {1: set(), 2: set(), 3: set()}
        for step in loading_sequence:
            container, layer = step
            if layer == 2 and container.id not in loaded[1]:
                return False
            if layer == 3 and container.id not in loaded[2]:
                return False
            loaded[layer].add(container.id)
        return True
