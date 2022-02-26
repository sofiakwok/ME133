import numpy as np


class Node:
    def __init__(self, state, parent=None) -> None:
        self.state = state
        self.parent = parent
        self.children = []

    def addChild(self, node):
        self.children.append(node)
        node.parent = self

    def distance(self, other):
        return self.state.distance(other.state)


class State:
    def __init__(self, x, y, z) -> None:
        self.l = np.array([x, y, z])

    def distance(self, other):
        return np.sqrt(np.sum((self.l - other.l) ** 2))

    def intermediate(self, other, prop):
        nl = self.l * (1 - prop) + other.l * prop
        return State(nl[0], nl[1], nl[2])
