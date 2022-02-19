import numpy as np


class Node:
    def __init__(self, state) -> None:
        self.state = state


class State:
    def __init__(self, x, y, z) -> None:
        self.l = np.array([x, y, z])
