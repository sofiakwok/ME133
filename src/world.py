from node import State

import numpy as np


class World:
    def __init__(self, init_world, colors=None) -> None:
        self.world = init_world
        self.size = np.array(init_world.shape) + 1

        if colors is not None:
            assert self.world.shape == colors.shape
            self.colors = colors
        else:
            self.colors = None

    def plot(self, ax):
        ax.voxels(self.world, facecolors=self.colors)

    def maxsize(self):
        return np.max(self.world.shape)

    def freespace(self, state):
        if np.any(state.l > self.size) or np.any(state.l) < 0:
            return False  # leaving the world is not allowed
        else:
            idx = state.l.astype(int)
            return not self.world[idx[0]][idx[1]][idx[2]]

    def visible_from(self, location):
        pass
