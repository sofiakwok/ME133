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

    def connectsTo(self, state1, state2, freq=10):
        checkpoints = np.arange(0, 1, 1 / (state1.distance(state2) * freq))
        for c in checkpoints:
            if not self.freespace(state1.intermediate(state2, c)):
                return False
        return True

        # mins = np.minimum(state1.l, state2.l)
        # maxs = np.maximum(state1.l, state2.l)
        # distances = maxs - mins

        # checkpoints = np.array([])
        # for i in range(len(distances)):
        #     np.concatenate(
        #         checkpoints,
        #         np.arange(np.floor(mins[i]), np.ceil(maxs[i])) / distances[i])

        #     nc = np.arange(np.floor(mins[i]), np.ceil(maxs[i]))
        #     if (state1.l[i] < state2.l[i]):
        #         nc -= mins[i]
        #     else:
        #         pass
        #     )

    def connectsToVoxel(self, state, x, y, z, maxdist=25, freq=10):
        for ox, oy, oz in [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]:
            vstate = State(x + ox, y + oy, z + oz)
            if (state.distance(vstate)) < 1 / freq:
                return True
            if (state.distance(vstate)) > maxdist:
                return False

            checkpoints = np.arange(0, 1, 1 / (state.distance(vstate) * freq))
            for i in range(len(checkpoints)):
                intermediate = state.intermediate(vstate, checkpoints[i])
                if i == len(checkpoints) - 1 or np.array_equal(
                    np.floor(intermediate.l), np.array([x, y, z])
                ):
                    return True
                if not self.freespace(state.intermediate(vstate, checkpoints[i])):
                    break

        return False

    def visibleFrom(self, state):
        init_world = np.zeros(self.world.shape, dtype=bool)
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                for k in range(self.world.shape[2]):
                    init_world[i, j, k] = self.world[i, j, k] and self.connectsToVoxel(
                        state, i, j, k
                    )

        return World(init_world, colors=self.colors)

    def combine(self, newWorld):
        combined = np.logical_or(self.world, newWorld.world)
        return World(combined, self.colors)
