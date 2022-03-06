from node import State

from multiprocessing import Pool
from functools import partial

import numpy as np
import scipy
from scipy import ndimage


class World:
    def __init__(self, init_world, colors=None, seen=None, inaccessible=None) -> None:
        self.world = init_world
        self.size = np.array(init_world.shape)

        if colors is not None:
            assert self.world.shape == colors.shape
            self.colors = colors
        else:
            self.colors = None

        if seen is not None:
            assert self.world.shape == seen.shape
            self.seen = seen
        else:
            self.seen = np.zeros(self.world.shape, dtype=bool)

        if inaccessible is not None:
            assert self.world.shape == inaccessible.shape
            self.inaccessible = inaccessible
        else:
            self.inaccessible = np.zeros(self.world.shape, dtype=bool)

        weight_size = 15
        weight_size = weight_size if weight_size % 2 == 1 else weight_size + 1
        weight_center = int(weight_size / 2)
        self.distance_weights = np.zeros((weight_size, weight_size, weight_size))
        for i in range(weight_size):
            for j in range(weight_size):
                for k in range(weight_size):
                    if i == weight_center and j == weight_center and k == weight_center:
                        self.distance_weights[i, j, k] = 1
                    else:
                        self.distance_weights[i, j, k] = 1 / (
                            (i - weight_center) ** 2
                            + (j - weight_center) ** 2
                            + (k - weight_center) ** 2
                        )
        self.distance_weights = self.distance_weights / np.sum(self.distance_weights)

    def plot(self, ax, seen_only=True, draw_unseen=False):
        if seen_only:
            ax.voxels(np.logical_and(self.world, self.seen), facecolors=self.colors)
            if draw_unseen:
                ax.voxels(np.logical_and(self.world, np.logical_not(self.seen)))
        else:
            ax.voxels(self.world, facecolors=self.colors)

    def maxsize(self):
        return np.max(self.world.shape)

    def freespace(self, state, seen_only):
        if np.any(state.l >= self.size) or np.any(state.l) < 0:
            return False  # leaving the world is not allowed
        else:
            idx = state.l.astype(int)
            if seen_only:
                return not (
                    self.world[idx[0]][idx[1]][idx[2]]
                    and self.seen[idx[0]][idx[1]][idx[2]]
                )
            return not self.world[idx[0]][idx[1]][idx[2]]

    def connectsTo(self, state1, state2, seen_only=True, freq=10):
        checkpoints = np.arange(0, 1, 1 / (state1.distance(state2) * freq))
        for c in checkpoints:
            if not self.freespace(state1.intermediate(state2, c), seen_only):
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
                if not self.freespace(
                    state.intermediate(vstate, checkpoints[i]), False
                ):
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

    def see_test(self, l, state):
        return self.seen[l[0], l[1], l[2]] or self.connectsToVoxel(
            state, l[0], l[1], l[2]
        )

    def multiSee(self, state):
        locs = []
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                for k in range(self.world.shape[2]):
                    locs.append((i, j, k))

        res = []
        with Pool(8) as p:
            res = p.map(partial(self.see_test, state=state), locs)

        for i in range(len(locs)):
            self.seen[locs[i][0], locs[i][1], locs[i][2]] = res[i]

    def see(self, state):
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                for k in range(self.world.shape[2]):
                    self.seen[i, j, k] = self.seen[i, j, k] or self.connectsToVoxel(
                        state, i, j, k
                    )

    def combine(self, newWorld):
        combined_world = np.logical_or(self.world, newWorld.world)
        combined_seen = np.logical_or(self.seen, newWorld.seen)
        return World(combined_world, self.colors, combined_seen)

    def closestUnexplored(self, state, best_state=None):
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                for k in range(self.world.shape[2]):
                    if (
                        not self.seen[i, j, k]
                        and not self.inaccessible[i, j, k]
                        and (
                            best_state is None
                            or state.distance(State(i + 0.5, j + 0.5, k + 0.5))
                            < state.distance(best_state)
                        )
                    ):
                        best_state = State(i + 0.5, j + 0.5, k + 0.5)
        return best_state

    def bestUnexplored(self, state, best_state=None):
        nearby_scores = scipy.ndimage.convolve(
            np.logical_not(self.seen).astype(np.float64),
            self.distance_weights,
            mode="constant",
        )

        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                for k in range(self.world.shape[2]):
                    if (
                        not self.seen[i, j, k]
                        and not self.inaccessible[i, j, k]
                        and (
                            best_state is None
                            or state.distance(State(i + 0.5, j + 0.5, k + 0.5))
                            / max(0.01, nearby_scores[i, j, k])
                            < state.distance(best_state)
                            / max(
                                0.01,
                                nearby_scores[
                                    int(best_state.l[0]),
                                    int(best_state.l[1]),
                                    int(best_state.l[2]),
                                ],
                            )
                        )
                    ):
                        best_state = State(i + 0.5, j + 0.5, k + 0.5)
        return best_state

    def stateSeen(self, state):
        idx = state.l.astype(int)
        return self.seen[idx[0], idx[1], idx[2]]

    def markInaccessible(self, state):
        idx = state.l.astype(int)
        self.inaccessible[idx[0], idx[1], idx[2]] = True
