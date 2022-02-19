from world import World
from node import Node, State

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generateForest(
    ws=200,
    wh=50,
    ntrees=50,
    mintreeheight=10,
    maxtreeheight=30,
    mincanopy=3,
    maxcanopy=7,
):
    world = np.zeros((ws, ws, wh), dtype=bool)

    floor = np.zeros((ws, ws, wh), dtype=bool)
    floor[:, :, 0] = True

    trunks = np.zeros((ws, ws, wh), dtype=bool)
    canopies = np.zeros((ws, ws, wh), dtype=bool)
    x, y, z = np.indices((ws, ws, wh))
    for _ in range(ntrees):
        tx = np.random.randint(ws)
        ty = np.random.randint(ws)
        th = np.random.randint(mintreeheight, maxtreeheight)
        tc = np.random.randint(mincanopy ** 2, maxcanopy ** 2)
        trunks[tx, ty, 1:th] = True

        canopies = canopies | (
            (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - th) * (z - th) < tc
        )

    world = floor | trunks | canopies

    colors = np.empty(world.shape, dtype=object)
    colors[floor] = "darkgreen"
    colors[trunks] = "brown"
    colors[canopies] = "lightgreen"

    return World(world, colors)


def plot(world, start=None, goal=None, tree=None, size=None):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # ax = plt.figure().add_subplot(projection="3d")
    world.plot(ax)

    if size is None:
        ax.axes.set_xlim3d(left=0, right=world.maxsize())
        ax.axes.set_ylim3d(bottom=0, top=world.maxsize())
        ax.axes.set_zlim3d(bottom=0, top=world.maxsize())

    plt.show()


if __name__ == "__main__":
    world = generateForest()
    plot(world)
