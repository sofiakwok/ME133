from tracemalloc import start
from world import World
from node import Node, State
from tree import Tree

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generateForest(
    ws=50,
    wh=50,
    ntrees=10,
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


def generateWall(ws=50, wh=50, wall_height=20, start=None, goal=None):
    world = np.zeros((ws, ws, wh), dtype=bool)

    floor = np.zeros((ws, ws, wh), dtype=bool)
    floor[:, :, 0] = True

    wall = np.zeros((ws, ws, wh), dtype=bool)
    wall[int(ws / 2), :, 1:wall_height] = True

    world = floor | wall

    colors = np.empty(world.shape, dtype=object)
    colors[floor] = "darkgreen"
    colors[wall] = "grey"

    return World(world, colors)


def plot(world, tree=None, size=None):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    world.plot(ax)
    if tree is not None:
        tree.plot(ax)

    if size is None:
        ax.axes.set_xlim3d(left=0, right=world.maxsize())
        ax.axes.set_ylim3d(bottom=0, top=world.maxsize())
        ax.axes.set_zlim3d(bottom=0, top=world.maxsize())

    plt.show()


if __name__ == "__main__":
    startstate = State(5, 5, 2)
    goalstate = State(45, 45, 2)

    print("generate")
    world = generateWall(wall_height=45)

    print("visible")
    visibleWorld = world.visibleFrom(startstate)

    print("tree")
    tree = Tree(goalstate)
    tree.RRT(visibleWorld, startstate)

    print(len(tree.tree))

    plot(visibleWorld, tree=tree)
