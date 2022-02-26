from tracemalloc import start
from world import World
from node import State
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

def generateSimpleMaze(ws=50, wh=50):
    floor = np.zeros((ws, ws, wh), dtype=bool)
    floor[:, :, 0] = True

    mask = np.random.uniform(0, 1, (ws, ws, 1)) > 0.9
    wall = np.repeat(mask, wh, axis=2)

    world = floor | wall

    colors = np.empty(world.shape, dtype=object)
    colors[floor] = "darkgreen"
    colors[wall] = "grey"
    return World(world, colors)

def generateMediumMaze(ws=50, wh=50):
    floor = np.zeros((ws, ws, wh), dtype=bool)
    floor[:, :, 0] = True

    wall = np.zeros((ws, ws, wh), dtype=bool)

    for x in range(ws-11):
        for y in range(ws-11):
            space = np.random.uniform(0, 1)
            length = np.random.randint(3, 10)
            multiplier = np.random.randint(0, 2)
            if space > 0.98:
                wall[x:x+(length*multiplier)+1, y:y+(length*(1-multiplier))+1, :] = True

    world = floor | wall

    colors = np.empty(world.shape, dtype=object)
    colors[floor] = "darkgreen"
    colors[wall] = "grey"
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


def plot(world, tree=None, size=None, save=None):
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    if tree is not None:
        tree.plot(ax)
    world.plot(ax)

    if size is None:
        ax.axes.set_xlim3d(left=0, right=world.maxsize())
        ax.axes.set_ylim3d(bottom=0, top=world.maxsize())
        ax.axes.set_zlim3d(bottom=0, top=world.maxsize())

    if save is None:
        plt.show()
    else:
        plt.savefig("out/n" + f"{save:04}" + ".png")
        ax.view_init(0, -90)
        plt.savefig("out/s" + f"{save:04}" + ".png")
        ax.view_init(90, 0)
        plt.savefig("out/t" + f"{save:04}" + ".png")
        plt.close()


if __name__ == "__main__":
    step_size = 1
    startstate = State(5, 5, 2)
    goalstate = State(45, 45, 2)

    print("generate")
    world = generateMediumMaze()

    visibleWorld = None
    tree = None
    i = 0
    while startstate.distance(goalstate) > 1:
        print("visible")
        newWorld = world.visibleFrom(startstate)
        visibleWorld = (
            newWorld if visibleWorld is None else visibleWorld.combine(newWorld)
        )

        print("tree")
        if tree is None:
            tree = Tree(goalstate)
            tree.RRT(visibleWorld, startstate)
        else:
            tree.prune(visibleWorld)
            tree.RRT(visibleWorld, startstate)

        path = tree.getPath()
        if path is None:
            print("PATH NOT FOUND")
            exit(0)
        else:
            startstate = startstate.intermediate(
                path[1].state, min(1, step_size / startstate.distance(path[1].state))
            )

        plot(visibleWorld, tree=tree, save=i)
        i += 1
