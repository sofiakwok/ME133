from node import Node, State

import numpy as np
import random
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PROP_GOAL_GROWTH = 0.1


class Tree:
    def __init__(self, goalstate):
        self.tree = [Node(goalstate, None)]
        self.goalstate = goalstate
        self.startstate = None
        self.startnode = None

    def RRT(self, world, startstate, Nmax=10000, dstep=1):
        self.startstate = startstate
        # Loop.

        # check if we actually need to grow the tree (start may already be visible)
        for n in self.tree:
            if world.connectsTo(startstate, n.state):
                self.startnode = Node(startstate, parent=n)
                n.addChild(self.startnode)
                self.tree.append(self.startnode)
                return self.startnode

        while True:
            # Determine the target state.
            to_start = random.uniform(0, 1) < PROP_GOAL_GROWTH
            lnew = (
                startstate.l
                if to_start
                else np.random.random(len(startstate.l)) * world.size
            )
            targetstate = State(lnew[0], lnew[1], lnew[2])

            # Find the nearest node (node with state nearest the target state).
            # This is inefficient (slow for large trees), but simple.
            list = [(node.state.distance(targetstate), node) for node in self.tree]
            (d, nearnode) = min(list)
            nearstate = nearnode.state

            # Determine the next state, a step size (dstep) away.
            nextstate = nearstate.intermediate(targetstate, dstep / d)

            # Check whether to attach (creating a new node).
            if world.connectsTo(nearstate, nextstate):
                nextnode = Node(nextstate, parent=nearnode)
                nearnode.addChild(nextnode)
                self.tree.append(nextnode)

                # Also try to connect the goal.
                if world.connectsTo(startstate, nextstate):
                    self.startnode = Node(startstate, parent=nextnode)
                    # print("children at creation:", self.startnode.children)
                    nextnode.addChild(self.startnode)
                    self.tree.append(self.startnode)
                    # print("children at build:", self.startnode.children)
                    return self.startnode

            # Check whether we should abort (tree has gotten too large).
            if len(self.tree) >= Nmax:
                return None

    def recursiveRemove(self, n):
        if n.parent is not None:
            n.parent.children.remove(n)
        self.tree.remove(n)
        for c in n.children:
            self.recursiveRemove(c)

    def prune(self, world):
        if self.startnode is not None:
            # print("children at prune:", self.startnode.children)
            self.recursiveRemove(self.startnode)
            self.startnode = None

        for n in self.tree:
            if n.parent is not None and not world.connectsTo(n.state, n.parent.state):
                self.recursiveRemove(n)

    def getPath(self):
        if self.startnode is None:
            return None

        path = []
        cnode = self.startnode
        while cnode is not None:
            path.append(cnode)
            cnode = cnode.parent

        return path

    def plot(self, ax):
        p1s = []
        p2s = []

        path = self.getPath()

        colors = []
        for n in self.tree:
            if n.parent is not None:
                p1s.append(n.state.l)
                p2s.append(n.parent.state.l)
                if colors is not None and n in path and n.parent in path:
                    colors.append("r")
                else:
                    colors.append("b")

        p1s = np.array(p1s)
        p2s = np.array(p2s)

        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=0.5, colors=colors)
        ax.add_collection(lc)

        if self.startstate is not None:
            pts = np.vstack((self.startstate.l, self.goalstate.l)).transpose()
            ax.scatter(pts[0], pts[1], pts[2], color=["green", "red"])
