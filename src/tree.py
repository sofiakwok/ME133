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

    def RRT(self, world, startstate, Nmax=10000, dstep=1):

        self.startstate = startstate
        # Loop.
        while True:
            # Determine the target state.
            to_goal = random.uniform(0, 1) < PROP_GOAL_GROWTH
            lnew = (
                startstate.l
                if to_goal
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
                    goalnode = Node(startstate, nextnode)
                    nextnode.addChild(goalnode)
                    self.tree.append(goalnode)
                    return goalnode

            # Check whether we should abort (tree has gotten too large).
            if len(self.tree) >= Nmax:
                return None

    def plot(self, ax):
        p1s = []
        p2s = []

        for n in self.tree:
            if n.parent is not None:
                p1s.append(n.state.l)
                p2s.append(n.parent.state.l)

        p1s = np.array(p1s)
        p2s = np.array(p2s)

        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=0.5, colors="b")
        ax.add_collection(lc)

        if self.startstate is not None:
            pts = np.vstack((self.startstate.l, self.goalstate.l)).transpose()
            ax.scatter(pts[0], pts[1], pts[2], color=["green", "red"])
