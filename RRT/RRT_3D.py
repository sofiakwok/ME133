#!/usr/bin/env python3
#
#   prmtriangles.py
#

import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from sklearn.neighbors import KDTree
#from planarutils import *
#from prmtools import *


######################################################################
#
#   General/World Definitions
#
#   List of objects, start, goal, and parameters.
#
(xmin, xmax) = (0, 14)
(ymin, ymax) = (0, 10)
(zmin, zmax) = (0, 10)

(startx, starty) = (1, 5, 10)
(goalx,  goaly)  = (8, 5, 5)

N  = 300   # PICK THE PARAMERTERS 
K  = 10

#### Node Definition
import bisect
import math


#
#   Node class upon which to build the graph (roadmap) and which
#   supports the A* search tree.
#
class Node:
    def __init__(self, state):
        # Save the state matching this node.
        self.state = state

        # Edges used for the graph structure (roadmap).
        self.children = []
        self.parents  = []

        # Status, edge, and costs for the A* search tree.
        self.seen        = False
        self.done        = False
        self.treeparent  = []
        self.costToReach = 0
        self.costToGoEst = math.inf
        self.cost        = self.costToReach + self.costToGoEst

    # Define the "less-than" to enable sorting by cost in A*.
    def __lt__(self, other):
        return self.cost < other.cost

    # Distance to another node, for A*, using the state distance.
    def Distance(self, other):
        return self.state.Distance(other.state)


#
#   A* Planning Algorithm
#
def AStar(nodeList, start, goal):
    # Prepare the still empty *sorted* on-deck queue.
    onDeck = []

    # Clear the search tree (for repeated searches).
    for node in nodeList:
        node.seen = False
        node.done = False

    # Begin with the start state on-deck.
    start.done        = False
    start.seen        = True
    start.treeparent  = None
    start.costToReach = 0
    start.costToGoEst = start.Distance(goal)
    start.cost        = start.costToReach + start.costToGoEst
    bisect.insort(onDeck, start)

    # Continually expand/build the search tree.
    while True:
        # Grab the next node (first on deck).
        node = onDeck.pop(0)

        # Add the children to the on-deck queue (or update)
        for child in node.children:
            # Skip if already done.
            if child.done:
                continue

            # Compute the cost to reach the child via this new path.
            costToReach = node.costToReach + node.Distance(child)

            # Just add to on-deck if not yet seen (in correct order).
            if not child.seen:
                child.seen        = True
                child.treeparent  = node
                child.costToReach = costToReach
                child.costToGoEst = child.Distance(goal)
                child.cost        = child.costToReach + child.costToGoEst
                bisect.insort(onDeck, child)
                continue

            # Skip if the previous cost was better!
            if child.costToReach <= costToReach:
                continue

            # Update the child's connection and resort the on-deck queue.
            child.treeparent  = node
            child.costToReach = costToReach
            child.cost        = child.costToReach + child.costToGoEst
            onDeck.remove(child)
            bisect.insort(onDeck, child)

        # Declare this node done.
        node.done = True

        # Check whether we have processed the goal (now done).
        if (goal.done):
            break

        # Also make sure we still have something to look at!
        if not (len(onDeck) > 0):
            return []

    # Build the path.
    path = [goal]
    while path[0].treeparent is not None:
        path.insert(0, path[0].treeparent)

    # Return the path.
    return path


######################################################################
#
#   State Definition
#
class State:
    def __init__(self, x, y, z):
        # Remember the (x,y) position.
        self.x = x
        self.y = y
        self.z = z


    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self):
        return ("<Point %2d,%2d,%2d>" % (self.x, self.y, self.z))

    # Compute/create an intermediate state.  This can be useful if you
    # need to check the local planner by testing intermediate states.
    def intermediate(self, other, alpha):
        return State(self.x + alpha * (other.x - self.x),
                     self.y + alpha * (other.y - self.y), 
                     self.z + alpha * (other.z - self.z))

    # Return a tuple of the coordinates.
    def coordinates(self):
        return (self.x, self.y, self.z)


    ############################################################
    # PRM Functions:

    # Compute the relative distance to another state.    
    def distance(self, other):
        x = np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
        return x


######################################################################
#
#   PRM Functions
#

#
# Sample the space
#
def addNodesToList(nodeList, N):
    # Add uniformly distributed samples
    while (N > 0):
        state = State(random.uniform(xmin, xmax),
                      random.uniform(ymin, ymax))
        if state.inFreespace():
            nodeList.append(Node(state))
            N = N-1


#
#   Connect the nearest neighbors
#
def connectNearestNeighbors(nodeList, K):
    # Clear any existing neighbors.
    for node in nodeList:
        node.children = []
        node.parents  = []

    # Determine the indices for the nearest neighbors.  This also
    # reports the node itself as the closest neighbor, so add one
    # extra here and ignore the first element below.
    X   = np.array([node.state.Coordinates() for node in nodeList])
    kdt = KDTree(X)
    idx = kdt.query(X, k=(K+1), return_distance=False)

    # Add the edges (from parent to child).  Ignore the first neighbor
    # being itself.
    for i, nbrs in enumerate(idx):
        for n in nbrs[1:]:
            if nodeList[i].state.connectsTo(nodeList[n].state):
                nodeList[i].children.append(nodeList[n])
                nodeList[n].parents.append(nodeList[i])


#
#  Post Process the Path
#
def PostProcess(path):
    '''refined_path = [x for x in path if x != 0]
    for i in range(len(path)-2):
        if path[i].state.ConnectsTo(path[i+2].state):
            refined_path.remove(path[i+1])'''
    return path

######################################################################
#
#  Main Code
#
def main():
    # Report the parameters.
    print('Running with ', N, ' nodes and ', K, ' neighbors.')

    # Create the figure.
    Visual = Visualization()

    # Create the start/goal nodes.
    startnode = Node(State(startx, starty, startz))
    goalnode  = Node(State(goalx,  goaly, goalz))
    
    # Create the list of sample points.
    start = time.time()
    nodeList = []
    addNodesToList(nodeList, N)
    print('Sampling took ', time.time() - start)

    # Add the start/goal nodes.
    nodeList.append(startnode)
    nodeList.append(goalnode)

    # Connect to the nearest neighbors.
    start = time.time()
    connectNearestNeighbors(nodeList, K)
    print('Connecting took ', time.time() - start)

    # Run the A* planner.
    start = time.time()
    path = AStar(nodeList, startnode, goalnode)
    print('A* took ', time.time() - start)
    if not path:
        print("UNABLE TO FIND A PATH")
        return


    # Post Process the path.
    PostProcess(path)

if __name__== "__main__":
    main()
