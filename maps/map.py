#!/usr/bin/env python3
#
#   hw5_localize.py
#
#   Homework 5 code framework to localize a robot in a grid...
#
#   Places to edit are marked as TODO.
#
import numpy as np

from hw5_utilities import Visualization, Robot


#
#  Define the Walls
#
w = ['xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
     'x               x               x               x',
     'x                x             x                x',
     'x                 x           x                 x',
     'x        xxxx      x         x                  x',
     'x        x   x      x       x                   x',
     'x        x    x      x     x      xxxxx         x',
     'x        x     x      x   x     xxx   xxx       x',
     'x        x      x      x x     xx       xx      x',
     'x        x       x      x      x         x      x',
     'x        x        x           xx         xx     x',
     'x        x        x           x           x     x',
     'x        x        x           x           x     x',
     'x        x        x           x           x     x',
     'x                 xx         xx           x     x',
     'x                  x         x                  x',
     'x                  xx       xx                  x',
     'x                   xxx   xxx                   x',
     'x                     xxxxx         x           x',
     'x                                   x          xx',
     'x                                   x         xxx',
     'x            x                      x        xxxx',
     'x           xxx                     x       xxxxx',
     'x          xxxxx                    x      xxxxxx',
     'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx']

walls = np.array([[1.0*(c == 'x') for c in s] for s in w])
rows  = np.size(walls, axis=0)
cols  = np.size(walls, axis=1)


#
#  Prediction
#
#    bel         Grid of probabilities (current belief)
#    drow, dcol  Delta in row/col
#    probCmd     Modeled probability of command executing
#    prd         Grid of probabilities (prediction)
#
def computePrediction(bel, drow, dcol, probCmd):
    # Prepare an empty prediction grid.
    prd = np.zeros((rows,cols))
    cutoff = np.max(bel)*0.2*0.2

    # Iterate over/determine the probability for all (non-wall) elements.
    for row in range(1, rows-1):
        for col in range(1, cols-1):
            #TODO
            wallHit = False
            if walls[row + drow, col + dcol] == 1:
                wallHit = True
            if bel[row, col] >= cutoff:
                if bel[row + drow, col + dcol] != 0 and wallHit == False:
                    prd[row + drow, col + dcol] = bel[row, col]*probCmd
                    prd[row, col] = bel[row, col]*(1-probCmd)
                elif wallHit == True:
                    prd[row + drow, col + dcol] = False
                elif bel[row + drow, col + dcol] == 0:
                    prd[row + drow, col + dcol] = bel[row, col]*probCmd
                    prd[row, col] = bel[row, col]*(1-probCmd)
    # Return the prediction grid
    return prd


#
#  Measurement Update (Correction)
#
#    prior       Grid of prior probabilities (belief)
#    probSensor  Grid of probability that (sensor==True)
#    sensor      Value of sensor
#    post        Grid of posterior probabilities (updated belief)

def updateBelief(prior, probSensor, sensor):
    # Update the belief
    #TODO for the two cases: (sensor==True) or (sensor==False)
    post = prior
    for row in range(1, rows-1):
        for col in range(1, cols-1):
            #TODO
            if prior[row, col] != 0:
                if sensor:
                    if probSensor[row, col] != 0:
                        post[row, col] = sensor
                    elif probSensor[row, col] == 0:
                        post[row, col] == 0
                else:
                    if probSensor[row, col] == 0:
                        post[row, col] = prior[row, col]
                    elif probSensor[row, col] != 0:
                        post[row, col] = 0
    # Normalize.
    s = np.sum(post)
    if (s == 0.0):
        print("LOST ALL BELIEF.  SHOULD NOT HAPPEN.  STARTING OVER!!!!")
        post = 1.0 - walls
        s    = np.sum(post)

    post = (1.0/s) * post
    return post


#
#  Pre-compute the Sensor Probability Grid
#
#    drow, dcol    Direction in row/col
#    probProximal  List of probability that sensor triggers at dist=(index+1)
#    prob          Grid of probability that (sensor==True)
#
def computeSensorProbability(drow, dcol, probProximal):
    # Prepare an empty probability grid.
    prob = np.zeros((rows, cols))
    length = len(probProximal)
    # Pre-compute the probability for each grid element, knowing the
    # walls and the direction of the sensor.
    for row in range(1, rows-1):
        for col in range(1, cols-1):
            if walls[row + drow, col + dcol] == 1:
                prob[row, col] = probProximal[0]
            elif length > 1 and walls[row + drow*2, col + dcol*2] == 1:
                prob[row, col] = probProximal[1]
            elif length > 1 and walls[row + drow*3, col + dcol*3]:
                prob[row, col] = probProximal[2]

    # Return the computed grid.
    return prob


# 
#
#  Main Code
#
def main():
    # Initialize the figure.
    visual = Visualization(walls)

    #STODO... PICK WHAT THE "REALITY" SHOULD SIMULATE:
    # Initialize the robot simulation.
    # part (a)
    #robot  = Robot(walls)
    # Part (b)
    robot = Robot(walls, row=12, col=26)
    # Part (c)
    #robot = Robot(walls, row=12, col=26, probProximal = [0.9, 0.6, 0.3])
    # Part (d), (e)
    #robot = Robot(walls, row=15, col=47, probCmd = 0.8, probProximal = [0.9, 0.6, 0.3])
    # And to play:
    # robot = Robot(walls, probCmd = 0.8, probProximal = [0.9, 0.6, 0.3])


    #TODO... PICK WHAT YOUR LOCALIZATION SHOULD ASSUME:
    # Pick the algorithm assumptions:
    probCmd      = 1.0                  # Part (a/b), (c), (d)
    #probCmd      = 0.8                  # Part (e)
    probProximal = [1.0]                # Part (a/b)
    #probProximal = [0.9, 0.6, 0.3]      # Part (c), (d), (e)


    # Pre-compute the probability grids for each sensor reading.
    probUp    = computeSensorProbability(-1,  0, probProximal)
    probRight = computeSensorProbability( 0,  1, probProximal)
    probDown  = computeSensorProbability( 1,  0, probProximal)
    probLeft  = computeSensorProbability( 0, -1, probProximal)

    #TODO.... DO YOU WanT TO SEE THE PROBABILITY OF THE PROXIMITY SENSOR?
    # Show the sensor probability maps.

    '''
    visual.Show(probUp)
    input("Probability of proximal sensor up reporting True")
    visual.Show(probRight)
    input("Probability of proximal sensor right reporting True")
    visual.Show(probDown)
    input("Probability of proximal sensor down reporting True")
    visual.Show(probLeft)
    input("Probability of proximal sensor left reporting True")
    '''


    # Start with a uniform belief grid.
    bel = 1.0 - walls
    bel = (1/np.sum(bel)) * bel


    # Loop continually.
    while True:
        # Show the current belief.  Also show the actual position.
        visual.Show(bel, robot.Position())

        # Get the command key to determine the direction.
        while True:
            key = input("Cmd (q=quit, i=up, m=down, j=left, k=right) ?")
            if   (key == 'q'):  return
            elif (key == 'i'):  (drow, dcol) = (-1,  0) ; break
            elif (key == 'm'):  (drow, dcol) = ( 1,  0) ; break
            elif (key == 'j'):  (drow, dcol) = ( 0, -1) ; break
            elif (key == 'k'):  (drow, dcol) = ( 0,  1) ; break

        # Move the robot in the simulation.
        robot.Command(drow, dcol)


        # Compute a prediction.
        prd = computePrediction(bel, drow, dcol, probCmd)
        visual.Show(prd)
        input("Showing the prediction")


        # Correct the prediction/execute the measurement update.
        bel = prd
        bel = updateBelief(bel, probUp,    robot.Sensor(-1,  0))
        bel = updateBelief(bel, probRight, robot.Sensor( 0,  1))
        bel = updateBelief(bel, probDown,  robot.Sensor( 1,  0))
        bel = updateBelief(bel, probLeft,  robot.Sensor( 0, -1))


if __name__== "__main__":
    main()