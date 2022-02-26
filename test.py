import numpy as np
ws = 50
wh = 50
mask = np.random.uniform(0, 1, (ws, ws)) > 0.9
world = np.tile(mask, (1, 1, wh))
print(world)