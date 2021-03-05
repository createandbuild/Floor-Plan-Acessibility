import logging
from enum import auto

import matplotlib.pyplot as plt
import numpy as np

import models
from environment.maze import Maze, Render

logging.basicConfig(format="%(levelname)-4s: %(message)s",
                    level=logging.INFO)

Q_LEARNING = auto()
test = Q_LEARNING

# three different scenarios of mazes

# maze = np.array([
#     [0,0,0,0,0],
#     [0,0,0,0,0],
#     [0,0,0,0,0],
#     [0,0,0,0,0],
#     [0,0,0,0,0]
# ])

# maze = np.array([
#     [0,0,1,1,1],
#     [1,0,1,0,0],
#     [1,0,0,0,1],
#     [1,1,0,0,1],
#     [1,1,1,0,0]
# ])

maze = np.array([
    [0,0,1,1,1],
    [0,0,0,0,1],
    [1,0,0,0,1],
    [1,0,0,0,0],
    [1,1,0,0,0]
])

game = Maze(maze)

game.render(Render.TRAINING)
model = models.QTableModel(game, name="QTableModel")
h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                            stop_at_convergence=True)

game.render(Render.MOVES)
game.play(model)

plt.show()
