import logging
from enum import Enum, IntEnum

import matplotlib.pyplot as plt
import numpy as np


class Cell(IntEnum):
    EMPTY = 0  # empty cell that agent can move to
    OCCUPIED = 1  # cell occupied by wall
    CURRENT = 2  # cell currently occupied by the agent


class Action(IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3


class Render(Enum):
    NOTHING = 0
    TRAINING = 1
    MOVES = 2


class Status(Enum):
    WIN = 0
    LOSE = 1
    PLAYING = 2


class Maze:
    actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN]  # all possible actions

    reward_exit = 10.0  # reward for reaching the exit cell
    penalty_move = -0.05  # penalty for a move which did not result in finding the exit cell
    penalty_visited = -0.25  # penalty for returning to a cell which was visited earlier
    penalty_impossible_move = -0.75  # penalty for trying to enter an occupied cell or moving out of the maze
    penalty_clearance = -0.75   # penalty for lack of clearance (walls in immediate surrounding)

    def __init__(self, maze, start_cell=(0, 0), exit_cell=None):
        self.maze = maze

        self.__minimum_reward = -0.5 * self.maze.size  # stop game if accumulated reward is below this threshold

        nrows, ncols = self.maze.shape
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == Cell.EMPTY]
        self.__exit_cell = (ncols - 1, nrows - 1) if exit_cell is None else exit_cell
        self.empty.remove(self.__exit_cell)

        # Check for impossible maze layout
        if self.__exit_cell not in self.cells:
            raise Exception("Error: exit cell at {} is not inside maze".format(self.__exit_cell))
        if self.maze[self.__exit_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: exit cell at {} is not free".format(self.__exit_cell))

        # Variables for rendering
        self.__render = Render.NOTHING  # what to render
        self.__ax1 = None  # axes for rendering the moves
        self.__ax2 = None  # axes for rendering the best action per cell

        self.reset(start_cell)

    def reset(self, start_cell=(0, 0)):
        if start_cell not in self.cells:
            raise Exception("Error: start cell at {} is not inside maze".format(start_cell))
        if self.maze[start_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: start cell at {} is not free".format(start_cell))
        if start_cell == self.__exit_cell:
            raise Exception("Error: start- and exit cell cannot be the same {}".format(start_cell))

        self.__previous_cell = self.__current_cell = start_cell
        self.__total_reward = 0.0  # accumulated reward
        self.__visited = set()  # a set() only stores unique values

        if self.__render in (Render.TRAINING, Render.MOVES):
            # render the maze
            nrows, ncols = self.maze.shape
            self.__ax1.clear()
            self.__ax1.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax1.set_xticklabels([])
            self.__ax1.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax1.set_yticklabels([])
            self.__ax1.grid(True)
            self.__ax1.plot(*self.__current_cell, "rs", markersize=30)  # start is a big red square
            self.__ax1.text(*self.__current_cell, "Start", ha="center", va="center", color="white")
            self.__ax1.plot(*self.__exit_cell, "gs", markersize=30)  # exit is a big green square
            self.__ax1.text(*self.__exit_cell, "Exit", ha="center", va="center", color="white")
            self.__ax1.imshow(self.maze, cmap="binary")
            self.__ax1.get_figure().canvas.draw()
            self.__ax1.get_figure().canvas.flush_events()

        return self.__observe()

    def __draw(self):
        self.__ax1.plot(*zip(*[self.__previous_cell, self.__current_cell]), "bo-")  # previous cells are blue dots
        self.__ax1.plot(*self.__current_cell, "ro")  # current cell is a red dot
        self.__ax1.get_figure().canvas.draw()
        self.__ax1.get_figure().canvas.flush_events()

    def render(self, content=Render.NOTHING):
        self.__render = content

        if self.__render == Render.NOTHING:
            if self.__ax1:
                self.__ax1.get_figure().close()
                self.__ax1 = None
            if self.__ax2:
                self.__ax2.get_figure().close()
                self.__ax2 = None
        if self.__render == Render.TRAINING:
            if self.__ax2 is None:
                fig, self.__ax2 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Best move")
                self.__ax2.set_axis_off()
                self.render_q(None)
        if self.__render in (Render.MOVES, Render.TRAINING):
            if self.__ax1 is None:
                fig, self.__ax1 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Maze")

        plt.show(block=False)

    def step(self, action):
        reward = self.__execute(action)
        self.__total_reward += reward
        status = self.__status()
        state = self.__observe()
        logging.debug("action: {:10s} | reward: {: .2f} | status: {}".format(Action(action).name, reward, status))
        return state, reward, status

    def __execute(self, action):
        possible_actions = self.__possible_actions(self.__current_cell)

        if not possible_actions:
            reward = self.__minimum_reward - 1  # cannot move anywhere --> force end of game
        elif action in possible_actions:
            col, row = self.__current_cell
            if action == Action.MOVE_LEFT:
                col -= 1
            elif action == Action.MOVE_UP:
                row -= 1
            if action == Action.MOVE_RIGHT:
                col += 1
            elif action == Action.MOVE_DOWN:
                row += 1

            self.__previous_cell = self.__current_cell
            self.__current_cell = (col, row)

            if self.__render != Render.NOTHING:
                self.__draw()

            nrows, ncols = self.maze.shape
            if self.__current_cell == self.__exit_cell:
                reward = Maze.reward_exit
            elif self.__current_cell in self.__visited:
                reward = Maze.penalty_visited
            elif (row > 0 and self.maze[row - 1, col] == Cell.OCCUPIED) or (row < nrows - 1 and self.maze[row + 1, col] == Cell.OCCUPIED) or (col > 0 and self.maze[row, col - 1] == Cell.OCCUPIED) or (col < ncols - 1 and self.maze[row, col + 1] == Cell.OCCUPIED): #walls in immediate surrounding
                reward = Maze.penalty_clearance
            else:
                reward = Maze.penalty_move

            self.__visited.add(self.__current_cell)
        else:
            reward = Maze.penalty_impossible_move

        return reward

    def __possible_actions(self, cell=None):
        if cell is None:
            col, row = self.__current_cell
        else:
            col, row = cell

        possible_actions = Maze.actions.copy()  # initially allow all actions

        # remove impossible actions from list
        nrows, ncols = self.maze.shape
        if row == 0 or (row > 0 and self.maze[row - 1, col] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_UP)
        if row == nrows - 1 or (row < nrows - 1 and self.maze[row + 1, col] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_DOWN)

        if col == 0 or (col > 0 and self.maze[row, col - 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_LEFT)
        if col == ncols - 1 or (col < ncols - 1 and self.maze[row, col + 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_RIGHT)

        return possible_actions

    def __status(self):     # game status
        if self.__current_cell == self.__exit_cell:
            return Status.WIN

        if self.__total_reward < self.__minimum_reward:  # force end of game after to much loss
            return Status.LOSE

        return Status.PLAYING

    def __observe(self):    # agent's current location - state of maze
        return np.array([[*self.__current_cell]])

    def play(self, model, start_cell=(0, 0)):
        self.reset(start_cell)

        state = self.__observe()

        while True:
            action = model.predict(state=state)
            state, reward, status = self.step(action)
            if status in (Status.WIN, Status.LOSE):
                return status

    def render_q(self, model):  # render recommended action for each cell
        def clip(n):
            return max(min(1, n), 0)

        if self.__render == Render.TRAINING:
            nrows, ncols = self.maze.shape

            self.__ax2.clear()
            self.__ax2.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax2.set_xticklabels([])
            self.__ax2.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax2.set_yticklabels([])
            self.__ax2.grid(True)
            self.__ax2.plot(*self.__exit_cell, "gs", markersize=30)  # exit is a big green square
            self.__ax2.text(*self.__exit_cell, "Exit", ha="center", va="center", color="white")

            for cell in self.empty:
                q = model.q(cell) if model is not None else [0, 0, 0, 0]
                a = np.nonzero(q == np.max(q))[0]

                for action in a:
                    dx = 0
                    dy = 0
                    if action == Action.MOVE_LEFT:
                        dx = -0.2
                    if action == Action.MOVE_RIGHT:
                        dx = +0.2
                    if action == Action.MOVE_UP:
                        dy = -0.2
                    if action == Action.MOVE_DOWN:
                        dy = 0.2

                    # color (red to green) represents the certainty
                    color = clip((q[action] - -1)/(1 - -1))

                    self.__ax2.arrow(*cell, dx, dy, color=(1 - color, color, 0), head_width=0.2, head_length=0.1)

            self.__ax2.imshow(self.maze, cmap="binary")
            self.__ax2.get_figure().canvas.draw()
