import logging
import random
from datetime import datetime

import numpy as np

from environment import Status
from models import AbstractModel


class QTableModel(AbstractModel):
    default_check_convergence_every = 5  # by default check for convergence every 5 episodes

    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)
        self.Q = dict()  # table with value for (state, action) combination

    def train(self, stop_at_convergence=False, **kwargs):
        discount = kwargs.get("discount", 0.90) # (gamma) preference for future rewards
        exploration_rate = kwargs.get("exploration_rate", 0.10) # (epsilon) 0 = preference for exploring
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # exploration rate reduction after each random step
        learning_rate = kwargs.get("learning_rate", 0.10)   # (alpha) preference for using new knowledge
        episodes = max(kwargs.get("episodes", 1000), 1) # number of training games to play
        check_convergence_every = kwargs.get("check_convergence_every", self.default_check_convergence_every)

        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()
        start_time = datetime.now()

        # training starts here
        for episode in range(1, episodes + 1):
            start_cell = (0,0)  #start only from top left (designated entrace)

            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())  # change np.ndarray to tuple so it can be used as dictionary key

            while True:
                # choose action epsilon greedy (off-policy, instead of only using the learned policy)
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                else:
                    action = self.predict(state)

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())

                cumulative_reward += reward

                if (state, action) not in self.Q.keys():  # ensure value exists for (state, action) to avoid a KeyError
                    self.Q[(state, action)] = 0.0

                max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])

                self.Q[(state, action)] += learning_rate * (reward + discount * max_next_Q - self.Q[(state, action)])

                if status in (Status.WIN, Status.LOSE):  # terminal state reached, stop training episode
                    break

                state = next_state

                self.environment.render_q(self)

            cumulative_reward_history.append(cumulative_reward)

            logging.info("episode: {:d}/{:d} | status: {:4s} | e: {:.5f} | r: {:.5f}"
                         .format(episode, episodes, status.name, exploration_rate, cumulative_reward))  #return cumulative_reward to qualify accessibility

            exploration_rate *= exploration_decay  # explore less as training progresses

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time

    def q(self, state):
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])

    def predict(self, state):
        q = self.q(state)

        logging.debug("q[] = {}".format(q))

        actions = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value from Q-table
        return random.choice(actions)
