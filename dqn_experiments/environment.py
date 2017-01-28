import numpy as np
import random

"""
Map Values:
Red = 20
Green = 30
Blue = 40
Wall = 60
Agent = 10
"""

class GridWorld():
    """
    Gridworld environment along with functions that allow an RL agent
    to navigate in the environment. Functions mimic OpenAI domain functions.
    """

    def __init__(self):
        filename = 'map1.txt' # Change map here
        self.original_board = []

        f = open(filename, 'r')
        info = f.readlines()

        for line in info:
            line = line.split(" ")

            for i in range(len(line)):
                line[i] = int(line[i])

            self.original_board.append(line)

        self.original_board = np.array(self.original_board)
        self.reset()

    def reset(self):
        """
        Returns a random starting state.

        :returns state
        :rtype numpy array
        """

        randx = random.randint(0, len(self.original_board[0]) - 1)
        randy = random.randint(0, len(self.original_board) - 1)

        while self.original_board[randx][randy] == 60:
            randy = random.randint(0, len(self.original_board) - 1)
            randx = random.randint(0, len(self.original_board[0]) - 1)

        self.state = self.original_board
        self.state[randy][randx] = 10

        self.x = randx
        self.y = randy

        sentence_map = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
        rand_sent = random.randint(0, 2)
        self.sentence = np.array(sentence_map[rand_sent])
        goal_map = {0: 20, 1: 30, 2: 40}
        self.goal = goal_map[np.argwhere(self.sentence)[0][0]]


        return self.state, self.sentence


    def step(self, action):
        """
        For a given action
        returns:
            - next state
            - reward for the state, action pair
            - whether or not the agent is now in a terminal state (use isTerminal function with next state as input)

        :param action: action that the agent has taken just now
                0 = up
                1 = right
                2 = down
                3 = left

        :returns: next state
        :rtype: numpy array
        :returns: reward
        :rtype: float
        :returns: whether or not episode is done
        :rtype: boolean
        """
        new_x = self.x
        new_y = self.y

        if action == 0:
            new_y -= 1
        elif action == 1:
            new_x += 1
        elif action == 2:
            new_y += 1
        elif action == 3:
            new_x -= 1

        self.state[self.y][self.x] = self.original_board[self.y][self.x]

        if 0 <= new_y < len(self.original_board) and \
        0 <= new_x < len(self.original_board[0]) and \
        self.original_board[new_y][new_x] != 60:
            self.x = new_x
            self.y = new_y

        self.state[self.y][self.x] = 10

        return self.state, self.reward(), self.isTerminal()


    def isTerminal(self):
        """
        For a given map and instruction,
        returns whether or not the agent is in
        a terminal state.

        :returns: whether or not current state is a terminal state
        :rtype: boolean
        """
        return self.original_board[self.y][self.x] == self.goal

    def reward(self):
        """
        returns: the reward the agent gets for being in
        the current state and taking the selected action.

        :returns reward
        :rtype float
        """

        if self.isTerminal():
            return 1
        else:
            return -.01
