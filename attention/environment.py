import random
import numpy as np

"""
Map Values:
Red = 20
Green = 30
Blue = 40
Yellow = 50
Agent = 10
"""
class World():

    def __init__(self):
        self.world, self.sentence, self.color = generate_map()
        self.x = 0
        self.y = 0
        self.attention = np.array([0, 0])

    def generate_map(self):
        world = []
        colors = [20, 30, 40, 50]
        colors_dict = {20: 0, 30: 1, 40: 2, 50: 3}
        random.shuffle(colors)
        lines = 10
        for color in colors:
            num_lines = random.randint(1, 4)
            while lines - num_lines < 0:
                num_lines = random.randint(1, 4)
            lines -= num_lines
            for x in range(1, num_lines):
                for y in range(1, 10):
                    world.append(color)
        old_color = world[0]
        world[0] = 10
        index = world[95]
        world = np.array(world)
        world = world.reshape((10, 10))
        sentence_index = colors_dict[index]
        sentence = np.zeros([1, 4])
        sentence[sentence_index] = 1
        return world, sentence, old_color

    def next_attention(self, action):
        """
        action = [up, down, left, right]
        """
        y = self.attention[0]
        x = self.attention[1]
        if action[0] == 1:
            y = max(0, y - 1)
        if action[1] == 1:
            y = min(4, y + 1)
        if action[2] == 1:
            x = max(0, x - 1)
        if action[3] == 1:
            x = min(4, x + 1)
        self.attention = np.array([y, x])
        return self.attention

    def move_agent(self, sentence):
        """
        sentence = [20, 30, 40, 50]
        """
        colors_dict = {0: 20, 1: 30, 2: 40, 3: 50}
        goal = colors_dict[sentence.index(1)]
        action = 0
        for y in range(0, 9):
            color = self.world[y][self.x]
            if color == goal:
                action = y
                break
        if self.y < y:
            self.y += 1
        if self.y > y:
            self.y -= 1


    def reward(self):
        colors_dict = {20: 0, 30: 1, 40: 2, 50: 3}
        if self.sentence[colors_dict[self.color]] == 1:
            return 1
        else:
            return -1

