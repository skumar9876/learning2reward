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

    def __init__(self, fixed=False):
        self.fixed = fixed
        self.reset()

    def reset(self):
        self.world, self.sentence, self.color = self.generate_map()
        self.x = 0
        self.y = 0
        self.attention = np.array([0, 0])

        self.update_attention()
        self.num_steps = 0

        self.MAX_STEPS = 200

        return self.attention_image, self.sentence


    def generate_map(self):
        world = []
        colors = [20, 30, 40, 50]
        colors_dict = {20: 0, 30: 1, 40: 2, 50: 3}
        random.shuffle(colors) # Randomly shuffle the colors
        lines = 10

        fixed_num_lines = [3, 2, 3, 2]

        for i in xrange(len(colors)):
            color = colors[i]

            num_lines = 0

            if i == len(colors) - 1: # This is the last color
                num_lines = lines 
            else:
                if self.fixed == False:
                    num_lines = random.randint(1, 4)
                    while lines - num_lines < 0:
                        num_lines = random.randint(1, 4)
                    lines -= num_lines
                else: 
                    num_lines = fixed_num_lines[i]
                    lines -= num_lines

            for y in range(num_lines):
                for x in range(10):
                    world.append(color)

        old_color = world[0]
        world[0] = 10
        index = world[95]
        world = np.array(world)
        world = world.reshape((10, 10))
        sentence_index = colors_dict[index]
        sentence = np.zeros([1, 4])

        sentence[0][sentence_index] = 1
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

    def update_attention(self):
        """
        Updates the 5x5 attention image
        """
        y = self.attention[0]
        x = self.attention[1]
        self.attention_image = np.array(self.world.copy())[y:y+5, x:x+5]

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

        self.world[self.y][self.x] = self.color

        if self.y < y:
            self.y += 1
        if self.y > y:
            self.y -= 1

        self.color = self.world[self.y][self.x]
        self.world[self.y][self.x] = 10

    def step(self, chosen_room, attention_action):
        sentence = [0, 0, 0, 0]
        sentence[chosen_room] = 1

        action = [0, 0, 0, 0, 0]
        action[attention_action] = 1

        self.move_agent(sentence)
        self.next_attention(action)

        self.update_attention()
        self.num_steps += 1

        done = self.isTerminal()

        print sentence
        print ""
        print action
        print ""
        print self.attention_image
        print ""
        print self.world
        print ""
        print ""
        print ""


        return self.attention_image, self.sentence, done

    def episode_reward(self):
        colors_dict = {20: 0, 30: 1, 40: 2, 50: 3}
        if self.sentence[0][colors_dict[self.color]] == 1:
            print "here"
            return 1
        else:
            return -1

    def isTerminal(self):
        colors_dict = {20: 0, 30: 1, 40: 2, 50: 3}


        if self.num_steps > self.MAX_STEPS or self.sentence[0][colors_dict[self.color]] == 1:
            return True
        else:
            return False