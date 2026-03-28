#Define the agent class
import math
import numpy as np
from actions import Actions

#consider putting actions in agents.py

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def move(self, agent_action):
        distance = 0.5
        if agent_action == Actions().MOVE_UP:
            self.y = self.y-distance
        elif agent_action == Actions().MOVE_DOWN:
            self.y = self.y+distance
        elif agent_action == Actions().MOVE_LEFT:
            self.x = self.x-distance
        elif agent_action == Actions().MOVE_RIGHT:
            self.x = self.x+distance

class Agent:
    def __init__(self, name, position, health=3):
        self.name = name
        self.position = position
        self.health = health
        self.action_list = Actions().available_actions

    def take_damage(self):
        """Reduces the agent's health by 1."""
        self.health -= 1

    def is_alive(self):
        """Check if the agent is still alive."""
        return 

    def pick_action(self, state):
        """Placeholder for the agent's decision-making logic."""
        decided_action = self.action_list[0]
        #Uses the current state of the environment to decide on an action
        return decided_action


#random agent

#epsilon greedy

#direct attack

#Avoidance