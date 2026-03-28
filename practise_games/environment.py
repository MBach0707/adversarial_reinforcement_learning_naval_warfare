class Environment:
    def __init__(self, size=(10.0, 10.0)):
        self.size = size
        position_index = range(0, size[0]*size[1], 1)
        print(position_index)
        self.starting_positions = []  # List to store the initial positions of agents
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)
        self.starting_positions.append(agent.position)  # Store the starting position of the agent

    def reset(self):
        # Reset the agents to their starting positions and health
        for agent, new_position in zip(self.agents, self.starting_positions):
            agent.position = new_position  # Corrected: Set each agent's position to its initial position
            agent.health = 3

    def get_state(self):
        # Return a list of each agent's position and health as a tuple (x, y, health)
        # Returns the state vector containing, x component, y component and health.
        return [(agent.position.x, agent.position.y, agent.health) for agent in self.agents]

    def act(self, agent_actions):
        # Update the agents' positions and health
        for agent, agent_actions in self.agents, agent_actions:
            if agent.is_alive():
                agent.position.move(agent_actions)

#    def time_step(self):
#        # Update the agents' positions and health
#        for agent in self.agents:
#            picked_action = agent.pick_action(self.get_state())
#            if picked_action == [1,2,3,4]:

