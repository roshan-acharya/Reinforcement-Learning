import numpy as np
import random

class GridWorld:
    def __init__(self, grid_size=6):
        self.grid_size= grid_size
        self.actions = ['up', 'down', 'left', 'right']
        self.reset()

    def reset(self): 
        position=random.sample(range(self.grid_size*self.grid_size), 3)
        coords= [divmod(p, self.grid_size) for p in position]
        self.agent_pos = np.array(coords[0])
        self.goal_pos = np.array(coords[1])
        self.danger_pos = np.array(coords[2])
        self.done = False

        return self.agent_pos.copy()
    
    #Set step Up, Down, Left, Right
    def step(self, action):
        if self.done:
            raise Exception("Episode has ended. Please reset the environment.")
        if action == "up":
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == "down":
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == "left":
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == "right":
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        else:
            raise ValueError("Invalid action! Choose from: up, down, left, right")
        
        reward= -1 #penalty for wrong step

        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 10 #reward
            self.done = True
            result="goal"
        
        elif np.array_equal(self.agent_pos, self.danger_pos):
            reward = -10 #penalty for loose
            self.done = True
            result="lose"
        
        else:
            result=None

        return self.agent_pos.copy(), reward, self.done, result
    
    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ' . ')
        grid[self.goal_pos[0], self.goal_pos[1]] = ' G '
        grid[self.danger_pos[0], self.danger_pos[1]] = ' D '
        grid[self.agent_pos[0], self.agent_pos[1]] = ' A '

        print("\n".join(["".join(row) for row in grid]))
        print()

if __name__ == "__main__":
    env = GridWorld(grid_size=6)
    state = env.reset()
    env.render()
    done = False