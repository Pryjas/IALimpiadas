import numpy as np
import os
import pickle

class Bot:
    def __init__(self, env):
        self.env = env
        self.obs = None
        
        if os.path.exists('best_sequence.pkl'):
            try:
                with open('best_sequence.pkl', 'rb') as f:
                    evolved_sequence = pickle.load(f)
                    solution = evolved_sequence * 150
            except:
                solution = self._get_default_solution()
        else:
            solution = self._get_default_solution()
        
        MOVEMENT_PERSIST = 6
        
        n = len(solution) * MOVEMENT_PERSIST
        x = np.zeros((n, 4))
        
        for (i, action) in enumerate(solution):
            idx_start = i * MOVEMENT_PERSIST
            idx_end = idx_start + MOVEMENT_PERSIST
            
            if action == 0:
                x[idx_start:idx_end, 0] = 1
                x[idx_start:idx_end, 1] = -1
            elif action == 1:
                x[idx_start:idx_end, 0] = -1
                x[idx_start:idx_end, 1] = 1
            elif action == 2:
                x[idx_start:idx_end, 2] = 1
                x[idx_start:idx_end, 3] = -1
            elif action == 3:
                x[idx_start:idx_end, 2] = -1
                x[idx_start:idx_end, 3] = 1
        
        self.x = x
        self.step = 0
        
    def _get_default_solution(self):
        ultra_start = [2, 1, 2, 1, 1, 1, 0, 3, 2, 1, 2, 1, 2, 1, 3, 3]
        
        efficient_cycle = [
            2, 2, 2, 2, 2, 2, 2, 2, 2,
            1, 2, 1, 2,
            3, 3, 3, 3, 3, 3, 3,
            0, 3, 3, 1, 3,
            2, 2, 2, 2, 2,
            1, 0, 3, 0,
            2, 2, 2, 3, 3, 3,
            0, 0, 3, 3, 0,
        ]
        
        return ultra_start + efficient_cycle * 400
        
    def act(self):
        if self.step < len(self.x):
            action = self.x[self.step]
            self.step += 1
        else:
            # Loop from beginning if we run out
            self.step = 0
            action = self.x[self.step]
            self.step += 1
            
        return action
    
    def observe(self, obs):
        self.obs = obs