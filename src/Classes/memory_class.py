from collections import namedtuple, deque
from game_class import Snake_Game
import random

class ExperienceReplayBuffer():
    def __init__(self, memory_length, number_of_samples):
        self.memory = deque(maxlen=memory_length)
        self.number_of_samples = number_of_samples
        self.Transition = namedtuple('Transition',
                                     ('curr_state', 'action', 'reward', 'next_state'))

    def push(self, s1, action, reward, s2):
        self.memory.append(self.Transition(s1, action, reward, s2))

    def __len__(self):
        return len(self.memory)
    
    def sample(self):
        return random.sample(self.memory, self.number_of_samples)
    

#Lille test    
# Skal have ændret på action så det er som i q-learning
if __name__ == "__main__":
    game = Snake_Game(render=False, kill_stuck=True, step_punish=-10, apple_reward=90, death_punish=-100)
    n = 2
    buffer = ExperienceReplayBuffer(2000, 50)
    while game.get_game_count() < n:
        s1 = game.get_state()
        game.move()
        action = game.get_move()
        game.has_apple()
        game_over = game.is_game_over()
        reward = game.get_reward()
        s2 = game.get_state()
        buffer.push(s1,action,reward,s2 if not game_over else None)
    
    for element in buffer.sample():
        print(element)