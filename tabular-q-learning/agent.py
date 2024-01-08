import torch
import random
import numpy as np
from collections import deque
from game_relative_directions import Snake_Game

max_memory = 100000
batch_size = 1000
lr = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0 # discount factor
        self.memory = deque(maxlen = max_memory) #popleft() fjern gammelt hukommelse
        self.model= None #TODO
        self.trainer = None #TODO



        # TODO: model, trainer

    

    def get_state_agent(self, game):
        return game.get_state()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft


    def train_long_memory(self):
        if len(self.memory) > batch_size:
            mini_sample = random.sample(self.memory, batch_size) # list of tuples
        else: 
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation: epsilon greedy
        self.epsilon = 80 - self.n_games # experimentel
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtyoe=torch.float)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item() # Converting tensor til int
            final_move[move] = 1
        
        return final_move



def train():
    plot_scores = []
    plot_mean = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Snake_Game()

    while True:
        # get current state
        state_old = agent.get_state_agent(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get state
        game.move(final_move)

        game.has_apple()
        done = game.game_over()
        reward = game.get_reward()
        score = game.score

        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done) 

        if done:
            # train long memory (replay memory)   
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record", record)

            # TODO plotting


if __name__ == '__main__':
    train()











# game = game_env(snake_length = 40)
# n = 1

# while game.get_game_count() < n:
#     s1 = game.get_state()
#     print(s1)
#     game.move()
#     action = game.get_move()
#     game.has_apple()
#     game_over = game.is_game_over()
#     reward = game.get_reward()
#     s2 = game.get_state()