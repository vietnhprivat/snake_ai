import torch
import random
import numpy as np
from collections import deque
from game_relative_directions import Snake_Game
from model import Linear_QNet, QTrainer
from helper import plot
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

max_memory = 10000
batch_size = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount factor skal være mindre end 1
        self.memory = deque(maxlen = max_memory) #popleft() fjern gammelt hukommelse
        self.model= Linear_QNet(11, 256, 3) #TODO
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state_agent(self, game):
        
        # Grid metode state space = w_x * w_y. f.eks. 20 x 20 = 400.  
        # return game.grid()

        # # 11 states metode
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
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # Converting tensor til int
            final_move[move] = 1
        
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = Snake_Game(window_x = 200, window_y = 200, snake_speed = 50, render=False, apple_reward=10, step_punish=0, death_punish= -10)
    agent = Agent()


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

        state_new = agent.get_state_agent(game)

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
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            


if __name__ == '__main__':
    train()

