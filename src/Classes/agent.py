import torch
import random
import numpy as np
from collections import deque
from game_class import Snake_Game
from reward_optimizer import RewardOptimizer
from model_dql import Linear_QNet, QTrainer
from helper_dql import plot
import torch.cuda



MAX_MEMORY = 1600
BATCH_SIZE = 32
LR = 0.01

class Agent:
    def __init__(self):
        self.device = 'cpu' #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Working on", self.device)
        self.n_games = 0
        self.epsilon = 1  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        # Assuming Linear_QNet and QTrainer are defined in the model
        self.model = Linear_QNet(11, 256, 3).to(self.device) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.epsilon_decay = 0.99  # Decaying rate per game
        self.epsilon_min = 0.01  # Minimum value of epsilon

    def get_state(self, game):
        # return game.grid()
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Update epsilon value
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)  # decay epsilon
        
        final_move = [0,0,0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)  # Assuming 3 actions
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device).clone().detach()
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    record = 0
    plot_mean_scores = []
    plot_scores = []
    total_score = 0
    agent = Agent()
    game = Snake_Game(snake_speed=5000, render=True, kill_stuck=True, window_x=300, window_y=300,
                      apple_reward=90, step_punish=-7, snake_length=4, death_punish=-120)
    reward_optim = RewardOptimizer('src\Classes\optim_of_tab_q-learn\metric_files\DQN_metric_test.txt')
    high_score = -1
    c = 0
    step_counter = 0
    if game.toggle_possible: 
        import pygame
        pygame.init()
        game.toggle_rendering()
    game_toggle_score, game_toggle_runs, quitting = False, False, False
    while True:
        step_counter += 1
        if game.toggle_possible:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:  # Detect spacebar press
                        game.toggle_rendering()  # Toggle rendering and speed
                    elif event.key == pygame.K_s:
                        game_toggle_score = not game_toggle_score
                    elif event.key == pygame.K_r:
                        game_toggle_runs = not game_toggle_runs
                    elif event.key == pygame.K_q:
                        quitting = True
                        print("\nBAILING - force pushing metrics...\n")
                        game_toggle_runs,game_toggle_score =False,False
                    elif event.key == pygame.K_UP:
                        game.snake_speed += 5
                    elif event.key == pygame.K_DOWN:
                        game.snake_speed -=5

        # if agent.n_games == 2000:
        #     game = Snake_Game(snake_speed=50, render=True, kill_stuck=True, window_x=300, window_y=300,
        #               apple_reward=250, step_punish=-0.1, snake_length=4, death_punish=-75)
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        game.move(final_move)  # make a move
        # game.has_apple()
        time_taken, game_over = game.has_apple(), game.is_game_over_bool()
        curr_score = game.score
        reward_optim.get_metrics(game.score, time_taken, game_over)


        done = game.is_game_over()
        reward = game.get_reward()
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        game_number = game.get_game_count()

        if done:
            agent.n_games += 1
            agent.train_long_memory()
            if curr_score > high_score:
                high_score = curr_score
                print("Highscore!", high_score)
                agent.model.save()
            if game_number % 100 == 0:
                c += 100
                print(c, "GAMES")
            if game_toggle_score or game_toggle_runs:
                print(f"RUN: {game_number}"*game_toggle_runs,f"SCORE: {curr_score}"*game_toggle_score)

            if game_number % 250 == 0 or quitting:
                reward_optim.clean_data(look_at=None)
                model_metrics = reward_optim.calculate_metrics()
                reward_optim.commit(0, 100, model_metrics, "NONE", 
                                "NONE", "NONE", "NONE")
                print(f"METRICS - Score: {model_metrics[0]} Time Between Apples: {model_metrics[1]}\n")
                reward_optim.push()
                reward_optim.clear_commits()
                print("METRICS PUSHED")
                if quitting: break


            # print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # plot_scores.append(curr_score)
            # total_score += curr_score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()