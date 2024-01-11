import torch
import random
import numpy as np
from collections import deque
from game_class import Snake_Game
from reward_optimizer import RewardOptimizer
from model_dql import Linear_QNet, QTrainer
import torch.cuda
from matplotlib import pyplot as plt
import pickle



class Agent:
    def __init__(self, file_path=None, step_reward=-7, apple_reward=90, death_reward=-120, 
                 window_x=200, window_y=200, render=True, training=True, state_rep="onestep", reward_closer=0, backstep=False, device=None):
        self.MAX_MEMORY = 1600
        self.BATCH_SIZE = 32
        self.LR = 0.01
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Working on", self.device)
        self.n_games = 0
        self.epsilon = 1 if training else 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=self.MAX_MEMORY)  # popleft()
        self.step_reward, self.apple_reward, self.death_reward = step_reward, apple_reward, death_reward
        self.window_x, self.window_y, self.render = window_x, window_y, render
        self.is_training = training
        self.state_rep = state_rep
        self.reward_closer = reward_closer
        self.backstep = backstep
        # Assuming Linear_QNet and QTrainer are defined in the model
        self.game = Snake_Game(snake_speed=5000, render=self.render, kill_stuck=True, window_x=self.window_x, window_y=self.window_y,
                        apple_reward=self.apple_reward, step_punish=self.step_reward, death_punish=self.death_reward, state_rep=self.state_rep,
                        reward_closer=self.reward_closer)
        if state_rep == "onestep":
            self.input, self.output = 11, 3
        elif state_rep == "vector":
            self.input, self.output = 21, 4
        elif state_rep == "grid":
            self.input, self.output = len(self.game.grid()), 4
        self.model = Linear_QNet(self.input, 256, self.output).to(self.device) 
        self.file_path = file_path
        if self.file_path is not None:
            self.model.load_state_dict(torch.load(self.file_path))
            self.model.eval()
        self.trainer = QTrainer(self.model, lr=self.LR, gamma=self.gamma)
        self.epsilon_decay = 0.9995 if state_rep=='onestep' else 0.999998  # Decaying rate per game
        self.epsilon_min = 0.01 #if self.is_training else 0  # Minimum value of epsilon
        self.reward_optim = RewardOptimizer('src\Classes\optim_of_tab_q-learn\metric_files\DQN_metric_test.txt')

    def get_state(self, game):
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE) # list of tuples
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
        if self.is_training: self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)  # decay epsilon
        
        final_move = [0,0,0] if self.state_rep == "onestep" else [0,0,0,0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)  # Assuming 3 actions
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device).clone().detach()
            prediction = self.model(state_tensor)
            if self.backstep == True and not self.state_rep == 'onestep':
                available_moves = np.array((0,0,0,0))
                if self.game.direction == "UP": available_moves[1] = 1
                elif self.game.direction == "DOWN": available_moves[0] = 1
                elif self.game.direction == "RIGHT": available_moves[3] = 1
                elif self.game.direction == "LEFT": available_moves[2] = 1
                index = np.argmax(available_moves)
                prediction[index] = -10000
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


    def train(self, rounds_to_play=False, plot_file_path=None):
        if plot_file_path is not None:
            scores_to_plot = []
            mean_score_to_plot = []
        high_score = -1
        c = 0
        step_counter = 0
        if self.game.toggle_possible: 
            import pygame
            pygame.init()
            self.game.toggle_rendering()
        game_toggle_score, game_toggle_runs, quitting = False, False, False
        toggle_epsilon, toggle_highscore = False, False
        while True:
            step_counter += 1
            if self.game.toggle_possible:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:  # Detect spacebar press
                            self.game.toggle_rendering()  # Toggle rendering and speed
                        elif event.key == pygame.K_s:
                            game_toggle_score = not game_toggle_score
                        elif event.key == pygame.K_r:
                            game_toggle_runs = not game_toggle_runs
                        elif event.key == pygame.K_q:
                            quitting = True
                            print("\nBAILING - force pushing metrics...\n")
                            game_toggle_runs,game_toggle_score =False,False
                        elif event.key == pygame.K_UP:
                            self.game.snake_speed += 5
                        elif event.key == pygame.K_DOWN:
                            self.game.snake_speed -=5
                        elif event.key == pygame.K_e:
                            toggle_epsilon = not toggle_epsilon
                        elif event.key == pygame.K_h:
                            toggle_highscore = not toggle_highscore

            # if agent.n_games == 2000:
            #     game = Snake_Game(snake_speed=50, render=True, kill_stuck=True, window_x=300, window_y=300,
            #               apple_reward=250, step_punish=-0.1, snake_length=4, death_punish=-75)
            state_old = self.get_state(self.game)
            final_move = self.get_action(state_old)
            self.game.move(final_move)  # make a move
            time_taken = self.game.has_apple()
            curr_score = self.game.score
            done = self.game.is_game_over()
            self.reward_optim.get_metrics(curr_score, time_taken, done)
            if self.is_training: 
                reward = self.game.get_reward()
                state_new = self.get_state(self.game)
                self.train_short_memory(state_old, final_move, reward, state_new, done)
                self.remember(state_old, final_move, reward, state_new, done)
            if done:
                if plot_file_path is not None:
                    scores_to_plot.append(curr_score)
                    mean_score_to_plot.append(sum(scores_to_plot)/len(scores_to_plot))
                game_number = self.game.get_game_count()
                if rounds_to_play: 
                    if rounds_to_play == game_number: quitting = True
                self.n_games += 1
                if self.is_training: self.train_long_memory()
                if curr_score > high_score:
                    high_score = curr_score
                    print("Highscore!", high_score)
                    self.model.save(index="11_states_negative")
                if game_number % 100 == 0:
                    c += 100
                    print(c, "GAMES")
                if game_toggle_score or game_toggle_runs:
                    print(f"RUN: {game_number}"*game_toggle_runs,f"SCORE: {curr_score}"*game_toggle_score)
                if toggle_highscore or toggle_epsilon:
                    print(f"HIGHSCORE: {high_score}"*toggle_highscore,f"EPSILON: {self.epsilon}"*toggle_epsilon)

                if game_number % 250 == 0 or quitting:
                    self.reward_optim.clean_data(look_at=None)
                    model_metrics = self.reward_optim.calculate_metrics()
                    self.reward_optim.commit(0, 100, model_metrics, "NONE", 
                                    "NONE", "NONE", "NONE", "NONE")
                    print(f"METRICS - Score: {model_metrics[0]} Time Between Apples: {model_metrics[1]}\n")
                    self.reward_optim.push()
                    print(len(self.reward_optim.scores))
                    if self.is_training: self.reward_optim.clear_commits()
                    print("METRICS PUSHED")
                    with open(plot_file_path, "wb") as f:
                        pickle.dump((scores_to_plot, mean_score_to_plot),f)
                    if quitting: break


                # print('Game', agent.n_games, 'Score', score, 'Record:', record)

                # plot_scores.append(curr_score)
                # total_score += curr_score
                # mean_score = total_score / agent.n_games
                # plot_mean_scores.append(mean_score)
                # plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    plot_file_path = 'src\Classes\DQL_PLOT\TEST_PLOTS\plot_file.txt'
    agent = Agent(render=True, training=True)
    agent.train(250, plot_file_path=plot_file_path)
    with open(plot_file_path, 'rb') as f:
        data = pickle.load(f)
        scores = data[0]
        mean_scores = data[1]
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.show()