import torch
import random
import numpy as np
from collections import deque
from game_class import Snake_Game
from reward_optimizer import RewardOptimizer
from model_testing import Linear_QNet, QTrainer
from helper_testing import plot

MAX_MEMORY = 10_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        # Assuming Linear_QNet and QTrainer are defined in the model
        self.model = Linear_QNet(11, 256, 3)  
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # Extracting the head of the snake and the fruit position
        head = game.snake_body[0]
        fruit = game.fruit_position

        # Constructing the danger directly ahead, to the left and to the right
        danger_straight = game.update_danger(head, game.window_x, game.window_y, game.snake_body)
        danger_left = danger_straight[-1:] + danger_straight[:-1]  # rotating the danger list to left
        danger_right = danger_straight[1:] + danger_straight[:1]  # rotating the danger list to right

        # Current direction of the snake
        direction = game.update_direction(game.direction)

        # Food location relative to the snake head
        food_dir = game.update_fruit(head, fruit)

        state = [
            # Danger straight
            danger_straight[0],
            # Danger right
            danger_right[0],
            # Danger left
            danger_left[0],
            # Current direction
            *direction,
            # Food direction
            *food_dir
        ]

        return np.array(state, dtype=int)

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
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 1000 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Snake_Game(snake_speed=150, render=False, kill_stuck=True, window_x=300, window_y=300,
                      apple_reward=95, step_punish=-0.5, snake_length=4)
    reward_optim = RewardOptimizer('Classes\optim_of_tab_q-learn\metric_files\DQN_metric_test.txt')
    high_score = -1
    c = 0
    while True:
        if agent.n_games == 50:
            game = Snake_Game(snake_speed=50, render=True, kill_stuck=True, window_x=300, window_y=300,
                      apple_reward=95, step_punish=-0.5)
        print(game.get_game_count())
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
        if done and curr_score > high_score:
            high_score = curr_score
            print("Highscore!", high_score)
        if done and game.get_game_count() % 100 == 0:
            c += 100
            print(c, "GAMES")

        if game.get_game_count() % 250 == 0 and done:
            reward_optim.clean_data(50)
            model_metrics = reward_optim.calculate_metrics()
            reward_optim.commit(0, 100, model_metrics, "NONE", 
                             "NONE", "NONE", "NONE")
            print("METRICS:",model_metrics)
            reward_optim.push()
            reward_optim.clear_commits()
            print("PUSHED")

        if done:
            #game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            score = game.score
            if score > record:
                record = score
                agent.model.save()

            #print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # plot_scores.append(curr_score)
            # total_score += curr_score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()