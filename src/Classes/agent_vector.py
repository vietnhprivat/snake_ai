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


## Agent tager filepath som input hvis man vil køre en model, der allerede er trænet.
class Agent:
    def __init__(self, file_path=None, step_reward=-7, apple_reward=90, death_reward=-120, 
                 window_x=200, window_y=200, render=True, training=True, state_rep="onestep", reward_closer=0, backstep=False, 
                 device=None, epsilon_decay=0.99999, learning_rate=0.01, model_name='testing', epsilon_min=0.01, gamma=0.9):
        self.MAX_MEMORY = 1600 ## Længde af buffer
        self.BATCH_SIZE = 32 ## Sample størrelse
        self.LR = learning_rate ## Learning rate (TIDLIGERE 0.01 for onestep)
        if device is not None: ## Her kan man vælge at køre cpu selvom man har cuda
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Working on:", f"{self.device}".upper())
        self.epsilon = 1 if training else 0  ## Tilfældighed
        self.gamma = gamma  ## Discount faktor
        self.memory = deque(maxlen=self.MAX_MEMORY)  ## popleft() buffer
        self.model_name = model_name

        ## Definerer en masse variable baseret på __init__ input
        self.step_reward, self.apple_reward, self.death_reward = step_reward, apple_reward, death_reward
        self.window_x, self.window_y, self.render = window_x, window_y, render
        self.is_training = training
        self.state_rep = state_rep
        self.reward_closer = reward_closer
        self.backstep = backstep

        ## Initialisér et spil og model
        self.game = Snake_Game(snake_speed=5000, render=self.render, kill_stuck=True, window_x=self.window_x, window_y=self.window_y,
                        apple_reward=self.apple_reward, step_punish=self.step_reward, death_punish=self.death_reward, state_rep=self.state_rep,
                        reward_closer=self.reward_closer)
        if state_rep == "onestep":
            self.input, self.output = 11, 3
        elif state_rep == "vector":
            self.input, self.output = 21, 4
        elif state_rep == "grid":
            self.input, self.output = len(self.game.grid()), 4
        self.model = Linear_QNet(self.input, 256, self.output, self.device, 
                                 grid_mode=True if self.state_rep =='grid' else False).to(self.device) 

        self.file_path = file_path

        ## Hvis vi har en sti til en model, vil vi loade den i stedet for at træne en ny
        if self.file_path is not None:
            self.model.load_state_dict(torch.load(self.file_path, map_location=self.device))
            self.model.eval()

        ## Initialisér trainer
        self.trainer = QTrainer(self.model, lr=self.LR, gamma=self.gamma)

        ## Vores epsilon behøver ikke at være så stor for onestep, da repræsentationen af staten er så simpel.
        ## Den skal være højere for vector og grid repræsentation
        self.epsilon_decay = epsilon_decay  #0.9995 if state_rep=='onestep' else 0.999998  # Decaying rate per game
        self.epsilon_min = epsilon_min ## Minimumsværdi af epsilon

        ## Initialisér en rewardoptimizer til at gemme metrics
        # self.reward_optim = RewardOptimizer(f'src\Classes\optim_of_tab_q-learn\metric_files\DQN_{state_rep}_metrics.pkl')
        self.reward_optim = RewardOptimizer(f'/zhome/db/e/206305/snake_ai/src/Classes/optim_of_tab_q-learn/metric_files/DQN_{state_rep}_metrics_15_01.pkl')

        

        ## Får state. se game_class
    def get_state(self, game):
        return game.get_state()

        ## Gemmer state-actionpar til buffer
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        ## Træner long memory
    def train_long_memory(self):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        ## Træner short memory
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

        ## Bestem en action
    def get_action(self, state):
        ## Opdatér epsilon værdi
        if self.is_training: self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)  # decay epsilon

        ## Liste med actions, er 3 eller 4 lang afhængig af om vi spiller fra slangens perspektiv eller ej.
        final_move = [0,0,0] if self.state_rep == "onestep" else [0,0,0,0]

        ## Nogle gange vil vi tage en tilfældig action, især i starten. Gælder kun under træning.
        if random.random() < self.epsilon:
            move = random.randint(0, len(final_move)-1)  # Depending on action space
            final_move[move] = 1
        else:

            ## Laver state om til tensor og får en prediction fra modellen
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device).clone().detach()
            if self.state_rep == 'grid': state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
            prediction = self.model(state_tensor)
            ## Backstep vil algoritmisk give en meget lav q-værdi for at gå imod bevægelsesretningen.
            ## Det er i game_class ikke muligt at gå baglæns, så dette er for at undgå at forlæns og baglæns
            ## får samme q-værdi. Vi vil gerne have én entydig action fra modellen.
            if self.backstep == True and not self.state_rep == 'onestep':
                available_moves = np.array((0,0,0,0))
                if self.game.direction == "UP": available_moves[1] = 1
                elif self.game.direction == "DOWN": available_moves[0] = 1
                elif self.game.direction == "RIGHT": available_moves[3] = 1
                elif self.game.direction == "LEFT": available_moves[2] = 1
                index = np.argmax(available_moves)
                prediction[index] = -10000
            
            ## Vælger den action med den højeste Q-værdi.
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    ## Træningsfunktion. Kan tage rounds_to_play som input, så vil vi spille så mange spil.
    ## Hvis den efterlades tom, kører den ind til vi afslutter (hvis rendering er aktiveret)
    ## eller til at terminalen dræbes. Kan også tage en filsti til et sted hvor information gemmes
    ## til plotting. Kan så senere hentes.
    def train(self, rounds_to_play=False, plot_file_path=None):
        ## Gør klar til at gemme data for runs til fil der kan plotte
        if plot_file_path is not None:
            scores_to_plot = []
            step_per_game_list = []
            epsilon_list = []
        high_score = -1
        ## Hvis kan rendere, kan det slås til og fra
        if self.game.toggle_possible: 
            import pygame
            pygame.init()
            self.game.toggle_rendering()
        ## Flere variable
        game_toggle_score, game_toggle_runs, quitting = False, False, False
        toggle_epsilon, toggle_highscore = False, False
        step_per_game = 0
        total_steps = 0
        steps_max = 5_000_000
        while True:
            total_steps +=1
            ## Hvis det er muligt at rendere, går vi igennem knapper
            if self.game.toggle_possible:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:  # Mellemrum, rendering tændt
                            self.game.toggle_rendering() 
                        elif event.key == pygame.K_s: # s, viser score for hvert run
                            game_toggle_score = not game_toggle_score
                        elif event.key == pygame.K_r: # r, viser hvilket run, vi er på
                            game_toggle_runs = not game_toggle_runs
                        elif event.key == pygame.K_q: # q, afslutter simulation
                            quitting = True
                            print("\nBAILING - force pushing metrics...\n")
                            game_toggle_runs,game_toggle_score =False,False
                        elif event.key == pygame.K_UP: # pil op, øger hastigheden ved rendering
                            self.game.snake_speed += 5
                        elif event.key == pygame.K_DOWN:
                            self.game.snake_speed -=5 # pil ned, omvendte af ovenstående
                        elif event.key == pygame.K_e:
                            toggle_epsilon = not toggle_epsilon # e, viser epsilon ved hvert run
                        elif event.key == pygame.K_h:
                            toggle_highscore = not toggle_highscore # h, viser highscore kontinuerligt

            # Få state til at vælge en action og tag den action
            state_old = self.get_state(self.game)
            final_move = self.get_action(state_old)
            step_per_game += 1
            self.game.move(final_move)  # make a move

            ## Gem hvor lang tid, der er gået siden sidste æble, få nuværende score og tjek om spillet er slut
            ## Hvis spillet er slut, nulstilles variable og genstarter, hvis vi er på et æble, spawner et nyt
            ## Se evt. dokumentation i game_class
            time_taken = self.game.has_apple()
            curr_score = self.game.score
            done = self.game.is_game_over()

            ## Gemmer score, tid, og om vi er døde til optimizeren, der senere kan lave statistik på det
            self.reward_optim.get_metrics(curr_score, time_taken, done)

            ## Tjek om vi træner eller bare bruger en tidligere model. Få reward, state, og så træn på det
            if self.is_training: 
                reward = self.game.get_reward()
                state_new = self.get_state(self.game)
                self.train_short_memory(state_old, final_move, reward, state_new, done)
                self.remember(state_old, final_move, reward, state_new, done)

            ## Hvis vi er færdige: gem data for vores run til potentiel plotting
            if done:
                game_number = self.game.get_game_count()
                if self.is_training: self.train_long_memory()
                if curr_score > high_score:
                    high_score = curr_score
                    print("Highscore!", high_score)
                    self.model.save(index=self.model_name)
                if plot_file_path is not None:
                    scores_to_plot.append(curr_score)
                    step_per_game_list.append(step_per_game)
                    step_per_game = 0
                    epsilon_list.append(self.epsilon)

                ## Tjek om vi har spillet de runs, vi gerne ville, hvis det er indstillet. Hvis vi har, gør klar til
                ## at afslutte
                if rounds_to_play: 
                    if rounds_to_play == game_number: quitting = True
                ## Træn hvis vi træner, log potentiel highscore

                ## Hvis vi er på et multipel af 100 spil, giv noget grundlæggende information om hvordan, vi klarer os.
                if game_number % 100 == 0:
                    print(game_number, "GAMES")
                if game_toggle_score or game_toggle_runs:
                    print(f"RUN: {game_number}"*game_toggle_runs,f"SCORE: {curr_score}"*game_toggle_score)
                if toggle_highscore or toggle_epsilon:
                    print(f"HIGHSCORE: {high_score}"*toggle_highscore,f"EPSILON: {self.epsilon}"*toggle_epsilon)

                ## Hvis vi er på et multipel af 250 eller har afsluttet, gemmer vi KI for data, gemmer plotting info
                ## til plotting fil og breaker loopet
                if game_number % 250 == 0 or quitting:
                    self.reward_optim.clean_data(look_at=None)
                    model_metrics = self.reward_optim.calculate_metrics()
                    self.reward_optim.commit(game_number/250, game_number, model_metrics, self.file_path, 
                             self.game.punish_no_apple, self.game.reward_apple, self.game.punish_death, self.game.reward_closer)
                    self.reward_optim.push()
                    if self.is_training: self.reward_optim.clear_commits()
                    print("METRICS PUSHED")
                    if plot_file_path is not None:
                        with open(plot_file_path, "wb") as f:
                            pickle.dump((scores_to_plot, step_per_game_list, epsilon_list),f)
                    if quitting: break
            
            if total_steps == steps_max: 
                quitting = True
                print("Steps hit, quitting")

if __name__ == '__main__':
    ## Fil til plotting information
    # plot_file_path = 'src\Classes\DQL_PLOT\TEST_PLOTS\plot_file.pkl'
    plot_file_path = '/zhome/db/e/206305/snake_ai/src/Classes/DQL_PLOT/TEST_PLOTS/plot_file_vector_15_01.pkl'
    ## Initialisér agent
    agent = Agent(state_rep='vector', apple_reward=68, step_reward=8, death_reward=-112, reward_closer=2, render=False, epsilon_decay=0.9999985,
                  learning_rate=0.0001, model_name="vector_15_01")
    agent.train(plot_file_path=plot_file_path)