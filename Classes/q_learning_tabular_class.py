from game_class import Snake_Game, Data
from reward_optimizer import RewardOptimizer
from collections import defaultdict, namedtuple
import numpy as np
import pickle
import random

#### Dette er Q-learning classen. Den skaber en model der kan trænes og gemmer Q-tabellen i en fil
# med det angivne filnavn.
     
class Q_learning:
    def __init__(self, game, training=False, file_path='peters_pickle.pkl', gamma=0.9, buffer=None):
        self.game = game
        self.action_space = self.game.action_space
        self.Q = defaultdict(lambda: [0., 0., 0.])
        self.training = training
        self.file_path = file_path
        self.gamma = gamma

        ## Hvis den ikke skal træne, åbner den filen med navnet som er angivet
        if not self.training:
            try:
                with open(self.file_path, 'rb') as f:
                    loaded_Q = pickle.load(f)
            except FileNotFoundError:
                print("Model fil findes ikke. Prøv en anden path eller tjek at path er deklereret i init af modellen.")
            self.Q.update(loaded_Q)
        self.score = 0
        self.buffer = buffer

    def get_q_current(self):
        state = self.game.get_state()
        return self.Q[tuple(state[0]),tuple(state[1]),tuple(state[2])]
    
    def get_action(self):
        return np.argmax(self.get_q_current())
    
    def update(self, qcurr, action, qnew):
        qcurr[action] = self.game.get_reward() + self.gamma * np.max(qnew)
        
    def get_action_index(self,action):
            return self.action_space[action]
    

    # Træningsfunktion:
    # Tager som input: hvor mange runs(obligatorisk).
    # Hvis man vil have data på forskellige rewards, kan man angive hvilken optimizer, man vil bruge.
    # Optimizeren er et objekt fra vores reward_optimizer fil, RewardOptimizer()
    # Hvis man angiver en optimizer, skal man også angive et index for modellen, og hvor mange
    # af de sidste runs, man vil se på (look_at), når man udregner gennemsnitlig score og tid
    def train(self,runs,show_loading=True, Optimizer=None, look_at = None, index=None):
        while self.game.get_game_count() < runs:
            # Kigger på nuværende state og vælger en action:
            qcurrent = self.get_q_current()
            action = self.get_action()

            # Fortæller spillet at den vil tage den action
            self.game.move(self.get_action_index(action))

            # Variable for hvor lang tid, der er gået siden æblefangst og bool for om spillet er slut
            # Funktionerne påvirker spillet og er derfor ikke mulige at udelade.
            time_taken, game_over = self.game.has_apple(), self.game.is_game_over_bool()

            # Tjekker om der er en optimizer. Hvis der er, sender den spillets informationer
            # Til den
            if Optimizer is not None:
                Optimizer.get_metrics(self.game.score,time_taken,game_over)

            # Funktion, der tjekker om spillet er slut, og genstarter spillet osv. hvis det er
            self.game.is_game_over()

            # Det her er bare en loading funktion, der viser hvor langt, vi er
            if show_loading and game_over and self.game.get_game_count() % (runs/20) == 0: 
                print(f"{round(self.game.get_game_count()/(runs/100))}%")
            
            # Får den nye state og opdaterer Q-tabel    
            qnew = self.get_q_current()
            self.update(qcurrent,action,qnew)

        # Hvis der er en optimizer, udvælger den de sidste look_at runs, udregner gns, og tilføjer dem
        # til en liste. Se dokumentation i reward_optimizer.py
        if Optimizer is not None:
            Optimizer.clean_data(look_at)
            model_metrics = Optimizer.calculate_metrics()
            Optimizer.commit(index, runs, model_metrics, self.file_path, 
                             self.game.punish_no_apple, self.game.reward_apple, self.game.punish_death)
            
        # Når træning er slut, gemmes Q-tabellen i en pickle-fil med den tidligere angivne path
        with open(self.file_path, 'wb') as f:
            pickle.dump(dict(self.Q), f)
             

             
        
          

if __name__ == "__main__":
    game = Snake_Game(snake_speed=150,kill_stuck=True, render=True)
    buffer = Data()
    model = Q_learning(game, buffer=buffer, training=False, file_path='Classes\optim_of_tab_q-learn\model_files\model_15_step_-10_apple_93.pkl')
    model.train(10)