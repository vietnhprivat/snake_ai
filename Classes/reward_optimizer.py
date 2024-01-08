from collections import namedtuple
import numpy as np

#### Dette er RewardOptimizeren. Den bruges til at holde styr på hvordan modellerne klarer sig.
# undervejs i et run vil den holde styr på gennemsnitsscore og tid. Når modellen/modellerne er trænede,
# kan den gemme data for modellerne i en fil med det angivne filnavn.
# Meningen med denne class er at kunne beregne data for både Q- og DQL-modellen.

class RewardOptimizer():
    def __init__(self, file_path):

        # File path er der, hvor alle modellers metrics til sidst gemmes
        self.file_path = file_path

        self.scores = []
        self.time_between_apples = []

        #Opretter en names tuple til informationer om modeller, vi vil gemme.
        self.metrics = namedtuple('metrics',
                                  ('index','mean_score','mean_time_apple', 'runs', 'file_path', 
                                   'step_punish', 'apple_reward', 'death_punish'))
        
        #Stack er en liste over alle modellers metrics
        self.stack = []
        self.look_at = None

        # Clean data kaldes inden metrics udregnes. Den udvælger hvor mange, af de sidste
        # runs, der skal tages i betragtning når vi udregner gns
    def clean_data(self,look_at):
        self.look_at = look_at
        self.scores = self.scores[look_at:]
        self.time_between_apples = self.time_between_apples[look_at:]

        
        # Funktionen nedenfor kaldes under træning af model. 
        # Tilføjer score og tid mellem æbler til lister når det er relevant
    def get_metrics(self, score, time, game_over):
        if not type(self.time_between_apples) == list:
            self.time_between_apples, self.scores = self.time_between_apples.tolist(), self.scores.tolist()
        self.time_between_apples.append(time)
        if game_over: self.scores.append(score)
        if self.time_between_apples: 
            if self.time_between_apples[-1] is None: 
                self.time_between_apples.pop()

        # Kaldes når træning er færdig. Udregner gennemsnitlig score og tid
    def calculate_metrics(self):
        self.scores = np.array(self.scores)
        self.time_between_apples = np.array(self.time_between_apples)
        if len(self.scores) == 0: mean_score = 0
        else: mean_score = np.mean(self.scores)
        if len(self.time_between_apples) == 0: mean_time_apple = 0
        else: mean_time_apple = np.mean(self.time_between_apples)
        score_var = np.var(self.scores)
        apple_var = np.var(self.time_between_apples)
        if self.look_at is not None:
            KI_score = [mean_score - 1.96* np.sqrt(score_var)/np.sqrt(self.look_at), mean_score + 1.96* np.sqrt(score_var)/np.sqrt(self.look_at)]
            KI_apple = [mean_time_apple - 1.96* np.sqrt(apple_var)/np.sqrt(self.look_at), mean_time_apple + 1.96* np.sqrt(apple_var)/np.sqrt(self.look_at)]
            return KI_score, KI_apple
        return mean_score, mean_time_apple

        # Tilføjer modellens udregnede metrics til en stack, der senere skubbes til filen.
        # Sender også data om spil/model som fx de rewards, der blev brugt
    def commit(self, index, runs, calculated_metrics, file_path, step_punish, apple_reward, death_punish):
        to_append = self.metrics(index, calculated_metrics[0], calculated_metrics[1], runs, file_path, 
                                 step_punish, apple_reward, death_punish)
        self.stack.append(to_append)

    def clear_commits(self):
        self.stack = []

        #Push sender stacken til en fil
    def push(self):
        with open(self.file_path, "a") as f:
            for model in self.stack:
                f.write(f"{model}\n")
