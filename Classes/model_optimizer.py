from game_class import Snake_Game
from reward_optimizer import RewardOptimizer
from q_learning_tabular_class import Q_learning
import random
from collections import namedtuple

#### Denne fil træner adskillige modeller og benytter sig af RewardOptimizeren til at skrive data.
# ModelOptimizeren kan desuden behandle data fra filerne.
# Tanken er også, at denne kan bruges på DQL-modellen. Hvis den skal bruges på DQL-modellen,
# Kan man eventuelt ændre, så rewards ikke vælges tilfældigt, det kan ændres i funktionen Train_Models.
# Når vi har en træningsfunktion for DQL-modellen, kan vi skifte Q-modellens træningsfunktion ud med DQL-modellens.
# Hvis data gemmes i samme format, kan denne også bruges uden at træne, men blot som databehandler.

class ModelOptimizer():
    def __init__(self, models_to_train, metric_folder_path='Classes\optim_of_tab_q-learn\metric_files\metric_test.txt', 
                 model_folder_path='Classes\optim_of_tab_q-learn\model_files\\'):
        # Models_to_train er hvor mange forskellige modeller, der skal trænes
        self.models_to_train = models_to_train

        # Den sti, hvor informationer om forskellige modeller ender.
        self.metric_folder_path = metric_folder_path

        # Stien til en MAPPE, hvor modellerne vil blive gemt
        self.model_folder_path = model_folder_path

        # Initialiserer et optimizer-objekt. Den angivne path er der, hvor de forskellige modeller
        # parametre gemmes, gns score og gns tid
        self.optimizer = RewardOptimizer(metric_folder_path)

        self.sorted_data = None

        # Funktion til at træne flere modeller og gamme metrics. runs er antallet af spil, hver
        # model skal spille. runs_to_look_at er hvor mange, af de seneste spil, der skal
        # tages højder for i beregningerne
        self.rewards_when_sort = namedtuple('Rewards',
                                            ('step_reward','apple_reward','death_reward'))

    def Train_Models(self,runs,runs_to_look_at):
        try:
            with open(self.metric_folder_path, "r") as f:
                data = f.read().splitlines()
                models_in_file = len(data)
        except:
            models_in_file = 0
        for i in range(self.models_to_train):

            # Vælg tilfældig straf for at tage et skridt og tilfældig reward for at fange æble
            step_reward = -random.randint(0,10)
            apple_reward = random.randint(10,100)

            # Laver en path, hvor selve modellen gemmes.
            model_file_path = f'model_{i+models_in_file}_step_{step_reward}_apple_{apple_reward}.pkl'
            model_file_path = f'{self.model_folder_path}{model_file_path}'

            # Initialiserer et spil med de valgte rewards og en model, der træner.
            game = Snake_Game(kill_stuck=True,render=False, step_punish=step_reward, apple_reward=apple_reward)
            model = Q_learning(game, training=True, file_path=model_file_path)
            # Fortæl, hvilken model, vi arbejder på, og træn den. ser lige nu på de sidste 250 runs.
            print(f"Training model {i}. Step reward: {step_reward}. Apple_reward: {apple_reward}.\n")
            model.train(runs, Optimizer=self.optimizer, look_at=runs_to_look_at, index=i+models_in_file)
        # Når alle modeller er trænet, laver optimizeren en fil, hvor gns score, tid osv. for alle modeller gemmes
        self.optimizer.push()

    def datahelper(self, pointer, model, type=float):
        val = ""
        for num in model[pointer:]:
            for num in model[pointer:]:
                if not num == "," and not pointer == len(model)-1: 
                    val += num
                    pointer +=1
                else: return type(val), pointer

    def get_data(self):
        with open(self.metric_folder_path, "r") as f:
            data_raw = f.read().splitlines()
        data = []
        for model in data_raw:
            # Finder modellens index. Første modelindextal har indexet 14 i strengen.
            pointer = 14
            index, pointer = self.datahelper(pointer,model)
            # Pointer er nu indekset af kommaet, der adskilder index og mean. Der er 13 indtil score begynder
            pointer += 13
            mean_score, pointer = self.datahelper(pointer,model)
            pointer += 18
            mean_time, pointer = self.datahelper(pointer,model)
            pointer += 7
            runs, pointer = self.datahelper(pointer,model)
            pointer += 12
            file_path, pointer = self.datahelper(pointer,model, type=str)
            pointer += 14
            step_punish, pointer = self.datahelper(pointer, model)
            pointer += 15
            apple_reward, pointer = self.datahelper(pointer,model)
            pointer += 15
            death_punish, pointer = self.datahelper(pointer,model)
            data.append((index, mean_score, mean_time, runs, file_path, step_punish, apple_reward, death_punish))
        return data
    
    def sort_data(self, index_to_sort_by):
        # Tager string med SCORE eller TIME som input, og sorterer derefter
        if index_to_sort_by == "SCORE": index = 1
        elif index_to_sort_by == "TIME": index = 2
    
        self.sorted_data = self.get_data()

        # Sorterer ud fra valgt sorteringsmetode.
        # Reverse sort er True hvis index er 1, altså hvis vi kigger på score. Hvis vi kigger på tid, er den false,
        # så lavere tider kommer først.
        self.sorted_data.sort(key=lambda x: x[index], reverse=index==1)
        return self.sorted_data
    
    def get_rewards(self, top_how_many=10):
        # Sørger for at man ikke kan vælge flere end der er
        top = top_how_many if not top_how_many > len(self.sorted_data) else len(self.sorted_data)
        best = []

        # Laver en named tuple med de rewards, der er blevet brugt
        for model in self.sorted_data[:top+1]:
            best.append([self.rewards_when_sort(model[5], model[6], model[7]), f"SCORE: {round(model[1])} ", 
                         f"TIME: {round(model[2])}", f"Model number: {model[0]}"])
        return best

if __name__ == "__main__":
    model_optimizer = ModelOptimizer(1)
    #model_optimizer.Train_Models(1200,400)
    model_optimizer.sort_data("SCORE")
    top_rewards = model_optimizer.get_rewards(10)
    for rewards in top_rewards:
        print(rewards)