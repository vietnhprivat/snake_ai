from game_class import Snake_Game
from reward_optimizer import RewardOptimizer
from ql_tab import Q_learning
import random
from collections import namedtuple

#### Denne fil træner adskillige modeller og benytter sig af RewardOptimizeren til at skrive data.
# ModelOptimizeren kan desuden behandle data fra filerne.
# Tanken er også, at denne kan bruges på DQL-modellen. Hvis den skal bruges på DQL-modellen,
# Kan man eventuelt ændre, så rewards ikke vælges tilfældigt, det kan ændres i funktionen Train_Models.
# Når vi har en træningsfunktion for DQL-modellen, kan vi skifte Q-modellens træningsfunktion ud med DQL-modellens.
# Hvis data gemmes i samme format, kan denne også bruges uden at træne, men blot som databehandler.

class ModelOptimizer():
    def __init__(self, models_to_train, metric_folder_path='src\Classes\optim_of_tab_q-learn\metric_files\metric_test.txt', 
                 model_folder_path='src\Classes\optim_of_tab_q-learn\model_files\\'):
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

        # Træner modeller. double_check kan vælges til True hvis du ikke vil risikere at sætte en stor sim igang.
        # Hvis double check er slået til, får man også mulighed for selv at vælge rewards.
    def Train_Models(self,runs,runs_to_look_at, double_check=False):
        apple_choice, step_choice = False, False
        if double_check:
            answer = input("Er du sikker på at du vil træne nu? Skriv \'nej' for at annullere, ellers skriv \'ja\': ").lower()
            if answer == "nej": return
            answer = input("Vælg reward for æble. Skriv \'skip\' for at vælge tilfældigt: ").lower()
            if answer == "skip" or answer == "s": pass
            else: 
                apple_choice = int(answer)
                answer = input("Vælg reward for step. Skriv \'skip\' for at vælge tilfældigt: ").lower()
                if answer == "skip" or answer == "s": pass
                else: step_choice = int(answer)


        try:
            with open(self.metric_folder_path, "r") as f:
                data = f.read().splitlines()
                models_in_file = len(data)
        except:
            models_in_file = 0
        for i in range(self.models_to_train):

            # Vælg tilfældig straf for at tage et skridt og tilfældig reward for at fange æble
            step_reward = step_choice if step_choice else -random.randint(-10,10)
            apple_reward = apple_choice if apple_choice else random.randint(1,100)
            death_reward = - random.randint(1,150)
            closer_reward = random.randint(0, 10)

            # Laver en path, hvor selve modellen gemmes.
            model_file_path = f'model_{i+models_in_file}_step_{step_reward}_apple_{apple_reward}_closer_{closer_reward}_death_{death_reward}.pkl'
            model_file_path = f'{self.model_folder_path}{model_file_path}'

            # Initialiserer et spil med de valgte rewards og en model, der træner.
            game = Snake_Game(kill_stuck=True,render=False, step_punish=step_reward, apple_reward=apple_reward, window_x=200, window_y=200,
                              death_punish=death_reward, reward_closer=closer_reward)
            model = Q_learning(game, training=True, file_path=model_file_path)
            # Fortæl, hvilken model, vi arbejder på, og træn den. ser lige nu på de sidste 250 runs.
            print(f"Training model {i}. Step reward: {step_reward}. Apple_reward: {apple_reward}.\n")
            model.train(runs, Optimizer=self.optimizer, look_at=runs_to_look_at, index=i+models_in_file)
            # Når alle modeller er trænet, laver optimizeren en fil, hvor gns score, tid osv. for alle modeller gemmes
            self.optimizer.push()


        # Datahelper hjælper med at konvertere info fra datafil fra string til float
        # Type er som udgangspunkt float, men kan ændres til andet
    def datahelper(self, pointer, model, type=float, seperator=","):
        # Data ligger i fil som strings.
        val = ""

        # Vi starter ved vores pointer og går igennem alle karakterer i linjen fra der hvor pointeren nu er.
        for num in model[pointer:]:
            # Tjek om den karakter, vi kigger på er et komma. Hvis det ikke er, skal vi
            # tilføje karakteren til vores output og flytte pointeren én hen. Hvis ikke,
            # er det fordi at der er et adskildelseskomma, og vi returnerer den værdi, vi har lavet, og hvor pointeren nu er.
            # Hvis vi er ved slutningen af linjen, skal vi også returnere.
            if not num == seperator and not pointer == len(model)-1: 
                val += num
                pointer +=1
            else: 
                if not type == list: return type(val), pointer
                else:
                    index_0 = ""
                    com = -1
                    for i, char in enumerate(val[1:]):
                        if char == ",": 
                            com = i + 1
                            break
                        index_0 += char
                    index_0 = float(index_0)
                    index_1 = ""
                    for char in val[com+1:]:
                        if char == "]": break
                        index_1 += char
                    index_1 = float(index_1)
                    return [index_0,index_1], pointer


    def get_data(self):
        with open(self.metric_folder_path, "r") as f:
            data_raw = f.read().splitlines()
        data = []
        for model in data_raw:
            # Pointer starter som 14, hvor første værdi begynder
            pointer = 14
            index, pointer = self.datahelper(pointer,model)
            # Pointer er nu indekset af kommaet, der adskilder index og mean. Der er 13 indtil score begynder
            pointer += 13
            if model[pointer] == "[":
                mean_score, pointer = self.datahelper(pointer,model, type=list, seperator="]")
                pointer += 19
            else: 
                mean_score, pointer = self.datahelper(pointer,model)
                pointer += 18
            if model[pointer] == "[":
                mean_time, pointer = self.datahelper(pointer,model, type=list, seperator="]")
                pointer += 8
            else: 
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
            data.append([index, mean_score, mean_time, runs, file_path, step_punish, apple_reward, death_punish])
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
            best.append([f"STEP Reward: {model[5]}", f"APPLE Reward: {model[6]}", f"DEATH Reward: {model[7]}", 
                         f"SCORE: {model[1]} ", f"TIME: {model[2]}", f"Model number: {model[0]}"])
        return best

    ## Removes duplicates from files
    def remove_duplicates(self, path_for_data_input = "INSERT PATH HERE", path_for_data_output = "INSERT PATH HERE"):
        with open(path_for_data_input, "r") as file_in:
            data_in_file = file_in.readlines()
            if data_in_file and not data_in_file[-1].endswith("\n"):
                data_in_file[-1] += "\n"
            remove_duplicates_from_data_in_file = set(data_in_file)
        with open(path_for_data_output, "w") as file_out:
            for items in remove_duplicates_from_data_in_file:
                file_out.write(str(items))
        print("Duplicates are removed from file")


if __name__ == "__main__":
    # Initialiserer en Optimizer. Tager som argument, hvor mange forskellige modeller, den skal træne.
    model_optimizer = ModelOptimizer(10) #model_folder_path='/zhome/db/e/206305/snake_ai/src/Classes/TQL/model_files/', 
                                      #metric_folder_path='/zhome/db/e/206305/snake_ai/src/Classes/TQL/metric_files/metric_test.txt') 

    # Træner modeller, argumenter er ant. træningsruns og ant. runs, der laves beregninger på. 
    # Slå double_check fra for bare at træne
    model_optimizer.Train_Models(400,300)

    # Tager på nuværende tidspunkt SCORE eller TIME som input og sorterer modellerne efter dem, der er bedst
    # på den parameter
    model_optimizer.sort_data("SCORE") 

    # Tager de top X bedste modeller indenfor den valgte parameter og viser dem.
    top_rewards = model_optimizer.get_rewards(10)  
    for rewards in top_rewards:
        print(rewards)