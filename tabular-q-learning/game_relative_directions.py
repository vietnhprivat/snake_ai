import pygame
import random
import numpy as np
from collections import namedtuple

class Snake_Game():
    def __init__(self, render=True, write_data = False, apple_reward = 50, step_punish = -1, death_punish = -100, 
                 window_x = 720, window_y = 480, snake_speed = 15, snake_length = 4, force_write_data = False, kill_stuck = True):
        self.snake_speed = snake_speed

        # Window size
        self.window_x = window_x
        self.window_y = window_y

        # defining colors
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)
        self.blue = pygame.Color(0, 0, 255)
        self.snake_length = snake_length
        self.kill_stuck = kill_stuck

        # Initialising pygame
        pygame.init()
        self.should_write_data = force_write_data if force_write_data else write_data
        self.force_write_data = force_write_data
        self.should_render = render
        self.game_count = 0
        # Initialise game window
        if self.should_render:
            pygame.display.set_caption('Slitherin Pythons')
            self.game_window = pygame.display.set_mode((self.window_x, self.window_y))

        # FPS (frames per second) controller
        self.fps = pygame.time.Clock()

        self.reset()
        self.curr_action = self.direction
        self.reward_apple, self.punish_no_apple, self.punish_death = apple_reward, step_punish, death_punish
        self.reward = 0
        self.action_space = [[1,0,0], [0,1,0], [0,0,1]]
        self.stuck_step = 0
        self.stuck = False

        #Æble-spawn funktion
    def spawn_apple(self, snake_coordinates):
        #Laver en liste med lister, hvor hvert element repræsenterer et koordinat:
        #Her får alle koordinater værdien 1
        grid = [[1 for _ in range(int(self.window_x/10))] for _ in range(int(self.window_y/10))]
        #Tager koordinaterne fra slangens krop og giver disse koordinater værdien 0
        for x,y in snake_coordinates: grid[int(y/10)][int(x/10)] = 0
        #Skaber en liste over de koordinater, der ikke er en slange på
        free_coordinates = [(int(x), int(y)) for x in range(int(self.window_x/10)) for y in range(int(self.window_y/10)) if grid[y][x] == 1]
        #Hvis der er ledige koordinater, vælger vi et tilfældigt ledigt koordinat 
        # Til at spawne æblet. Hvis ikke, har vi vundet
        if free_coordinates: 
            new_apple = random.choice(free_coordinates)
            new_apple = new_apple[0]*10, new_apple[1]*10
            return new_apple
        else: 
            return "WINNER"

    def reset(self):
        #Sætter tilfældig startkoordinat og æblekoordinat
        snake_length = self.snake_length
        start_pos = [random.randrange(snake_length, (self.window_x//10)-10) * 10, random.randrange(1, (self.window_y//10)) * 10]
        snake_body_local = [[start_pos[0] - 10*i,start_pos[1]] for i in range(snake_length)]
        start_fruit = self.spawn_apple(snake_body_local)
        #Gør score til 0 og angiver at der er et æble på brættet
        fruit_spawn_local = True
        score_local = 0
        #Spawner slangen med at den går mod højre og skaber slangens krops koord
        # Ud fra den tilfældige startposition.
        direction_local = 'RIGHT'
        change_to_local = direction_local
        time_steps = 0
        self.stuck, self.stuck_step = False, 0
        (self.snake_position, self.fruit_position,self.fruit_spawn,
            self.score,self.direction,self.change_to, 
            self.snake_body, self.time_steps) = (start_pos, start_fruit,
                                                    fruit_spawn_local, score_local, 
                                                    direction_local, change_to_local, 
                                                    snake_body_local, time_steps)
    
    def update_danger(self,spos,wx,wy,body):

        # danger = [0,0,0,0]
        # if spos[0] == 0: danger[3] = 1
        # if spos[0] == wx - 10: danger[2] = 1
        # if spos[1] == 0: danger[0] = 1
        # if spos[1] == wy - 10: danger[1] = 1

        # if [spos[0] + 10, spos[1]] in body: danger[2] = 1
        # if [spos[0] - 10, spos[1]] in body: danger[3] = 1
        # if [spos[0], spos[1] + 10] in body: danger[1] = 1
        # if [spos[0], spos[1] - 10] in body: danger[0] = 1

        ### Relative direction
        # The binary directions is defined as [straight, left, right]
        danger = [0,0,0]
        if self.direction == "UP":
            if spos[1] == 0 or [spos[0], spos[1] - 10] in body: danger[0] = 1
            if spos[0] == 0 or [spos[0] - 10, spos[1]] in body: danger[1] = 1
            if spos[0] == wx - 10 or [spos[0] + 10, spos[1]] in body: danger[2] = 1


        if self.direction == "DOWN":
            if spos[1] == wy - 10 or [spos[0], spos[1] + 10] in body: danger[0] = 1 
            if spos[0] == 0 or [spos[0] - 10, spos[1]] in body: danger[2] = 1
            if spos[0] == wx - 10 or [spos[0] + 10, spos[1]] in body: danger[1] = 1
        
        if self.direction == "RIGHT":
            if spos[0] == wx - 10 or [spos[0] + 10, spos[1]] in body: danger[0] = 1
            if spos[1] == 0 or [spos[0], spos[1] - 10] in body: danger[1] = 1
            if spos[1] == wy - 10 or [spos[0], spos[1] + 10] in body: danger[2] = 1

        if self.direction == "LEFT":
            if spos[0] == 0 or [spos[0] - 10, spos[1]] in body: danger[0] = 1
            if spos[1] == 0 or [spos[0], spos[1] - 10] in body: danger[2] = 1
            if spos[1] == wy - 10 or [spos[0], spos[1] + 10] in body: danger[1] = 1
        
        return danger

    # Observe apple in the four possible next states
    def update_fruit(self, spos,fpos):
        fruit = [0,0,0,0]
        if spos[0] < fpos[0]: fruit[2] = 1
        if spos[0] > fpos[0]: fruit[3] = 1
        if spos[1] < fpos[1]: fruit[1] = 1
        if spos[1] > fpos[1]: fruit[0] = 1
        return fruit
    
    def get_state(self):
        # Updating states
        danger_state = self.update_danger(self.snake_position, self.window_x,self.window_y, self.snake_body)
        direction_state = self.update_direction(self.direction)
        fruit_state = self.update_fruit(self.snake_position, self.fruit_position)

        # Concatenating states and converting into array
        # state_array = np.array([danger_state, direction_state, fruit_state])

        return np.concatenate((danger_state,direction_state, fruit_state), dtype=int)

        # # Old code
        # return (self.update_danger(self.snake_position, self.window_x,self.window_y,
        #                            self.snake_body), self.update_direction(self.direction), 
        #         self.update_fruit(self.snake_position, self.fruit_position))
    
    
    def update_direction(self, input_direction):
        output_direction = [0,0,0,0] # up, down, right, left
        if input_direction == "UP": output_direction[0] = 1
        if input_direction == "DOWN": output_direction[1] = 1
        if input_direction == "RIGHT": output_direction[2] = 1
        if input_direction == "LEFT": output_direction[3] = 1
        return output_direction
    
    
    def move(self, action_index = None):
        self.reward = -1
        self.stuck_step += 1
        if action_index:
            direction_local = self.update_direction(self.direction)
            if direction_local[0] == 1:
                if action_index[0] == 1: action = "UP"
                elif action_index[1] == 1: action = "LEFT" 
                elif action_index[2] == 1: action = "RIGHT"

        
        # Snake direction syd
            if direction_local[1] == 1:
                if action_index[0] == 1: action = "DOWN"
                elif action_index[1] == 1: action = "RIGHT" 
                elif action_index[2] == 1: action = "LEFT"

            # Snake direction øst
            if direction_local[2] == 1:
                if action_index[0] == 1: action = "RIGHT"
                elif action_index[1] == 1: action = "UP" 
                elif action_index[2] == 1: action = "DOWN"		

            # Snake direction VEST
            if direction_local[3] == 1:
                if action_index[0] == 1: action = "LEFT"
                elif action_index[1] == 1: action = "DOWN" 
                elif action_index[2] == 1: action = "UP"	

            self.change_to = action
        else:
            for event in pygame.event.get():		
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.change_to = 'UP'
                    if event.key == pygame.K_DOWN:
                        self.change_to = 'DOWN'
                    if event.key == pygame.K_LEFT:
                        self.change_to = 'LEFT'
                    if event.key == pygame.K_RIGHT:
                        self.change_to = 'RIGHT'
                    if event.key == pygame.K_r:
                        self.reset()
                    if event.key == pygame.K_q:
                        exit_program = True
                if event.type == pygame.QUIT:
                    self.exit_program = True
        
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        if self.direction == 'UP':
            self.snake_position[1] -= 10
        if self.direction == 'DOWN':
            self.snake_position[1] += 10
        if self.direction == 'LEFT':
            self.snake_position[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_position[0] += 10
        self.time_steps += 1
        self.curr_action = self.direction
        self.snake_body.insert(0, list(self.snake_position))
        if self.should_render: self.render()

    def get_time_for_apple(self):
        return self.stuck_step

    def has_apple(self):
        # Snake body growing mechanism
        # if fruits and snakes collide then scores
        # will be incremented by 10
        time_taken = False
        if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
            self.score += 10
            self.reward = self.reward_apple
            self.fruit_spawn = False
            time_taken = self.get_time_for_apple()
            self.stuck_step = 0
        else:
            self.snake_body.pop()

        if not self.fruit_spawn:
            self.fruit_position = self.spawn_apple(self.snake_body)
            if self.fruit_position == "WINNER": 
                self.has_won = True
                self.reset()
        self.fruit_spawn = True
        return time_taken if time_taken else None

    def render(self):

        self.game_window.fill(self.black)
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, self.green,
							pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(self.game_window, self.red, pygame.Rect(
			self.fruit_position[0], self.fruit_position[1], 10, 10))
        #   displaying score continuously
		#self.show_score(1, env.white, 'times new roman', 20)
		# Refresh game screen
        # creating font object score_font
        score_font = pygame.font.SysFont('times new roman', 20)
        
        # create the display surface object 
        # score_surface
        score_surface = score_font.render('Score : ' + str(self.score), True, self.white)
        
        # create a rectangular object for the text
        # surface object
        score_rect = score_surface.get_rect()
        
        # displaying text
        self.game_window.blit(score_surface, score_rect)
        pygame.display.update()
		# Frame Per Second /Refresh Rate
        self.fps.tick(self.snake_speed)

    def is_game_over(self):
        if self.snake_position[0] < 0 or self.snake_position[0] > self.window_x-10:
            #run_data.append(log_data(has_won,env.score,env.time_steps,env.snake_position))
            self.reset()
            self.game_count +=1
            self.reward = self.punish_death
            return True
        if self.snake_position[1] < 0 or self.snake_position[1] > self.window_y-10:
            self.reset()
            self.game_count +=1
            self.reward = self.punish_death
            return True

        # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                self.reset()
                self.game_count +=1
                self.reward = self.punish_death
                return True
            
        # Tjekker om slangen sidder fast og slutter hvis den gør.
        if self.kill_stuck and self.is_stuck():
            self.reset()
            self.game_count +=1
            self.reward = self.punish_death
            return True
        return False
    
    def game_over(self):
        if self.snake_position[0] < 0 or self.snake_position[0] > self.window_x-10:
            #run_data.append(log_data(has_won,env.score,env.time_steps,env.snake_position))
            # self.reset()
            # self.game_count +=1
            self.reward = self.punish_death
            return True
        if self.snake_position[1] < 0 or self.snake_position[1] > self.window_y-10:
            # self.reset()
            # self.game_count +=1
            self.reward = self.punish_death
            return True

        # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                # self.reset()
                # self.game_count +=1
                self.reward = self.punish_death
                return True
            
        # Tjekker om slangen sidder fast og slutter hvis den gør.
        if self.kill_stuck and self.is_stuck():
            # self.reset()
            # self.game_count +=1
            self.reward = self.punish_death
            return True
        return False           
    

    def get_reward(self):
        return self.reward
    def get_game_count(self):
        return self.game_count
    def get_move(self):
        return self.direction
    def write_data(self):
        return self.force_write_data if self.force_write_data else self.should_write_data
    
        # Tjekker, om slangen sidder fast. Jo længere den er, jo længere tid har den til at finde 
        # Et nyt æble. Hvis den sidder fast, afsluttes spillet oppe i is_game_over
    def is_stuck(self):
        if self.stuck_step > len(self.snake_body)*100: return True
        else: return False

        # Gør det samme som is_game_over, men ændrer ikke på spillet. Kan bruges til at finde ud af,
        # om spillet er ovre uden at genstarte og resette værdier.
    def is_game_over_bool(self):
        if self.snake_position[0] < 0 or self.snake_position[0] > self.window_x-10:
            return True
        if self.snake_position[1] < 0 or self.snake_position[1] > self.window_y-10:
            return True
        if self.kill_stuck and self.is_stuck():
            return True
        for block in self.snake_body[1:]:
            if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                return True
        return False
    
    def grid(self):
        # Konvertere spillets koordinater til "rigtige koordinater"
        x = self.window_x // 10
        y = self.window_y // 10
        body = np.array(self.snake_body) // 10
        fruit = np.array(self.fruit_position) // 10
        
        # Lav grid fra canvas
        grid = np.zeros((y, x), dtype=int)

        # Bug fix da den nogle giver værdier højere end canvas størrelse. np.clip(input, min, max) definere et interval
        body[:, 0] = np.clip(body[:, 0], 0, x - 1)
        body[:, 1] = np.clip(body[:, 1], 0, y - 1)
        
        # Ændre body og fruit placering i grid til 1
        grid[body[:, 1], body[:, 0]] = 1
        fruit = np.clip(fruit, 0, [x - 1, y - 1])  # Ensure fruit coordinates are within bounds
        grid[fruit[1], fruit[0]] = 1
        
        # Konvertere grid til 2D array
        grid2D = grid.ravel()
        
        return grid

class Data():
    def __init__(self):
        self.data = []

    def commit(self, data_other):
        self.data.append(data_other)

    def push(self, should_write, game):
        if should_write:
            try:
                with open("src\ERB.txt", "w") as f:
                    for pair in self.data:
                        f.write(f"{str(pair)}\n")
                    print("Data logged.")
            except FileNotFoundError:
                if game.force_write_data:
                    try:
                        with open("ERB.txt", "w") as f:
                            for pair in self.data:
                                f.write(f"{str(pair)}\n")
                            print("Data logged.")
                        return
                    except FileNotFoundError:
                        with open("ERB.txt", "x") as f:
                            for pair in self.data:
                                f.write(f"{str(pair)}\n")
                            print("ERB oprettet, data logged.")
                        return
                else:
                    print("Datafil er ikke fundet. Angiv path til datafil herunder. Skriv \'nej\' for at annullere.\n"
                      "skriv \'opret\' for at oprette \'ERB\':\n")
                    choice = input("path: ").lower()
                if choice == "nej":
                    pass
                elif choice == "opret":
                    with open("ERB.txt", "x") as f:
                        for pair in self.data:
                            f.write(f"{str(pair)}\n")
                        print("ERB oprettet, data logged.")
                else:
                    self.write_to_file(should_write, game)

if __name__ == "__main__":
    game = Snake_Game(snake_length = 40)
    n = 1
    buffer = Data()
    Transition = namedtuple("Transition",
                            ("state","action","reward","next_state"))
    
    while game.get_game_count() < n:
        s1 = game.get_state()
        game.move()
        action = game.get_move()
        game.has_apple()
        game_over = game.is_game_over()
        reward = game.get_reward()
        s2 = game.get_state()
        buffer.commit(Transition(s1,action,reward,s2 if not game_over else None))
    buffer.push(game.write_data(), game)



