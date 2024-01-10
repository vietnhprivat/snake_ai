import random
from collections import namedtuple
import torch
import importlib
pygame = None
import numpy as np
import math


class Snake_Game():
    def __init__(self, render=True, write_data = False, apple_reward = 50, step_punish = -1, death_punish = -100, 
                 window_x = 720, window_y = 480, snake_speed = 15, snake_length = 4, force_write_data = False, kill_stuck = True,
                 grid_state=False, reward_closer=0, backstep = False):
        self.backstep = backstep
        self.snake_speed = snake_speed
        self.grid_mode = grid_state

        # Window size
        self.window_x = window_x
        self.window_y = window_y
        self.direction_space = ["UP", "DOWN", "RIGHT", "LEFT"]

        self.device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.snake_length = snake_length
        self.kill_stuck = kill_stuck

        # Initialising pygame
        self.should_write_data = force_write_data if force_write_data else write_data
        self.force_write_data = force_write_data
        self.should_render = render
        self.game_count = 0
        # Initialise game window
        if self.should_render:
            self.pygame = importlib.import_module('pygame')
            self.pygame.init()
            self.black = self.pygame.Color(0, 0, 0)
            self.white = self.pygame.Color(255, 255, 255)
            self.red = self.pygame.Color(255, 0, 0)
            self.green = self.pygame.Color(0, 255, 0)
            self.blue = self.pygame.Color(0, 0, 255)
            self.pygame.display.set_caption('Slitherin Pythons')
            self.game_window = self.pygame.display.set_mode((self.window_x, self.window_y))
            self.toggle_possible = True
            # defining colors

        # FPS (frames per second) controller
            self.fps = self.pygame.time.Clock()
        else: self.pygame, self.toggle_possible = None, False

        self.reset()
        self.curr_action = self.direction
        self.reward_apple, self.punish_no_apple, self.punish_death, self.reward_closer= apple_reward, step_punish, death_punish, reward_closer
        self.reward = 0
        self.action_space = [[1,0,0], [0,1,0], [0,0,1]] if not self.grid_mode else [[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1]]
        self.stuck_step = 0
        self.stuck = False

        #Tjekker i action om vi bliver feedet med træk
        self.getting_moves = False

        #Æble-spawn funktion
    def spawn_apple(self, snake_coordinates):
        #Laver en liste med lister, hvor hvert element repræsenterer et koordinat:
        #Her får alle koordinater værdien 1
        apple_grid = np.ones((int(self.window_x/10), int(self.window_y/10)))

        #Tager koordinaterne fra slangens krop og giver disse koordinater værdien 0
        snake_positions = np.array(snake_coordinates) // 10
        apple_grid[snake_positions[:, 0],snake_positions[:, 1]] = 0
        #Skaber en liste over de koordinater, der ikke er en slange på
        free_coordinates = np.argwhere(apple_grid == 1)
        #Hvis der er ledige koordinater, vælger vi et tilfældigt ledigt koordinat 
        if len(free_coordinates) > 0:
             chosen_coordinate = free_coordinates[random.randint(0, len(free_coordinates) - 1)] * 10
             return chosen_coordinate[0], chosen_coordinate[1]
        else: 
            return "WINNER"

    def reset(self):
        self.direction = self.direction_space[random.randint(0,3)]
        self.change_to = self.direction
        self.snake_position = [random.randrange(self.snake_length*2, (self.window_x//10)-self.snake_length*2) * 10, random.randrange(self.snake_length*2, (self.window_y//10)-self.snake_length*2) * 10]
        if self.direction == "RIGHT":
            self.snake_body = [[self.snake_position[0] - 10*i,self.snake_position[1]] for i in range(self.snake_length)]
        elif self.direction == "LEFT":
            self.snake_body = [[self.snake_position[0] + 10*i,self.snake_position[1]] for i in range(self.snake_length)]
        elif self.direction == "UP":
            self.snake_body = [[self.snake_position[0],self.snake_position[1] + 10*i] for i in range(self.snake_length)]
        elif self.direction == "DOWN":
            self.snake_body = [[self.snake_position[0],self.snake_position[1] - 10*i] for i in range(self.snake_length)]
        self.fruit_position = self.spawn_apple(self.snake_body)
        self.fruit_spawn = True
        self.score = 0
        self.time_steps = 0
        self.stuck, self.stuck_step = False, 0
    
    def update_danger(self,spos,wx,wy,body):
        danger = [0,0,0]
        if self.direction == "UP":
            if spos[1] == 0 or [spos[0], spos[1] - 10] in body: danger[0] = 1
            if spos[0] == 0 or [spos[0] - 10, spos[1]] in body: danger[1] = 1
            if spos[0] == wx - 10 or [spos[0] + 10, spos[1]] in body: danger[2] = 1

        elif self.direction == "DOWN":
            if spos[1] == wy - 10 or [spos[0], spos[1] + 10] in body: danger[0] = 1 
            if spos[0] == 0 or [spos[0] - 10, spos[1]] in body: danger[2] = 1
            if spos[0] == wx - 10 or [spos[0] + 10, spos[1]] in body: danger[1] = 1

        elif self.direction == "RIGHT":
            if spos[0] == wx - 10 or [spos[0] + 10, spos[1]] in body: danger[0] = 1
            if spos[1] == 0 or [spos[0], spos[1] - 10] in body: danger[1] = 1
            if spos[1] == wy - 10 or [spos[0], spos[1] + 10] in body: danger[2] = 1

        else:
            if spos[0] == 0 or [spos[0] - 10, spos[1]] in body: danger[0] = 1
            if spos[1] == 0 or [spos[0], spos[1] - 10] in body: danger[2] = 1
            if spos[1] == wy - 10 or [spos[0], spos[1] + 10] in body: danger[1] = 1
        
        return danger

    # Observe apple in the four possible next states
    def update_fruit(self, spos,fpos):
        fruit = [0,0,0,0]
        if spos[0] < fpos[0]: fruit[2] = 1
        elif spos[0] > fpos[0]: fruit[3] = 1
        if spos[1] < fpos[1]: fruit[1] = 1
        elif spos[1] > fpos[1]: fruit[0] = 1
        return fruit
    
    def get_state(self, is_tensor=False, game_over=False):
        if game_over: return None
        if self.grid_mode:
            if is_tensor: return torch.tensor((self.grid()),dtype=int).to(self.device)
            return self.grid()
        danger_state = self.update_danger(self.snake_position, self.window_x,self.window_y, self.snake_body)
        direction_state = self.update_direction(self.direction)
        fruit_state = self.update_fruit(self.snake_position, self.fruit_position)

        # Concatenating states and converting into array
        # state_array = np.array([danger_state, direction_state, fruit_state])
        if is_tensor:
            return torch.tensor(np.concatenate((danger_state,direction_state, fruit_state), dtype=int)).to(self.device)

        return np.concatenate((danger_state,direction_state, fruit_state), dtype=int)
    
    def update_direction(self, input_direction):
        output_direction = [0,0,0,0] # up, down, right, left
        if input_direction == "UP": output_direction[0] = 1
        elif input_direction == "DOWN": output_direction[1] = 1
        elif input_direction == "RIGHT": output_direction[2] = 1
        elif input_direction == "LEFT": output_direction[3] = 1
        return output_direction
    
    
    def move(self, action_index = None):
        curr_dist = math.dist(self.snake_position, self.fruit_position)
        self.reward = self.punish_no_apple
        self.stuck_step += 1
        if action_index:
            self.getting_moves = True
            direction_local = self.update_direction(self.direction)
            if self.grid_mode:
                if action_index[0] == 1:
                    #Model vil gå nord
                    action = "UP"
                elif action_index[1] == 1:
                    action = "DOWN"
                elif action_index[2] == 1:
                    action = "RIGHT"
                elif action_index[3] == 1:
                    action = "LEFT"
            else:
                if direction_local[0] == 1:
                    if action_index[0] == 1: action = "UP"
                    elif action_index[1] == 1: action = "LEFT" 
                    elif action_index[2] == 1: action = "RIGHT"


            # Snake direction syd
                elif direction_local[1] == 1:
                    if action_index[0] == 1: action = "DOWN"
                    elif action_index[1] == 1: action = "RIGHT" 
                    elif action_index[2] == 1: action = "LEFT"

                # Snake direction øst
                elif direction_local[2] == 1:
                    if action_index[0] == 1: action = "RIGHT"
                    elif action_index[1] == 1: action = "UP" 
                    elif action_index[2] == 1: action = "DOWN"		

                # Snake direction VEST
                else:
                    if action_index[0] == 1: action = "LEFT"
                    elif action_index[1] == 1: action = "DOWN" 
                    elif action_index[2] == 1: action = "UP"	

            self.change_to = action

        elif self.pygame is not None and not self.getting_moves:
            for event in self.pygame.event.get():		
                if event.type == self.pygame.KEYDOWN:
                    if event.key == self.pygame.K_UP:
                        self.change_to = 'UP'
                    if event.key == self.pygame.K_DOWN:
                        self.change_to = 'DOWN'
                    if event.key == self.pygame.K_LEFT:
                        self.change_to = 'LEFT'
                    if event.key == self.pygame.K_RIGHT:
                        self.change_to = 'RIGHT'
                    if event.key == self.pygame.K_r:
                        self.reset()
                    if event.key == self.pygame.K_q:
                        exit_program = True
                if event.type == self.pygame.QUIT:
                    self.exit_program = True
        
        self.direction = self.change_to

        if self.direction == 'RIGHT':
            self.snake_position[0] += 10
        elif self.direction == 'UP':
            self.snake_position[1] -= 10
        elif self.direction == 'DOWN':
            self.snake_position[1] += 10
        elif self.direction == 'LEFT':
            self.snake_position[0] -= 10
        self.time_steps += 1
        new_dist = math.dist(self.snake_position, self.fruit_position)
        if new_dist < curr_dist:
            self.reward += self.reward_closer
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
    
    def toggle_rendering(self):
        self.should_render = not self.should_render
        if self.should_render:
            self.snake_speed = 25
        else: self.snake_speed = 2000

    def render(self):
        if self.pygame is None:
            self.pygame = importlib.import_module('pygame')
            self.pygame.init()
            self.black = self.pygame.Color(0, 0, 0)
            self.white = self.pygame.Color(255, 255, 255)
            self.red = self.pygame.Color(255, 0, 0)
            self.green = self.pygame.Color(0, 255, 0)
            self.blue = self.pygame.Color(0, 0, 255)
            self.pygame.display.set_caption('Slitherin Pythons')
            self.game_window = self.pygame.display.set_mode((self.window_x, self.window_y))
            # defining colors

        # FPS (frames per second) controller
            self.fps = self.pygame.time.Clock()

        self.game_window.fill(self.black)
        for pos in self.snake_body:
            self.pygame.draw.rect(self.game_window, self.green,
							self.pygame.Rect(pos[0], pos[1], 10, 10))
        self.pygame.draw.rect(self.game_window, self.red, self.pygame.Rect(
			self.fruit_position[0], self.fruit_position[1], 10, 10))
        #   displaying score continuously
		#self.show_score(1, env.white, 'times new roman', 20)
		# Refresh game screen
        # creating font object score_font
        score_font = self.pygame.font.SysFont('times new roman', 20)
        
        # create the display surface object 
        # score_surface
        score_surface = score_font.render('Score : ' + str(self.score), True, self.white)
        
        # create a rectangular object for the text
        # surface object
        score_rect = score_surface.get_rect()
        
        # displaying text
        self.game_window.blit(score_surface, score_rect)
        self.pygame.display.update()
		# Frame Per Second /Refresh Rate
        self.fps.tick(self.snake_speed)

    def is_game_over(self):
        if self.snake_position[0] < 0 or self.snake_position[0] > self.window_x-10:
            #run_data.append(log_data(has_won,env.score,env.time_steps,env.snake_position))
            self.reset()
            self.game_count +=1
            self.reward = self.punish_death
            return True
        elif self.snake_position[1] < 0 or self.snake_position[1] > self.window_y-10:
            self.reset()
            self.game_count +=1
            self.reward = self.punish_death
            return True
        # Tjekker om slangen sidder fast og slutter hvis den gør.
        elif self.kill_stuck and self.is_stuck():
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
        elif self.snake_position[1] < 0 or self.snake_position[1] > self.window_y-10:
            return True
        elif self.kill_stuck and self.is_stuck():
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
        grid[fruit[1], fruit[0]] = 5
        grid[self.snake_position[1] // 10, self.snake_position[0] // 10] = 2

        to_app = np.full((2,x), -1)

        grid = np.append(grid, to_app, 0)
        grid = np.roll(grid, 1, 0)
        to_app_y = np.full((y+2,2), -1)
        grid = np.concatenate((grid,to_app_y), 1)
        grid = np.roll(grid, 1, 1)
        
        # Konvertere grid til 2D array
        grid2D = grid.flatten()
        
        return grid2D
    
    def get_state_vector(self):
        
        # Hoved koordinater
        head_x = self.snake_position[0] // 10
        head_y = self.snake_position[1] // 10
        head_cordinates = np.array((head_x, head_y))

        # Æble koordinater
        local_fruit_position = np.array(self.fruit_position) // 10

        # Distance mellem slangen og æble
        distance = np.linalg.norm(head_cordinates - local_fruit_position)

        # Retningsvektor
        local_direction = self.update_direction(self.direction)

        # Boolean flag, om slange krop er mellem hovedet og æble

        # Mulige movement space
        available_moves = np.array((1,1,1,1))
        if self.direction == "UP": available_moves[1] = 0
        elif self.direction == "DOWN": available_moves[0] = 0
        elif self.direction == "RIGHT": available_moves[3] = 0
        elif self.direction == "LEFT": available_moves[2] = 0
    
        # Wall danger
        wall_danger = np.array((0,0,0,0))
        if self.snake_position[0] == 0: wall_danger[3] = 1
        if self.snake_position[0] == self.window_x - 10: wall_danger[2] = 1
        if self.snake_position[1] == 0: wall_danger[0] = 1
        if self.snake_position[1] == self.window_y - 10: wall_danger[1] = 1

        # Body danger
        body_danger = np.array((0,0,0,0))
        if [self.snake_position[0] + 10, self.snake_position[1]] in self.snake_body: body_danger[2] = 1
        if [self.snake_position[0] - 10, self.snake_position[1]] in self.snake_body: body_danger[3] = 1
        if [self.snake_position[0], self.snake_position[1] + 10] in self.snake_body: body_danger[1] = 1
        if [self.snake_position[0], self.snake_position[1] - 10] in self.snake_body: body_danger[0] = 1
        
        # Shortest distance from head to snake’s segments
        # print(f"head_cord: {head_cordinates}, fruit: {local_fruit_position}, distance: {distance}")
              
        # print(f"head_cord: {head_cordinates}, distance: {distance}, local_direction {local_direction}, available moves: {available_moves}, wall: {wall_danger}, body {body_danger}")
        
        state_output = np.concatenate((head_cordinates, local_fruit_position, distance, local_direction, available_moves, wall_danger, body_danger), axis = None)

        return state_output





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
    game = Snake_Game(window_x=300,window_y=300, snake_speed=15)
    n = 1
    buffer = Data()
    Transition = namedtuple("Transition",
                            ("state","action","reward","next_state"))
    print_grid = True
    while game.get_game_count() < n:
        s1 = game.get_state()
        game.get_state_vector()
        game.move()
        action = game.get_move()
        game.has_apple()
        game_over = game.is_game_over()
        reward = game.get_reward()
        s2 = game.get_state()
        buffer.commit(Transition(s1,action,reward,s2 if not game_over else None))
    buffer.push(game.write_data(), game)



