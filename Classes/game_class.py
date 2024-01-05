import pygame
import random

class Snake_Game():
    def __init__(self, render=True, write_data = False, apple_reward = 50, step_punish = -1, death_punish = -100, 
                 window_x = 720, window_y = 480, snake_speed = 15, snake_length = 4):
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

        # Initialising pygame
        pygame.init()
        self.should_write_data = write_data
        self.should_render = render
        self.game_count = 0
        # Initialise game window
        if self.should_render:
            pygame.display.set_caption('GeeksforGeeks Snakes')
            self.game_window = pygame.display.set_mode((self.window_x, self.window_y))

        # FPS (frames per second) controller
        self.fps = pygame.time.Clock()

        self.reset()
        self.curr_action = self.direction
        self.reward_apple, self.punish_no_apple, self.punish_death = apple_reward, step_punish, death_punish
        self.reward = 0

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
        # Ud fra den tilfældige startposition. Returnerer en tuple
        direction_local = 'RIGHT'
        change_to_local = direction_local
        time_steps = 0
        (self.snake_position, self.fruit_position,self.fruit_spawn,
            self.score,self.direction,self.change_to, 
            self.snake_body, self.time_steps) = (start_pos, start_fruit,
                                                    fruit_spawn_local, score_local, 
                                                    direction_local, change_to_local, 
                                                    snake_body_local, time_steps)
    
    def update_danger(self,spos,wx,wy,body):
        danger = [0,0,0,0]
        if spos[0] == 0: danger[3] = 1
        if spos[0] == wx - 10: danger[2] = 1
        if spos[1] == 0: danger[0] = 1
        if spos[1] == wy - 10: danger[1] = 1

        if [spos[0] + 10, spos[1]] in body: danger[2] = 1
        if [spos[0] - 10, spos[1]] in body: danger[3] = 1
        if [spos[0], spos[1] + 10] in body: danger[1] = 1
        if [spos[0], spos[1] - 10] in body: danger[0] = 1
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
        return (self.update_danger(self.snake_position, self.window_x,self.window_y,self.snake_body), 
                self.update_fruit(self.snake_position, self.fruit_position))
    
    def move(self, action = None):
        self.reward = -1
        if action:
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

    def has_apple(self):
        # Snake body growing mechanism
        # if fruits and snakes collide then scores
        # will be incremented by 10
        if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
            self.score += 10
            self.reward = self.reward_apple
            self.fruit_spawn = False
        else:
            self.snake_body.pop()

        if not self.fruit_spawn:
            self.fruit_position = self.spawn_apple(self.snake_body)
            if self.fruit_position == "WINNER": 
                self.has_won = True
                self.reset()
        self.fruit_spawn = True

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
        if self.snake_position[1] < 0 or self.snake_position[1] > self.window_y-10:
            self.reset()
            self.game_count +=1
            self.reward = self.punish_death

        # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                self.reset()
                self.game_count +=1
                self.reward = self.punish_death

    def get_reward(self):
        return self.reward
    def get_game_count(self):
        return self.game_count
    def get_move(self):
        return self.direction
    def write_data(self):
        return self.should_write_data
    


class Data():
    def __init__(self):
        self.data = []

    def __add__(self, data_other):
        self.data.append(data_other)

    def write_to_file(self, should_write):
        if should_write:
            with open("src\ERB.txt", "w") as f:
                for pair in self.data:
                    f.write(f"{str(pair)}\n")

if __name__ == "__main__":
    game = Snake_Game()
    n = 2
    buffer = Data()
    while game.get_game_count() < n:
        if not game.fruit_spawn:
            game.spawn_apple()
        s1 = game.get_state()
        game.move()
        game.has_apple()
        game.is_game_over()
        action = game.get_move()
        reward = game.get_reward()
        s2 = game.get_state()
        buffer + (s1,action,reward,s2)
    buffer.write_to_file(game.write_data())



