# importing libraries
import pygame
import time
import random

snake_speed = 15

# Window size
window_x = 720
window_y = 480

# defining colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

# Initialising pygame
pygame.init()

# Initialise game window
pygame.display.set_caption('GeeksforGeeks Snakes')
game_window = pygame.display.set_mode((window_x, window_y))

# FPS (frames per second) controller
fps = pygame.time.Clock()

def spawn_apple(snake_coordinates):
	grid = [[1 for _ in range(int(window_x/10))] for _ in range(int(window_y/10))]
	for x,y in snake_coordinates: grid[int(y/10)][int(x/10)] = 0
	free_coordinates = [(int(x), int(y)) for x in range(int(window_x/10)) for y in range(int(window_y/10)) if grid[y][x] == 1]
	if free_coordinates: 
		new_apple = random.choice(free_coordinates)
		new_apple = new_apple[0]*10, new_apple[1]*10
		return new_apple
	else: 
		return "WINNER"




# displaying Score function
def show_score(choice, color, font, size):

	# creating font object score_font
	score_font = pygame.font.SysFont(font, size)
	
	# create the display surface object 
	# score_surface
	score_surface = score_font.render('Score : ' + str(score), True, color)
	
	# create a rectangular object for the text
	# surface object
	score_rect = score_surface.get_rect()
	
	# displaying text
	game_window.blit(score_surface, score_rect)

# game over function
def game_over():

	# creating font object my_font
	my_font = pygame.font.SysFont('times new roman', 50)
	
	# creating a text surface on which text 
	# will be drawn
	game_over_surface = my_font.render(
		'Your Score is : ' + str(score), True, red)
	
	# create a rectangular object for the text 
	# surface object
	game_over_rect = game_over_surface.get_rect()
	
	# setting position of the text
	game_over_rect.midtop = (window_x/2, window_y/4)
	
	# blit will draw the text on screen
	game_window.blit(game_over_surface, game_over_rect)
	pygame.display.flip()
	
	# after 2 seconds we will quit the program
	#time.sleep(2)
	
	# deactivating pygame library
	pygame.quit()
	
	# quit the program
	quit()
def close():
	pygame.quit()
def reset():
	start_pos = [random.randrange(4, (window_x//10)-10) * 10, random.randrange(1, (window_y//10)) * 10]
	start_fruit = [random.randrange(1, (window_x//10)) * 10, random.randrange(1, (window_y//10)) * 10]
	fruit_spawn_local = True
	score_local = 0
	direction_local = 'RIGHT'
	change_to_local = direction_local
	snake_body_local = [[start_pos[0],start_pos[1]],
			[start_pos[0]-10, start_pos[1]],
			[start_pos[0]-20, start_pos[1]],
			[start_pos[0]-30, start_pos[1]]
			]
	steps = 0
	return start_pos, start_fruit, fruit_spawn_local, score_local, direction_local, change_to_local, snake_body_local, steps
	
snake_position, fruit_position, fruit_spawn, score, direction, change_to, snake_body, steps = reset()