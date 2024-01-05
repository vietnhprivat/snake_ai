#AI VERSION
import tab_q_snake_ai_test as env
import pygame
from collections import defaultdict 
import numpy as np
import pickle



render = True
write_data = False
training = False
number_of_runs = 10000
punish_no_apple = -1
punish_death = -100
reward_apple = 50
count = 0
exit_program, has_won, run_nr,run_data, to_buffer = False, False,0,[],[]

def log_data(won, score, time, pos):
	return won, score, time, pos

def update_danger(spos,wx,wy,body):
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

def update_direction(input_direction):
	output_direction = [0,0,0,0] # up, down, right, left
	if input_direction == "UP": output_direction[0] = 1
	if input_direction == "DOWN": output_direction[1] = 1
	if input_direction == "RIGHT": output_direction[2] = 1
	if input_direction == "LEFT": output_direction[3] = 1
	return output_direction

# Q-learning table initialize
Q = defaultdict(lambda: [0., 0., 0.])

if training == False:
	file_path = 'tabular-q-learning\q-table.pkl'
	with open(file_path, 'rb') as file:
		loaded_Q = pickle.load(file)

	Q.update(loaded_Q)



def update_fruit(spos,fpos):
	fruit = [0,0,0,0]
	if spos[0] < fpos[0]: fruit[2] = 1
	if spos[0] > fpos[0]: fruit[3] = 1
	if spos[1] < fpos[1]: fruit[1] = 1
	if spos[1] > fpos[1]: fruit[0] = 1
	return fruit

while not exit_program:
	reward = punish_no_apple
	# Udregner danger tuple
	danger = update_danger(env.snake_position,env.window_x,env.window_y,env.snake_body)
	direction = update_direction(env.direction)
	fruit = update_fruit(env.snake_position,env.fruit_position)

	
	s1 = danger, direction, fruit
	
	# Q-table step 1. current stage
	
	qcurrent = Q[(tuple(danger), tuple(direction), tuple(fruit))]
	action_index = np.argmax(qcurrent)
	


	# Snake direction Nord
	if direction[0] == 1:
		if action_index == 0: action = "UP"
		if action_index == 1: action = "LEFT" 
		if action_index == 2: action = "RIGHT"

	
	# Snake direction syd
	if direction[1] == 1:
		if action_index == 0: action = "DOWN"
		if action_index == 1: action = "RIGHT" 
		if action_index == 2: action = "LEFT"

	# Snake direction øst
	if direction[2] == 1:
		if action_index == 0: action = "RIGHT"
		if action_index == 1: action = "UP" 
		if action_index == 2: action = "DOWN"		

	# Snake direction VEST
	if direction[3] == 1:
		if action_index == 0: action = "LEFT"
		if action_index == 1: action = "DOWN" 
		if action_index == 2: action = "UP"	

	env.change_to = action




	# Handling 
	# key events
	# Hvad der sker når man trykker knapper. Kan erstates med modellens valg
	for event in pygame.event.get():		
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_UP:
				env.change_to = 'UP'
			if event.key == pygame.K_DOWN:
				env.change_to = 'DOWN'
			if event.key == pygame.K_LEFT:
				env.change_to = 'LEFT'
			if event.key == pygame.K_RIGHT:
				env.change_to = 'RIGHT'
			if event.key == pygame.K_r:
				(env.snake_position, env.fruit_position,env.fruit_spawn,
	 			env.score,env.direction,env.change_to, env.snake_body, 
				env.time_steps) = env.reset()
			if event.key == pygame.K_q:
				exit_program = True
	if event.type == pygame.QUIT:
		exit_program = True

	# If two keys pressed simultaneously
	# we don't want snake to move into two 
	# directions simultaneously
	if env.change_to == 'UP' and env.direction != 'DOWN':
		env.direction = 'UP'
	if env.change_to == 'DOWN' and env.direction != 'UP':
		env.direction = 'DOWN'
	if env.change_to == 'LEFT' and env.direction != 'RIGHT':
		env.direction = 'LEFT'
	if env.change_to == 'RIGHT' and env.direction != 'LEFT':
		env.direction = 'RIGHT'

	# Moving the snake
	if env.direction == 'UP':
		env.snake_position[1] -= 10
	if env.direction == 'DOWN':
		env.snake_position[1] += 10
	if env.direction == 'LEFT':
		env.snake_position[0] -= 10
	if env.direction == 'RIGHT':
		env.snake_position[0] += 10
	env.time_steps += 1

	curr_action = env.direction
	# Snake body growing mechanism
	# if fruits and snakes collide then scores
	# will be incremented by 10
	env.snake_body.insert(0, list(env.snake_position))
	if env.snake_position[0] == env.fruit_position[0] and env.snake_position[1] == env.fruit_position[1]:
		env.score += 10
		reward = reward_apple
		env.fruit_spawn = False
	else:
		env.snake_body.pop()
	
	# Hvis der ikke er et æble på brættet, skal et nyt spawnes, 
	# se environment for den funktion
	if not env.fruit_spawn:
		env.fruit_position = env.spawn_apple(env.snake_body)
		if env.fruit_position == "WINNER": 
			has_won = True
			(env.snake_position, env.fruit_position,env.fruit_spawn,
	 			env.score,env.direction,env.change_to, env.snake_body, 
				env.time_steps) = env.reset()
			#Indtil videre, vi skal have implementeret en vinderfunktion

	env.fruit_spawn = True
	
	#tegner slangen og æblet
	if render:
		env.game_window.fill(env.black)
		for pos in env.snake_body:
			pygame.draw.rect(env.game_window, env.green,
							pygame.Rect(pos[0], pos[1], 10, 10))
		pygame.draw.rect(env.game_window, env.red, pygame.Rect(
			env.fruit_position[0], env.fruit_position[1], 10, 10))

	# Game Over conditions
	if env.snake_position[0] < 0 or env.snake_position[0] > env.window_x-10:
		run_data.append(log_data(has_won,env.score,env.time_steps,env.snake_position))
		(env.snake_position, env.fruit_position,env.fruit_spawn,
	 			env.score,env.direction,env.change_to, env.snake_body, 
				env.time_steps) = env.reset()
		run_nr += 1
		reward = punish_death
	if env.snake_position[1] < 0 or env.snake_position[1] > env.window_y-10:
		run_data.append(log_data(has_won,env.score,env.time_steps,env.snake_position))
		(env.snake_position, env.fruit_position,env.fruit_spawn,
	 			env.score,env.direction,env.change_to, env.snake_body, 
				env.time_steps) = env.reset()
		run_nr += 1
		reward = punish_death

	# Touching the snake body
	for block in env.snake_body[1:]:
		if env.snake_position[0] == block[0] and env.snake_position[1] == block[1]:
			run_data.append(log_data(has_won,env.score,env.time_steps,env.snake_position))
			(env.snake_position, env.fruit_position,env.fruit_spawn,
	 			env.score,env.direction,env.change_to, env.snake_body, 
				env.time_steps) = env.reset()
			run_nr += 1
			reward = punish_death

	# Udregner danger tuple
	danger = update_danger(env.snake_position,env.window_x,env.window_y,env.snake_body)
	fruit = update_fruit(env.snake_position,env.fruit_position)
	direction = update_direction(env.direction)
	s2 = danger, direction, fruit

	qnew = Q[(tuple(danger), tuple(direction), tuple(fruit))]
	gamma = 0.9
	qcurrent[action_index] = reward + gamma * np.max(qnew)

	if render:
		#   displaying score continuously
		env.show_score(1, env.white, 'times new roman', 20)
		# Refresh game screen
		pygame.display.update()
		# Frame Per Second /Refresh Rate
		env.fps.tick(env.snake_speed)
	if run_nr == number_of_runs:
		exit_program = True
		env.close()
		print(run_data)
	to_buffer.append((s1,curr_action,reward,s2))

	## round timer 
	# if run_nr%10 == 0:
	# 	print(run_nr)




file_path = 'tabular-q-learning\q-table.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(dict(Q), file)


if write_data:
	with open("src\ERB.txt", "w") as f:
		for pair in to_buffer:
			f.write(f"{str(pair)}\n")
