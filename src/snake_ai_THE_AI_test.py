#AI VERSION
import snake_ai_test as env
import pygame

render = False
write_data = False
number_of_runs = 2
punish_no_apple = -1
punish_death = -100
reward_apple = 50

exit_program, has_won, run_nr,run_data, to_buffer = False, False,0,[],[]
def log_data(won, score, time, pos):
	return won, score, time, pos


while not exit_program:
	reward = punish_no_apple
	# Udregner danger tuple
	danger = [0,0,0,0] #Nord,syd,øst,vest
	if env.snake_position[0] == 0: danger[3] = 1
	if env.snake_position[0] == env.window_x - 10: danger[2] = 1
	if env.snake_position[1] == 0: danger[0] = 1
	if env.snake_position[1] == env.window_y - 10: danger[1] = 1

	if [env.snake_position[0] + 10, env.snake_position[1]] in env.snake_body: danger[2] = 1
	if [env.snake_position[0] - 10, env.snake_position[1]] in env.snake_body: danger[3] = 1
	if [env.snake_position[0], env.snake_position[1] + 10] in env.snake_body: danger[1] = 1
	if [env.snake_position[0], env.snake_position[1] - 10] in env.snake_body: danger[0] = 1

	fruit = [0,0,0,0] #Nord, syd, øst, vest
	if env.snake_position[0] < env.fruit_position[0]: fruit[2] = 1
	if env.snake_position[0] > env.fruit_position[0]: fruit[3] = 1
	if env.snake_position[1] < env.fruit_position[1]: fruit[1] = 1
	if env.snake_position[1] > env.fruit_position[1]: fruit[0] = 1
	

	s1 = danger, fruit
	# Handling key events
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
	danger = [0,0,0,0] #Nord,syd,øst,vest
	if env.snake_position[0] == 0: danger[3] = 1
	if env.snake_position[0] == env.window_x - 10: danger[2] = 1
	if env.snake_position[1] == 0: danger[0] = 1
	if env.snake_position[1] == env.window_y - 10: danger[1] = 1

	if [env.snake_position[0] + 10, env.snake_position[1]] in env.snake_body: danger[2] = 1
	if [env.snake_position[0] - 10, env.snake_position[1]] in env.snake_body: danger[3] = 1
	if [env.snake_position[0], env.snake_position[1] + 10] in env.snake_body: danger[1] = 1
	if [env.snake_position[0], env.snake_position[1] - 10] in env.snake_body: danger[0] = 1

	fruit = [0,0,0,0] #Nord, syd, øst, vest
	if env.snake_position[0] < env.fruit_position[0]: fruit[2] = 1
	if env.snake_position[0] > env.fruit_position[0]: fruit[3] = 1
	if env.snake_position[1] < env.fruit_position[1]: fruit[1] = 1
	if env.snake_position[1] > env.fruit_position[1]: fruit[0] = 1

	s2 = danger, fruit
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
	to_buffer.append((s1,env.direction,reward,s2))

if write_data:
	with open("src\ERB.txt", "w") as f:
		for pair in to_buffer:
			f.write(f"{str(pair)}\n")
