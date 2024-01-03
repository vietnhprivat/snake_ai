#AI VERSION
import snake_ai_test as snake_ai_test
import pygame

env = snake_ai_test
exit_program = False
render = True

while not exit_program:
	# handling key events
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
				env.snake_position, env.fruit_position,env.fruit_spawn,env.score,env.direction,env.change_to, env.snake_body = env.reset() ########
	if event.type == pygame.QUIT:
		exit_program = True
		#env.game_over()
		pygame.quit()
		env.close()
		break

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

	# Snake body growing mechanism
	# if fruits and snakes collide then scores
	# will be incremented by 10
	env.snake_body.insert(0, list(env.snake_position))
	if env.snake_position[0] == env.fruit_position[0] and env.snake_position[1] == env.fruit_position[1]:
		env.score += 10
		env.fruit_spawn = False
	else:
		env.snake_body.pop()
	
	
	if not env.fruit_spawn:
		env.fruit_position = [env.random.randrange(1, (env.window_x//10)) * 10, 
						env.random.randrange(1, (env.window_y//10)) * 10]
		
		
	env.fruit_spawn = True
	env.game_window.fill(env.black)
	
	for pos in env.snake_body:
		pygame.draw.rect(env.game_window, env.green,
						pygame.Rect(pos[0], pos[1], 10, 10))
	pygame.draw.rect(env.game_window, env.red, pygame.Rect(
		env.fruit_position[0], env.fruit_position[1], 10, 10))

	# Game Over conditions
	if env.snake_position[0] < 0 or env.snake_position[0] > env.window_x-10:
		env.game_over()
		break
	if env.snake_position[1] < 0 or env.snake_position[1] > env.window_y-10:
		env.game_over()
		break

	# Touching the snake body
	for block in env.snake_body[1:]:
		if env.snake_position[0] == block[0] and env.snake_position[1] == block[1]:
			env.game_over()
			break
	if render:
		#   displaying score continuously
		env.show_score(1, env.white, 'times new roman', 20)
		# Refresh game screen
		pygame.display.update()
		# Frame Per Second /Refresh Rate
		env.fps.tick(env.snake_speed)
