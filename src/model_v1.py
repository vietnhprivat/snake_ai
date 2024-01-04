#model_v1
import snake_ai_THE_AI_test
import time
env = snake_ai_THE_AI_test
while not env.exit_program:
    time.sleep(2)
    print(env.env.snake_position, env.env.snake_body, env.env.steps, env.env.score)