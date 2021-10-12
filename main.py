
# from SnakeEngine import SnakeEngine as SE
#
# snake = SE(10)
# snake.run_game_in_real_time()

from QLearning import QLearning

RL = QLearning(1000000, debug_on=False, visuals_on=False, load_on=True, file_name="q-table.json")
RL.learning_loop()
RL.run_optimal_game()
