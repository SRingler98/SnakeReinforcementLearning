
# from SnakeEngine import SnakeEngine as SE
#
# snake = SE(10)
# snake.run_game_in_real_time()

from QLearning import QLearning

RL = QLearning(10000,
               debug_on=False,
               visuals_on_while_training=False,
               load_on=True,
               save_on=False,
               file_name="Higher_failure-q-table.json",
               show_score_plot=True,
               training_on=False)

RL.run_optimal_game(n_times=10)
