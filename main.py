
# from SnakeEngine import SnakeEngine as SE
#
# snake = SE(10)
# snake.run_game_in_real_time()

from QLearning import QLearning

RL = QLearning(100,
               debug_on=False,
               visuals_on_while_training=True,
               load_on=True,
               save_on=False,
               file_name="better-q-table.json",
               show_score_plot=False,
               training_on=True)

RL.run_optimal_game(n_times=10)
