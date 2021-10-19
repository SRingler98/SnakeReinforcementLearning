
# from SnakeEngine import SnakeEngine as SE
#
# snake = SE(10)
# snake.run_game_in_real_time()

from QLearning import QLearning

RL = QLearning(10000,
               debug_on=False,
               visuals_on_while_training=False,
               load_on=False,
               save_on=False,
               file_name="high_reward_high_pain-q-table.json",
               show_score_plot=True,
               training_on=True)

# RL.run_optimal_game(n_times=10)
RL.run_optimal_game_graph(1000)
