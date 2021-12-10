
# from SnakeEngine import SnakeEngine as SE
#
# snake = SE(10)
# snake.run_game_in_real_time()

from typing import final
from matplotlib import pyplot as plt
import numpy as np
import statistics as stat
import pandas as pd
from QLearning import QLearning
from graphing import learning_loop_multiple_agents_create_graph
#from Menu import *

#menu()

RL = QLearning(100,
               debug_on=False,
               visuals_on_while_training=False,
               load_on=False,
               save_on=False,
               file_name="NEW10ep",
               show_score_plot=False,
               training_on=True)

#RL.run_optimal_game(n_times=1)

#RL.run_optimal_game_graph(10)

RL.learning_loop_create_graphs(10)

#learning_loop_multiple_agents_create_graph(100, 10, 10)


