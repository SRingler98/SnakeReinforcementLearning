
# from SnakeEngine import SnakeEngine as SE
#
# snake = SE(10)
# snake.run_game_in_real_time()

from typing import final
from matplotlib import pyplot as plt
import numpy as np
import statistics as stat
import time
from QLearning import QLearning

"""RL = QLearning(3000,
               debug_on=False,
               visuals_on_while_training=False,
               load_on=False,
               save_on=False,
               file_name="saved_tables/my_Q_table.json",
               show_score_plot=False,
               training_on=False)"""

# RL.run_optimal_game(n_times=10)
# RL.run_optimal_game_graph(10)

#RL.learning_loop_create_graphs(5)

time0 = time.time()

episodes = 2000
agents = 100
optimals = 20

avgs_list = []

for i in range(0, agents):
    print("Agent " + str(i) + " of " + str(agents))
    agent = QLearning(episodes,
               debug_on=False,
               visuals_on_while_training=False,
               load_on=False,
               save_on=False,
               file_name="",
               show_score_plot=False,
               training_on=False)

    agent_scores = agent.learning_loop_create_data(optimals)
    avgs_list.append(agent_scores)

time1 = time.time()

print(time1-time0)

ep_count = []
for i in range(0, episodes):
    ep_count.append(i+1)

final_avgs = []
for i in range(0, episodes):
    current_list = []
    for j in avgs_list:
        current_list.append(j[i])
    final_avgs.append(stat.mean(current_list))

plt.scatter(ep_count, final_avgs)
plt.xlabel("Episodes Trained")
plt.ylabel("Average Score")
plt.title("Average Scores over Training Episodes")
z = np.polyfit(ep_count, final_avgs, 2)
p = np.polyval(z, ep_count)
plt.plot(ep_count, p, "r--")
plt.show()
