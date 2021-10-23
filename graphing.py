# Creating graphs to measure performance gains

from QLearning import QLearning

def graph_performance():
    num_of_episodes_list = [1, 10, 100, 1000, 10000, 100000]
    for n in num_of_episodes_list:
        agent = QLearning(n,
                          debug_on=False,
                          visuals_on_while_training=False,
                          load_on=False,
                          save_on=False,
                          file_name="",
                          show_score_plot=False,
                          training_on=True)

        scores = []
        avg_score = 0
        max_score = 0
        min_score = 0

        agent.run_optimal_game(n_times=1)

graph_performance()