# Creating graphs to measure performance gains
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np

from QLearning import QLearning

def graph_performance():
    """
    This function iteratively creates a new agent for every increment in episode amount
    and takes an average of scores from that agent's final learned Q-table then graphs 
    all of the average scores.
    WARNING: Due to the fact that a new agent is created every single iteration, the 
    amount of times the SnakeGame runs becomes O(n^2) instead of O(n).
    """
    avgs = []
    maxs = []
    mins = []
    meds = []

    num_of_episodes_list = []
    
    for i in range(0,500):
        num_of_episodes_list.append(i)

    for n in num_of_episodes_list:
        print("Current episode number: ", n)
        scores_list = []
        for agent_num in range(0,100):
            agent = QLearning(n,
                          debug_on=False,
                          visuals_on_while_training=False,
                          load_on=False,
                          save_on=False,
                          file_name="",
                          show_score_plot=False,
                          training_on=True)

            scores_list.append(agent.run_optimal_game_and_return_scores(n_times=100))

        averages_list = []
        for scores in scores_list:
            averages_list.append(stat.mean(scores))
        
        final_avg = stat.mean(averages_list)
        
        avgs.append(final_avg)


    # plotting
    plt.scatter(num_of_episodes_list, avgs) #average scores
    #plt.scatter(num_of_episodes_list, maxs) #max scores
    plt.ylabel('Average Score')
    plt.xlabel('Episodes Trained')
    plt.show()

graph_performance()