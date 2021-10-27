# Creating graphs to measure performance gains
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np

from QLearning import QLearning

def learning_loop_multiple_agents_create_graph(episodes, agents, optimals):
    """
    Runs the learning loop to create average score data on multiple agents
    and plots the average of all of the agent averages.

    Args: 
        episodes: Number of episodes to train the agent(s)
        agents: Number of agents to train
        optimals: Number of optimal runs (no exploration) per episode

    Returns:
        Returns nothing for now but outputs a matplotlib graph which 
        can then be saved.
    """
    #time0 = time.time()

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

    #time1 = time.time()
    #print(time1-time0)

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
    #z = np.polyfit(ep_count, final_avgs, 2)
    #p = np.polyval(z, ep_count)
    #plt.plot(ep_count, p, "r--")
    plt.show()

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
