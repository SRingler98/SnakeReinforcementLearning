from QLearning import QLearning
from QLearning import OptimalRunsQLearning as ORQL


class CreateReplayBufferData:
    def __init__(self):
        self.list_of_goals = [(1, 2),
                              (2, 2),
                              (3, 2),
                              (4, 2),
                              (1, 4),
                              (2, 4),
                              (3, 4),
                              (4, 4)]
        self.list_of_q_learning_policies = []
        self.replay_buffer_list = []

    def generate_different_agents(self, n_times, debug_on=False):
        for goal in self.list_of_goals:
            temp_policy = QLearning(goal).learning_loop(n_times, debug_on=debug_on)
            self.list_of_q_learning_policies.append(temp_policy)
            if debug_on:
                print("Goal: " + str(goal))
                print("Policy: " + str(temp_policy))

    def optimal_runs_q_learning(self, auto=False):
        count = 0
        while count < len(self.list_of_q_learning_policies):
            temp_optimal = ORQL(self.list_of_q_learning_policies[count], self.list_of_goals[count])
            self.replay_buffer_list.append(temp_optimal.run_optimal_n_times_given_policy(auto=auto))
            count += 1
        # print(self.replay_buffer_list)


# start of main

# crbd = CreateReplayBufferData()
# crbd.generate_different_agents(1000, debug_on=False)
# crbd.optimal_runs_q_learning(auto=True)
# replay_data = crbd.replay_buffer_list
#
# print(len(replay_data))
