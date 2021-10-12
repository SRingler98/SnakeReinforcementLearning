import numpy as np
import random
from SnakeEngine import SnakeEngine


class TwelveBooleanState:
    def __init__(self):
        self.list_of_binary = []
        self.list_of_states = []
        for x in range(12):
            self.list_of_binary.append(0)
        self.generate_list_of_twelve_binary_state_string()

    def go_up_one_binary(self, index):
        if index < len(self.list_of_binary):
            self.list_of_binary[index] += 1
            if self.list_of_binary[index] == 2:
                self.go_up_one_binary(index + 1)
                self.list_of_binary[index] = 0

    def print_binary(self):
        temp_dict = {}
        count = 0
        temp_string = ""
        for digit in self.list_of_binary:
            temp_dict[count] = digit
            count += 1

        count -= 1
        temp_reverse_dict = {}

        for item in temp_dict:
            temp_reverse_dict[count - item] = temp_dict[item]

        for item in temp_reverse_dict:
            temp_string += str(temp_dict[item])

        print(temp_string)

    def generate_list_of_twelve_binary_state_string(self):
        list_of_binary = []
        binary_count = 0

        while binary_count < 4096:
            temp_dict = {}
            count = 0
            temp_string = ""
            for digit in self.list_of_binary:
                temp_dict[count] = digit
                count += 1

            count -= 1
            temp_reverse_dict = {}

            for item in temp_dict:
                temp_reverse_dict[count - item] = temp_dict[item]

            for item in temp_reverse_dict:
                temp_string += str(temp_dict[item])

            list_of_binary.append(temp_string)
            self.go_up_one_binary(0)
            binary_count += 1

        self.list_of_states = list_of_binary

    def get_apple_above_snake(self, state_string):
        if self.list_of_states[state_string][0] == '1':
            return True
        else:
            return False

    def get_apple_below_snake(self, state_string):
        if self.list_of_states[state_string][1] == '1':
            return True
        else:
            return False

    def get_apple_left_snake(self, state_string):
        if self.list_of_states[state_string][2] == '1':
            return True
        else:
            return False

    def get_apple_right_snake(self, state_string):
        if self.list_of_states[state_string][3] == '1':
            return True
        else:
            return False

    def get_wall_above_snake(self, state_string):
        if self.list_of_states[state_string][4] == '1':
            return True
        else:
            return False

    def get_wall_below_snake(self, state_string):
        if self.list_of_states[state_string][5] == '1':
            return True
        else:
            return False

    def get_wall_left_snake(self, state_string):
        if self.list_of_states[state_string][6] == '1':
            return True
        else:
            return False

    def get_wall_right_snake(self, state_string):
        if self.list_of_states[state_string][7] == '1':
            return True
        else:
            return False

    def get_snake_facing_above(self, state_string):
        if self.list_of_states[state_string][8] == '1':
            return True
        else:
            return False

    def get_snake_facing_below(self, state_string):
        if self.list_of_states[state_string][9] == '1':
            return True
        else:
            return False

    def get_snake_facing_left(self, state_string):
        if self.list_of_states[state_string][10] == '1':
            return True
        else:
            return False

    def get_snake_facing_right(self, state_string):
        if self.list_of_states[state_string][11] == '1':
            return True
        else:
            return False


class Rewards:
    def __init__(self):
        self.reward_storage = {}
        self.state_class = TwelveBooleanState()
        for state in self.state_class.list_of_states:
            self.reward_storage[state] = 0

    def get_reward(self, state_string):
        return self.reward_storage[state_string]


class QTable:
    def __init__(self):
        self.q_table = {}
        self.states_class = TwelveBooleanState()
        self.list_of_actions = ['up', 'down', 'left', 'right']
        for state in self.states_class.list_of_states:
            temp_action_dict = {}
            for action in self.list_of_actions:
                temp_action_dict[action] = 0.25
            self.q_table[state] = temp_action_dict

    def set_q_value(self, state, action, new_value):
        self.q_table[state][action] = new_value

    def get_q_value(self, state, action):
        return self.q_table[state][action]

    def choose_action_randomly_given_state(self, state):
        # code used from RinglerShawn_ReinforcementLearning_HW1
        list_of_actions = ['up', 'down', 'left', 'right']

        # get the values stored in the policy for the four actions
        first_percentage = self.q_table[state][list_of_actions[0]]
        second_percentage = self.q_table[state][list_of_actions[1]]
        third_percentage = self.q_table[state][list_of_actions[2]]
        fourth_percentage = self.q_table[state][list_of_actions[3]]

        # create a list of these percentages
        list_of_percentages = [first_percentage, second_percentage, third_percentage, fourth_percentage]

        # sum these percentages up
        sum_of_percentages = sum(list_of_percentages)

        # normalize these percentages to be on a [0, 100] int scale
        first_percentage = (first_percentage / sum_of_percentages) * 100
        second_percentage = (second_percentage / sum_of_percentages) * 100
        third_percentage = (third_percentage / sum_of_percentages) * 100
        fourth_percentage = (fourth_percentage / sum_of_percentages) * 100

        # makes the percentages cumulative
        second_percentage += first_percentage
        third_percentage += second_percentage
        fourth_percentage += third_percentage

        # generate a random number between [1, 99]
        # note: 0 and 100 cannot be used, as it breaks the percentages
        random_number = random.randint(1, 99)

        # choose the action based on the random action
        if 0 < random_number <= first_percentage:
            return list_of_actions[0]
        elif first_percentage < random_number <= second_percentage:
            return list_of_actions[1]
        elif second_percentage < random_number <= third_percentage:
            return list_of_actions[2]
        elif third_percentage < random_number <= fourth_percentage:
            return list_of_actions[3]
        else:
            # else something has gone wrong with choosing an action
            # WARNING: This should never be reachable
            print("Somethings gone wrong")


class QLearning:
    def __init__(self, episode_count):
        self.q_table_class = QTable()
        self.episode_count = episode_count
        self.act_on_episode = ActOnEpisode()

    def learn(self):
        num_of_episodes = 0

        while num_of_episodes < self.episode_count:
            self.act_on_episode.run_one_episode(self.q_table_class)


class ActOnEpisode:
    def __init__(self):
        self.snake_game = SnakeEngine(10)
        self.dict_of_actions_and_states = {}

    def run_one_episode(self, q_table_class):
        self.snake_game.start_game_in_steps(False)
        step_count = 0

        while self.snake_game.snake_alive:
            current_state = self.snake_game.get_current_twelve_boolean_state()

            chosen_action = q_table_class.choose_action_randomly_given_state(current_state)

            temp_list = [current_state, chosen_action]

            self.dict_of_actions_and_states[step_count] = temp_list

            self.snake_game.move_player(chosen_action)

        return self.dict_of_actions_and_states

    def reset_episode(self):
        self.snake_game = SnakeEngine(10)
        self.dict_of_actions_and_states = {}




# main function

test_q = QTable()
print(test_q.q_table)
