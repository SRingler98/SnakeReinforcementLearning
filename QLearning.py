import numpy as np
import random, json
import matplotlib.pyplot as plt
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
        self.epislon = 0.1

    def set_q_value(self, state, action, new_value):
        self.q_table[state][action] = new_value

    def get_q_value(self, state, action):
        return self.q_table[state][action]

    def get_max_a_q_value(self, state):
        max_action = "null"
        max_value = -1

        for action in self.list_of_actions:
            if self.get_q_value(state, action) > max_value:
                max_action = action
                max_value = self.get_q_value(state, action)

        return max_value

    def get_max_a_q_value_action(self, state):
        max_action = self.list_of_actions[0]
        max_value = self.get_q_value(state, self.list_of_actions[0])

        for action in self.list_of_actions[1::3]:
            if self.get_q_value(state, action) > max_value:
                max_action = action
                max_value = self.get_q_value(state, action)

        if max_action == "null":
            list_of_actions = ['up', 'down', 'left', 'right']

            # generate a random number between [1, 99]
            # note: 0 and 100 cannot be used, as it breaks the percentages
            random_number = random.randint(1, 99)

            # choose the action based on the random action
            if 0 < random_number <= 25:
                return list_of_actions[0]
            elif 25 < random_number <= 50:
                return list_of_actions[1]
            elif 50 < random_number <= 75:
                return list_of_actions[2]
            elif 75 < random_number < 100:
                return list_of_actions[3]
            else:
                # else something has gone wrong with choosing an action
                # WARNING: This should never be reachable
                print("Somethings gone wrong (MAX_action)")
        else:
            return max_action

    def update_q_value(self, state, action, value):
        self.q_table[state][action] = value

    # chooses action randomly given the values from the Q-table
    def choose_action_randomly_given_state(self, state):
        probability_of_choosing_best = (1 - self.epislon + (self.epislon / 4)) * 100

        random_chance = random.randint(0, 100)

        if random_chance < probability_of_choosing_best:
            return self.get_max_a_q_value_action(state)
        else:
            # code used from RinglerShawn_ReinforcementLearning_HW1
            list_of_actions = ['up', 'down', 'left', 'right']

            # generate a random number between [1, 99]
            # note: 0 and 100 cannot be used, as it breaks the percentages
            random_number = random.randint(1, 99)

            # choose the action based on the random action
            if 0 < random_number <= 25:
                return list_of_actions[0]
            elif 25 < random_number <= 50:
                return list_of_actions[1]
            elif 50 < random_number <= 75:
                return list_of_actions[2]
            elif 75 < random_number <= 100:
                return list_of_actions[3]
            else:
                # else something has gone wrong with choosing an action
                # WARNING: This should never be reachable
                print("Somethings gone wrong (Chosen action randomly)")

    def save_q_table_to_file(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.q_table, outfile)
            print("Saving q-table file was successful")

    def load_q_table_from_file(self, filename):
        with open(filename, 'r') as infile:
            self.q_table = json.load(infile)
            print("Loading q-table file was successful")


class QLearning:
    def __init__(self, episode_count, debug_on, visuals_on_while_training, load_on, save_on, file_name,
                 show_score_plot, training_on):
        self.q_table_class = QTable()
        self.episode_count = episode_count
        self.step_size = 0.9
        self.discount_value = 0.9
        self.snake_game = SnakeEngine(10)
        self.debug_mode = debug_on
        self.visual_mode = visuals_on_while_training

        if load_on:
            self.q_table_class.load_q_table_from_file(file_name)
        self.save_mode = save_on
        self.file_name = file_name
        self.show_score_plot = show_score_plot

        if training_on:
            self.learning_loop()

    def learning_loop(self):
        num_of_episodes = 0
        self.snake_game = SnakeEngine(10)
        list_of_scores = []

        while num_of_episodes < self.episode_count:

            episode_running = True
            self.snake_game.start_game_for_steps(self.visual_mode)
            current_state = self.snake_game.get_current_twelve_boolean_state()
            episdoe_data = {}
            step_count = 0

            while episode_running:
                chosen_action = self.q_table_class.choose_action_randomly_given_state(current_state)
                self.snake_game.move_player_step(chosen_action)
                new_state = self.snake_game.get_current_twelve_boolean_state()
                reward_value = self.snake_game.current_reward

                new_q_value = self.q_table_class.get_q_value(current_state, chosen_action)
                TD_error = reward_value + self.discount_value * self.q_table_class.get_max_a_q_value(new_state)
                TD_error -= self.q_table_class.get_q_value(current_state, chosen_action)
                new_q_value += self.step_size * TD_error

                self.q_table_class.update_q_value(current_state, chosen_action, new_q_value)

                if not self.snake_game.snake_alive:
                    episode_running = False
                else:
                    self.snake_game.refresh_after_step(self.visual_mode)

                episdoe_data[step_count] = [current_state, chosen_action, reward_value, new_state, new_q_value]
                step_count += 1

            if self.debug_mode:
                print(episdoe_data)

            num_of_episodes += 1
            list_of_scores.append(self.snake_game.score)

        if self.show_score_plot:
            plt.plot(list_of_scores)
            plt.ylabel('Score')
            plt.xlabel('Episodes')
            plt.show()
        print("Event Log: Training has finished.")
        if self.save_mode:
            self.q_table_class.save_q_table_to_file(self.file_name)

    def run_optimal_game(self, n_times=1):
        self.snake_game.run_game_using_policy(self.q_table_class, n_times)

    def save_q_table_to_file(self, filename):
        self.q_table_class.save_q_table_to_file(filename)


class ActOnEpisode:
    def __init__(self):
        self.snake_game = SnakeEngine(10)
        self.dict_of_actions_and_states = {}

    def run_one_episode(self, q_table_class):
        self.snake_game.start_game_for_steps(False)
        step_count = 0

        while self.snake_game.snake_alive:
            current_state = self.snake_game.get_current_twelve_boolean_state()

            chosen_action = q_table_class.choose_action_randomly_given_state(current_state)

            temp_list = [current_state, chosen_action]

            self.dict_of_actions_and_states[step_count] = temp_list

            self.snake_game.move_player_step(chosen_action)

        return self.dict_of_actions_and_states, self.snake_game.score

    def reset_episode(self):
        self.snake_game = SnakeEngine(10)
        self.dict_of_actions_and_states = {}
