import random
import pygame
import sys
import time
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import clone_model
from DisplayGrid import DisplayGrid
from SnakeEngine import SnakeEngine


def build_model(model_name):
    model = Sequential(name="model_name")
    model.add(layers.InputLayer(input_shape=(12,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='linear'))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


def build_model2(model_name):
    model = Sequential(name="model_name")
    model.add(layers.InputLayer(input_shape=(2,)))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(4, activation='linear'))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


def convert_from_grid_to_list(grid):
    temp_list = []

    for row in grid:
        for item in row:
            temp_list.append(item)

    return temp_list


def create_good_data():
    se = SnakeEngine(10)
    states, actions, rewards, next_states = se.run_game_in_real_time()

    temp_states = []

    for grid in states:
        temp_states.append(convert_from_grid_to_list(grid))

    states = temp_states

    temp_actions = []
    for item in actions:
        temp_action = convert_action_into_number(item)
        temp_actions.append(temp_action)

    actions = temp_actions

    temp_next_states = []

    for grid in next_states:
        temp_next_states.append(convert_from_grid_to_list(grid))

    next_states = temp_next_states

    max_count = len(states)

    index = 0

    episode_list = []

    while index < max_count:
        episode_list.append([states[index],
                             actions[index],
                             rewards[index],
                             next_states[index]])
        index += 1

    return episode_list


def load_replay_data_from_csv(filename):
    file = open(filename, 'r')
    reader = csv.reader(file)

    data = list(reader)

    print(data[0])
    print(type(data[0][0]))

    file.close()


class ActionSpace:
    def __init__(self):
        self.action_space_list = ['up', 'down', 'left', 'right']

    def get_action_space(self):
        return self.action_space_list

    def get_size_of_action_space(self):
        return len(self.action_space_list)


def pure_random_action():
    temp_action_space = ActionSpace()
    temp_action_list = temp_action_space.get_action_space()

    choice = random.randint(0, 100)

    if 0 <= choice < 25:
        return temp_action_list[0]
    elif 25 <= choice < 50:
        return temp_action_list[1]
    elif 50 <= choice < 75:
        return temp_action_list[2]
    elif 75 <= choice <= 100:
        return temp_action_list[3]


def convert_number_into_action(number):
    if number == 0:
        return 'up'
    elif number == 1:
        return 'down'
    elif number == 2:
        return 'left'
    elif number == 3:
        return 'right'


def convert_action_into_number(action):
    if action == 'up':
        return 0
    elif action == 'down':
        return 1
    elif action == 'left':
        return 2
    elif action == 'right':
        return 3


def get_max_value_from_tf_sensor(tf_array):
    max_list = []

    for x in tf_array:
        for row in x:
            max_value = 0
            max_index = 0
            temp_index = 0
            for item in row:
                if item > max_value:
                    max_value = item
                    max_index = temp_index
                temp_index += 1
            max_list.append(max_value)

    return max_list


def get_max_value_from_np_array(np_array):
    max_list = []

    for row in np_array:
        max_value = 0
        max_index = 0
        temp_index = 0
        for item in row:
            if item > max_value:
                max_value = item
                max_index = temp_index
            temp_index += 1
        max_list.append(max_value)

    return max_list


def reverse_items_in_list(list_given):
    temp_index = len(list_given) - 1
    temp_list = []

    while temp_index >= 0:
        temp_list.append(list_given[temp_index])
        temp_index -= 1

    return temp_list


class ReplayBuffer:
    def __init__(self, max_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.max_index = 0
        self.max_size = max_size

    def store(self, state, action, reward, next_state):
        if self.max_size == -1:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.max_index += 1
        else:
            if self.max_index < self.max_size:
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.next_states.append(next_state)
                self.max_index += 1
            else:
                self.states.pop(0)
                self.actions.pop(0)
                self.rewards.pop(0)
                self.next_states.pop(0)

                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.next_states.append(next_state)

    def get_mini_batch(self, batch_size=50):
        random_index_list = []

        for i in range(batch_size):
            random_index = random.randint(0, self.max_index - 1)
            random_index_list.append(random_index)

        temp_states = []
        temp_actions = []
        temp_rewards = []
        temp_next_states = []

        for index in random_index_list:
            temp_states.append(self.states[index])
            temp_actions.append(self.actions[index])
            temp_rewards.append(self.rewards[index])
            temp_next_states.append(self.next_states[index])

        return temp_states, temp_actions, temp_rewards, temp_next_states

    def get_entire_buffer(self):
        return self.states, self.actions, self.rewards, self.next_states

    def get_entire_buffer_reversed(self):
        temp_states = reverse_items_in_list(self.states)
        temp_actions = reverse_items_in_list(self.actions)
        temp_rewards = reverse_items_in_list(self.rewards)
        temp_next_states = reverse_items_in_list(self.next_states)
        return temp_states, temp_actions, temp_rewards, temp_next_states

    def get_size_of_replay_buffer(self):
        return self.max_index

    def load_data_into_buffer(self, replay_buffer_data, batch_size):
        while self.max_index < batch_size:
            for item in replay_buffer_data:
                self.states.append(item[0])
                self.actions.append(convert_action_into_number(item[1]))
                self.rewards.append(item[2])
                self.next_states.append(item[3])
                self.max_index += 1


# uses entire episodes as the indexes of the replay buffer
class ReplayBuffer2:
    def __init__(self, max_size):
        self.storage = []
        self.max_index = 0
        self.max_size = max_size

    def store(self, state_list, action_list, reward_list, next_state_list):

        length = len(action_list)

        temp_episode = []

        temp_index = 0

        while temp_index < length:
            temp_step = [state_list[temp_index],
                         action_list[temp_index],
                         reward_list[temp_index],
                         next_state_list[temp_index]
                         ]
            temp_index += 1
            temp_episode.append(temp_step)

        self.storage.append(temp_episode)

        self.max_index += 1

    def get_mini_batch(self, batch_size=50):
        random_index_list = []

        for i in range(batch_size):
            random_index = random.randint(0, self.max_index - 1)
            random_index_list.append(random_index)

        temp_episodes_list = [self.storage[len(self.storage) - 1]]

        for index in random_index_list:
            temp_episodes_list.append(self.storage[index])

        return temp_episodes_list

    def get_entire_buffer(self):
        return self.storage

    def get_size_of_replay_buffer(self):
        return self.max_index

    def load_data_into_buffer(self, replay_buffer_data, batch_size):
        while self.max_index < batch_size:
            for item in replay_buffer_data:
                self.storage.append(item)
                self.max_index += 1


class DQNModel:
    def __init__(self, model_location='models/'):
        self.model = build_model(model_location)
        self.model_location = model_location

    def argmax_action(self, state):
        prob_predictions = self.model.predict(state)

        max_value = 0
        max_index = 0
        temp_index = 0
        for prediction in prob_predictions:
            if prediction > max_value:
                max_value = prediction
                max_index = temp_index
            temp_index += 1

        return max_index

    def random_action_minus_max(self, state):
        max_action_number = self.policy(state)
        max_action = convert_number_into_action(max_action_number)

        temp_action_space = ActionSpace()
        temp_action_list = temp_action_space.get_action_space()

        total_actions_length = len(temp_action_list)

        if total_actions_length == 4:
            choice = random.randint(0, 100)
            if 0 <= choice < 25:
                return temp_action_list[0]
            elif 25 <= choice < 50:
                return temp_action_list[1]
            elif 50 <= choice < 75:
                return temp_action_list[2]
            elif 75 <= choice <= 100:
                return temp_action_list[3]

    def policy(self, state_in, debug=False):
        state_input = tf.convert_to_tensor(state_in, dtype=tf.float32)
        # print(state_input)
        action_q = self.model(state_input)
        if debug:
            print("\t" + str(action_q))
        action = np.argmax(action_q.numpy()[0], axis=0)
        return action

    def save_model(self):
        tf.keras.models.save_model(self.model, self.model_location)

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_location)


class DQNLearning:
    def __init__(self, env, target_name, epsilon=0.1, discount_factor=0.9, episode_count=1000, min_batch_size=50,
                 max_batch_size=150, load_model=True, fit_on_step=5, train=True, save_model=True, show_graphs=False):
        self.env = env
        self.target_name = target_name
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.episode_count = episode_count
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.load_model = load_model
        self.fit_on_step = fit_on_step
        self.q_was_pressed = False
        self.is_train = train
        self.save_model = save_model
        self.x_was_pressed = False
        self.show_graphs = show_graphs

    def train(self, debug=False, replay_buffer_data=None):
        start_time = time.time()
        agent = DQNModel(model_location=str('models/' + str(self.target_name)))

        if self.load_model:
            agent.load_model()

        if self.is_train:
            replay = ReplayBuffer2(self.max_batch_size)
            target = DQNModel()

            self.env.reset()
            self.sample_random_data(agent, replay, until=self.min_batch_size, debug=debug)

            # print("Play one game of snake!")
            # replay.storage.append(create_good_data())

            buffer_data = replay.get_mini_batch(batch_size=10)

            self.fit_replay2_data(buffer_data, agent, target)

            done_training = False
            if not self.is_train:
                done_training = True

            self.env.reset()

            current_episode_count = 0

            avg_rewards = []
            list_of_rewards = []

            improvement_score = 0
            previous_improvement_score = 0
            improvement_stayed_the_same = 0
            prob = (1 - self.epsilon + (self.epsilon / self.env.action_space_size)) * 100
            while not done_training:
                previous_improvement_score = improvement_score

                self.env.reset()
                step_count = 0
                policy_used = 0
                # self.epsilon = 1 - (current_episode_count / self.episode_count)
                total_reward = 0
                states = []
                actions = []
                rewards = []
                next_states = []
                while not self.env.get_terminal_state():
                    state = self.env.get_current_state()
                    states.append(state)
                    rand = random.randint(0, 100)

                    action = "null"
                    max_action_number = -1

                    state_list = [state]

                    if rand <= prob:
                        if debug:
                            print("Choosing max")
                        max_action_number = agent.policy(state_list)
                        action = convert_number_into_action(max_action_number)
                        policy_used += 1
                    else:
                        if debug:
                            print("choosing random")
                        # action = agent.random_action_minus_max(state_list)
                        action = pure_random_action()
                        max_action_number = convert_action_into_number(action)

                    next_state, reward, temp_done = self.env.step(action)

                    actions.append(max_action_number)
                    rewards.append(reward)
                    next_states.append(next_state)

                    total_reward += reward

                    if step_count % 5 == 0:
                        target.model = clone_model(agent.model)
                    step_count += 1

                    if step_count % 1000 == 0:
                        print("\tWARNING 1000 steps reached, stopping training")
                        break

                    if step_count % 100 == 0:
                        print("\tWARNING multiple of 100 steps!")
                    if step_count % 500 == 0:
                        print("\tWARNING 500 steps reached, stopping training")
                        break
                # end episode loop

                replay.store(states, actions, rewards, next_states)

                buffer_data = replay.get_mini_batch(batch_size=10)

                self.fit_replay2_data(buffer_data, agent, target)

                if current_episode_count >= self.episode_count:
                    done_training = True
                else:
                    current_episode_count += 1
                    improvement_score = total_reward / step_count
                    print("\tEpisode " + str(current_episode_count) + " finished\tSteps: " + str(step_count)
                          + "\tTotal Reward: " + str(total_reward) + "\tPolicy Used: " +
                          str((policy_used / step_count) * 100) + "%\tImprovement Score: " + str(improvement_score))
                    list_of_rewards.append(total_reward)
                    avg_rewards.append(total_reward / current_episode_count)

                    if improvement_score == previous_improvement_score:
                        improvement_stayed_the_same += 1
                    else:
                        improvement_stayed_the_same = 0

                    if improvement_stayed_the_same >= 25:
                        print("Model has not improved in 5 episodes, ending training early.")
                        done_training = True

            # end training loop
            if self.save_model:
                agent.save_model()

            print("\tReplay Buffer Size: " + str(replay.get_size_of_replay_buffer()))

            if self.show_graphs:
                if len(avg_rewards) > 0:
                    plt.plot(avg_rewards)
                    plt.show()

                if len(list_of_rewards) > 0:
                    plt.plot(list_of_rewards)
                    plt.show()

            agent.model.summary()

        end_time = time.time()

        print("Running Time: " + str(end_time - start_time))

        return agent

    def fit_replay_data(self, replay_data, agent, target):
        temp_states, temp_actions, temp_rewards, temp_next_states = replay_data

        different_state_list = tf.convert_to_tensor(temp_states, dtype=tf.float32)

        target_q = np.array(target.model(different_state_list))

        different_next_state_list = tf.convert_to_tensor(temp_next_states, dtype=tf.float32)
        max_q = np.array(target.model(different_next_state_list))

        previous_value = 0

        amount = len(temp_states)

        for i in range(amount):
            if i == 0:
                target_q[i][temp_actions[i]] += temp_rewards[i]
                previous_value = temp_rewards[i]
            else:
                update_value = temp_rewards[i] + self.discount_factor * (max_q[i][temp_actions[i]] - previous_value)
                target_q[i][temp_actions[i]] += update_value
                previous_value = update_value

        x = np.array(different_state_list)
        y = np.array(target_q)

        verbose_number = 0

        # gradient decent on model
        return agent.model.fit(x=x, y=y, verbose=verbose_number, batch_size=1)

    def fit_replay2_data(self, replay_data, agent, target):
        for episode in replay_data:
            length_of_episode = len(episode) - 1

            temp_states = []
            temp_actions = []
            temp_rewards = []
            temp_next_states = []

            while length_of_episode >= 0:
                temp_states.append(episode[length_of_episode][0])
                temp_actions.append(episode[length_of_episode][1])
                temp_rewards.append(episode[length_of_episode][2])
                temp_next_states.append(episode[length_of_episode][3])

                length_of_episode -= 1

            different_state_list = tf.convert_to_tensor(temp_states, dtype=tf.float32)

            target_q = np.array(target.model(different_state_list))

            different_next_state_list = tf.convert_to_tensor(temp_next_states, dtype=tf.float32)
            max_q = np.array(target.model(different_next_state_list))

            previous_value = 0

            amount = len(temp_states)

            for i in range(amount):
                if i == 0:
                    target_q[i][temp_actions[i]] += temp_rewards[i]
                    previous_value = temp_rewards[i]
                else:
                    update_value = temp_rewards[i] + self.discount_factor * (max_q[i][temp_actions[i]] - previous_value)
                    target_q[i][temp_actions[i]] += update_value
                    previous_value = update_value

            x = np.array(different_state_list)
            y = np.array(target_q)

            verbose_number = 0

            # gradient decent on model
            return agent.model.fit(x=x, y=y, verbose=verbose_number, batch_size=1)

    def sample_random_data(self, agent, replay, until, debug=False):
        temp_count = 0

        temp_states = []
        temp_actions = []
        temp_rewards = []
        temp_next_states = []

        while temp_count < until:
            state = self.env.get_current_state()
            temp_states.append(state)

            if self.env.get_terminal_state():
                self.env.reset()

                replay.store(temp_states, temp_actions, temp_rewards, temp_next_states)

                temp_states = []
                temp_actions = []
                temp_rewards = []
                temp_next_states = []

                state = self.env.get_current_state()
                temp_states.append(state)

            action = pure_random_action()
            max_action_number = convert_action_into_number(action)

            next_state, reward, temp_done = self.env.step(action)

            temp_actions.append(max_action_number)
            temp_rewards.append(reward)
            temp_next_states.append(next_state)

            temp_count += 1

    def evaluate(self, agent, num_of_times, epsilon=0.1):
        total_reward = 0
        for i in range(num_of_times):
            self.q_was_pressed = False
            self.x_was_pressed = False
            self.env.reset()
            step_count = 0
            while not self.env.get_terminal_state():
                self.env.render()
                if self.space_was_pressed():
                    if self.x_was_pressed:
                        break
                    state = self.env.get_current_state()
                    state_list = [state]
                    max_action_number = agent.policy(state_list, debug=True)
                    action = convert_number_into_action(max_action_number)
                    next_state, reward, temp_done = self.env.step(action)
                    print("\tstep #" + str(step_count) + " " + " " + str(action) + " " + str(reward))
                    total_reward += reward
                    step_count += 1
        return total_reward / num_of_times

    def space_was_pressed(self):
        self.env.render()
        button_pressed = False
        while not button_pressed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        button_pressed = True
                        return True
                    elif event.key == pygame.K_q:
                        button_pressed = True
                        self.q_was_pressed = not self.q_was_pressed
                        return True
                    elif event.key == pygame.K_x:
                        button_pressed = True
                        self.x_was_pressed = True
                        return True

                if self.q_was_pressed:
                    button_pressed = True
                    return True

        return False
