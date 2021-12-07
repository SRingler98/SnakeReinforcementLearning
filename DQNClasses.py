import random
import pygame
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import clone_model
from DisplayGrid import DisplayGrid
# from CreateReplayBufferData import CreateReplayBufferData as CRBD


def build_model(model_name):
    model = Sequential(name="model_name")
    model.add(layers.InputLayer(input_shape=(100,)))
    model.add(layers.Dense(1024, activation='sigmoid'))
    model.add(layers.Dense(1024, activation='sigmoid'))
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


class ReplayBuffer2:
    def __init__(self, max_size):
        self.storage = []
        self.max_index = 0
        self.max_size = max_size

    def store(self, state_list, action_list, reward_list, next_state_list):

        length = len(state_list)

        temp_episode = []

        temp_index = 0

        while temp_index < length:
            temp_step = []
            temp_step.append(state_list[temp_index])
            temp_step.append(action_list[temp_index])
            temp_step.append(reward_list[temp_index])
            temp_step.append(next_state_list[temp_index])
            temp_index += 1
            temp_episode.append(temp_step)

        self.storage.append(temp_episode)

        self.max_index += 1

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

        # removed_action_list = []
        #
        # for item in temp_action_list:
        #     if item != max_action:
        #         removed_action_list.append(item)

        total_actions_length = len(temp_action_list)

        # if total_actions_length == 3:
        #     choice = random.randint(0, 100)
        #     if 0 <= choice < 33:
        #         return removed_action_list[0]
        #     elif 33 <= choice < 66:
        #         return removed_action_list[1]
        #     elif 66 <= choice <= 100:
        #         return removed_action_list[2]

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

    def policy(self, state_in):
        state_input = tf.convert_to_tensor(state_in, dtype=tf.float32)
        # print(state_input)
        action_q = self.model(state_input)
        action = np.argmax(action_q.numpy()[0], axis=0)
        return action

    def save_model(self):
        tf.keras.models.save_model(self.model, self.model_location)

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_location)
        # self.model = tf.saved_model.load(self.model_location)


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
        replay = ReplayBuffer(self.max_batch_size)

        agent = DQNModel(model_location=str('models/' + str(self.target_name)))

        if self.load_model:
            agent.load_model()

        target = DQNModel()
        target.model = clone_model(agent.model)

        done_training = False
        self.env.reset()
        state = self.env.current_state
        self.sample_random_data(agent, replay, until=self.min_batch_size, debug=debug)

        done_training = False
        if not self.is_train:
            done_training = True

        self.env.reset()

        current_episode_count = 0
        old_epsilon = self.epsilon

        avg_rewards = []
        list_of_rewards = []

        improvement_score = 0
        previous_improvement_score = 0
        improvement_stayed_the_same = 0

        while not done_training:
            previous_improvement_score = improvement_score
            # state = self.env.current_state
            #
            # state_list = [[state[0], state[1]]]
            #
            # max_action_number = agent.policy(state_list)
            #
            # print("Max Action: " + str(max_action_number))
            # print(convert_number_into_action(max_action_number))
            #
            # break
            self.env.reset()
            state = self.env.current_state
            step_count = 0
            policy_used = 0
            # self.epsilon = 1 - (current_episode_count / self.episode_count)
            total_reward = 0
            while not self.env.get_terminal_state():
                state = self.env.current_state
                prob = (1 - self.epsilon + (self.epsilon / self.env.action_space_size)) * 100
                # prob = self.epsilon * 100
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

                total_reward += reward

                # print("\tStep " + str(step_count) + " " + str(state) + " " + str(action) + " " + str(reward) +
                #       " " + str(next_state))

                replay.store(state, max_action_number, reward, next_state)

                different_state_list = tf.convert_to_tensor(state_list, dtype=tf.float32)

                target_q = np.array(target.model(different_state_list))

                next_state_batch_tf_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)

                next_q = np.array(target.model(next_state_batch_tf_tensor))
                max_next_q = get_max_value_from_np_array(next_q)

                if state == self.env.get_terminal_state():
                    target_q[0][max_action_number] += reward
                else:
                    target_q[0][max_action_number] += reward + self.discount_factor * max_next_q[0]

                x = np.array(different_state_list)
                y = np.array(target_q)

                verbose_number = 0

                # gradient decent on model
                result = agent.model.fit(x=x, y=y, verbose=verbose_number, batch_size=1)

                # mini_batch = replay.get_mini_batch()

                # if there are more then 50 actions stored in the replay buffer
                # if step_count % self.fit_on_step == 0 and step_count != 0:
                #     if replay.get_size_of_replay_buffer() > self.min_batch_size:
                #         # state_batch, action_batch, reward_batch, next_state_batch = replay.get_entire_buffer()
                #         state_batch, action_batch, reward_batch, next_state_batch = replay.get_mini_batch()
                #
                #         size_of_mini_batch = len(state_batch)
                #
                #         next_state_batch_list = []
                #
                #         for item in next_state_batch:
                #             next_state_batch_list.append([item])
                #
                #         next_state_batch_tf_tensor = tf.convert_to_tensor(next_state_batch_list, dtype=tf.float32)
                #
                #         different_state_list = tf.convert_to_tensor(state_batch, dtype=tf.float32)
                #         # current_q = agent.model(different_state_list)
                #         target_q = np.array(target.model(different_state_list))
                #
                #         next_q = target.model(next_state_batch_tf_tensor)
                #         max_next_q = get_max_value_from_tf_sensor(next_q)
                #
                #         for i in range(len(target_q)):
                #             if state == self.env.get_terminal_state():
                #                 target_q[i][action_batch[i]] = reward_batch[i]
                #             else:
                #                 target_q[i][action_batch[i]] = reward_batch[i] + self.discount_factor * max_next_q[i]
                #
                #         verbose_number = 0
                #         if debug:
                #             verbose_number = 1
                #
                #         x = np.array(different_state_list)
                #         y = np.array(target_q)
                #
                #         # gradient decent on model
                #         result = agent.model.fit(x=x, y=y, verbose=verbose_number, batch_size=1)
                #
                #         if debug:
                #             print(result)

                if step_count % 5 == 0:
                    target.model = clone_model(agent.model)
                step_count += 1

                if step_count % 100 == 0:
                    print("\tWARNING 100 steps reached, stopping training")
                    break

                if step_count % 100 == 0:
                    print("\tWARNING multiple of 100 steps!")
                if step_count % 500 == 0:
                    print("\tWARNING 500 steps reached, stopping training")
                    break
            # end episode loop

            if current_episode_count >= self.episode_count:
                done_training = True
            else:
                current_episode_count += 1
                improvement_score = total_reward / step_count
                # print("\tEpisode " + str(current_episode_count) + " finished Steps:   " + str(step_count) + " "
                #       + str(policy_used / step_count) + "%")
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

        return agent

    def sample_random_data(self, agent, replay, until, debug=False):
        temp_count = 0
        while temp_count < until:
            state = self.env.current_state
            if self.env.get_terminal_state():
                self.env.reset()
                state = self.env.current_state

            action = pure_random_action()
            max_action_number = convert_action_into_number(action)

            next_state, reward, temp_done = self.env.step(action)

            replay.store(state, max_action_number, reward, next_state)
            temp_count += 1

    def evaluate(self, agent, num_of_times, epsilon=0.1):
        self.q_was_pressed = False
        self.x_was_pressed = False
        total_reward = 0
        for i in range(num_of_times):
            self.env.reset()
            state = self.env.current_state
            step_count = 0
            while not self.env.get_terminal_state():
                self.env.render()
                if self.space_was_pressed():
                    if self.x_was_pressed:
                        break
                    state = self.env.current_state
                    state_list = [state]
                    prob = (1 - epsilon + (epsilon / self.env.action_space_size)) * 100
                    rand = random.randint(0, 100)
                    if rand < prob:
                        max_action_number = agent.policy(state_list)
                    else:
                        temp_action = agent.random_action_minus_max(state_list)
                        max_action_number = convert_action_into_number(temp_action)
                    action = convert_number_into_action(max_action_number)
                    next_state, reward, temp_done = self.env.step(action)
                    print("\tstep #" + str(step_count) + " " + str(state) + " " + str(action) + " " + str(reward) +
                          " " + str(next_state))
                    total_reward += reward
                    state = next_state
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

# example of calling DQNClasses for learning
# print("Creating model")
# env = SnakeEnv()
# temp_learn = DQNLearning(env=env,
#                          target_name=str(str(target[0]) + "," + str(target[1])),
#                          episode_count=10,
#                          min_batch_size=50,
#                          max_batch_size=-1,
#                          load_model=False,
#                          fit_on_step=10,
#                          train=True,
#                          save_model=False,
#                          show_graphs=False)
#
# print("Training model")
# temp_agent = temp_learn.train(debug=False,
#                               # replay_buffer_data=replay_data[target_index]
#                               )
