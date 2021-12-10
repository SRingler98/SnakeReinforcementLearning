from SnakeEngine import SnakeEngine
from DQNClasses import convert_action_into_number
import csv


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


class CreateReplayBufferData:
    pass


# start of main

crbd = CreateReplayBufferData()
create_good_data()
