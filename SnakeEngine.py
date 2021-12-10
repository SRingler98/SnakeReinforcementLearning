import sys, random
import numpy as np
import pygame
import matplotlib.pyplot as plt
from DisplayGrid import DisplayGrid as DG


# class to run and handle snake game
class SnakeEngine:
    # initializes snake game
    def __init__(self, grid_size):
        self.grid_size = grid_size  # sets grid size to input variable
        self.grid_array = np.zeros((grid_size, grid_size))  # creates a int array of zeroes based on the grid size
        self.player_head_pos = (4, 4)  # sets the player to start at 4,
        self.grid_array[self.player_head_pos[0]][self.player_head_pos[1]] = 1  # sets player position in grid array
        tail_segment_one = (self.player_head_pos[0] - 1, self.player_head_pos[1])
        tail_segment_two = (self.player_head_pos[0] - 2, self.player_head_pos[1])
        self.player_pos_list = [self.player_head_pos]  # sets the head to be the start of the tail
        self.player_pos_list.append(tail_segment_one)
        self.player_pos_list.append(tail_segment_two)
        self.add_snake_body_to_grid()
        self.display = DG(grid_size, 50, self.player_head_pos)
        self.apple_spawned = False
        self.apple_pos = (-1, -1)
        self.score = 0
        self.snake_alive = True
        self.current_reward = 0
        self.current_state = self.get_current_twelve_boolean_state()

    def add_snake_body_to_grid(self):
        for snake_part in self.player_pos_list:
            self.grid_array[snake_part[0]][snake_part[1]] = 1

        self.grid_array[self.player_pos_list[0][0]][self.player_pos_list[0][1]] = 4

    def run_game_in_real_time(self):
        temp_states = []
        temp_actions = []
        temp_rewards = []
        temp_next_states = []
        while self.snake_alive:
            if not self.apple_spawned:
                self.spawn_apple_randomly()
            self.display.draw_grid(self.grid_array)
            temp_states.append(np.copy(self.grid_array))
            temp_actions.append(self.event_handler())
            temp_rewards.append(self.current_reward)
            temp_next_states.append(np.copy(self.grid_array))
        print("Game Over!\nFinal Score was: " + str(self.score))

        return temp_states, temp_actions, temp_rewards, temp_next_states

    def get_current_twelve_int_boolean_state(self):
        temp_list = []

        player_pos_x = self.player_pos_list[0][0]
        player_pos_y = self.player_pos_list[0][1]
        apple_pos_x = self.apple_pos[0]
        apple_pos_y = self.apple_pos[1]

        # apple above
        if apple_pos_y > player_pos_y:
            temp_list.append(1)
        elif apple_pos_y <= player_pos_y:
            temp_list.append(0)

        # apple below
        if apple_pos_y < player_pos_y:
            temp_list.append(1)
        elif apple_pos_y >= player_pos_y:
            temp_list.append(0)

        # apple left
        if apple_pos_x < player_pos_x:
            temp_list.append(1)
        elif apple_pos_x >= player_pos_x:
            temp_list.append(0)

        # apple left
        if apple_pos_x > player_pos_x:
            temp_list.append(1)
        elif apple_pos_x <= player_pos_x:
            temp_list.append(0)

        # wall up
        if player_pos_y - 1 <= 0:
            temp_list.append(1)
        elif self.grid_array[player_pos_x][player_pos_y - 1] == 1:
            temp_list.append(1)
        else:
            temp_list.append(0)

        # wall down
        if player_pos_y + 1 >= 10:
            temp_list.append(1)
        elif self.grid_array[player_pos_x][player_pos_y + 1] == 1:
            temp_list.append(1)
        else:
            temp_list.append(0)

        # wall left
        if player_pos_x - 1 <= 0:
            temp_list.append(1)
        elif self.grid_array[player_pos_x - 1][player_pos_y] == 1:
            temp_list.append(1)
        else:
            temp_list.append(0)

        # wall right
        if player_pos_x + 1 >= 10:
            temp_list.append(1)
        elif self.grid_array[player_pos_x + 1][player_pos_y] == 1:
            temp_list.append(1)
        else:
            temp_list.append(0)

        # get position difference to see how snake is facing
        position_difference = (self.player_pos_list[0][0] - self.player_pos_list[1][0],
                               self.player_pos_list[0][1] - self.player_pos_list[1][1])

        # snake facing up
        if position_difference == (0, -1):
            temp_list.append(1)
        else:
            temp_list.append(0)

        # snake facing down
        if position_difference == (0, 1):
            temp_list.append(1)
        else:
            temp_list.append(0)

        # snake facing left
        if position_difference == (-1, 0):
            temp_list.append(1)
        else:
            temp_list.append(0)

        # snake facing right
        if position_difference == (1, 0):
            temp_list.append(1)
        else:
            temp_list.append(0)

        return temp_list

    def run_game_using_policy(self, q_table, n_times):
        print("Optimal demo is ready to run. Press Space to step through an episode.")

        optimal_runs = 0

        while optimal_runs < n_times:
            self.start_game_for_steps(True)
            while self.snake_alive:
                if not self.apple_spawned:
                    self.spawn_apple_randomly()
                self.display.draw_grid(self.grid_array)

                if self.space_was_pressed():
                    self.current_state = self.get_current_twelve_boolean_state()
                    print("Current State: " + str(self.current_state))
                    chosen_action = q_table.choose_action_optimally(self.current_state)
                    print("\tChosen action: " + str(chosen_action))
                    print("\tQTable Values: " + str(q_table.q_table[self.current_state]))
                    self.move_player_step(chosen_action)
                    self.current_state = self.get_current_twelve_boolean_state()

            print("Game Over!\nFinal Score was: " + str(self.score))

            self.grid_array[self.player_pos_list[0][0]][self.player_pos_list[0][1]] = 3
            self.current_state = self.get_current_twelve_boolean_state()

            self.space_was_pressed()

            optimal_runs += 1

    def run_game_using_policy_and_generate_graph_then_demo(self, q_table, n_times):
        print("Generating Optimal Data")

        optimal_runs = 0
        score_list = []

        while optimal_runs < n_times:
            self.start_game_for_steps(False)
            step_count = 0
            while self.snake_alive:
                if not self.apple_spawned:
                    self.spawn_apple_randomly()
                # self.display.draw_grid(self.grid_array, self.current_state)

                self.current_state = self.get_current_twelve_boolean_state()

                chosen_action = q_table.choose_action_optimally(self.current_state)
                self.move_player_step(chosen_action)
                self.current_state = self.get_current_twelve_boolean_state()

                step_count += 1

                if step_count > 200:
                    self.snake_alive = False

            self.grid_array[self.player_pos_list[0][0]][self.player_pos_list[0][1]] = 3
            self.current_state = self.get_current_twelve_boolean_state()

            score_list.append(self.score)

            optimal_runs += 1

        x_values = []
        for x in range(n_times):
            x_values.append(x)

        plt.scatter(x_values, score_list)
        plt.show()

        self.run_game_using_policy(q_table, 10)

    def run_game_using_policy_and_return_scores(self, q_table, n_times):
        """
        Runs the Snake game using the policy supplied.
        No exploration. Only optimal choices.
        Dies if caught in a loop.
        Returns a list of scores for all of the games.
        """

        optimal_runs = 0
        score_list = []

        while optimal_runs < n_times:
            self.start_game_for_steps(False)
            step_count = 0
            while self.snake_alive:
                if not self.apple_spawned:
                    self.spawn_apple_randomly()
                # self.display.draw_grid(self.grid_array, self.current_state)

                self.current_state = self.get_current_twelve_boolean_state()

                chosen_action = q_table.choose_action_optimally(self.current_state)
                self.move_player_step(chosen_action)
                self.current_state = self.get_current_twelve_boolean_state()

                step_count += 1

                if step_count > 200:    #snake caught in loop
                    self.snake_alive = False

            self.grid_array[self.player_pos_list[0][0]][self.player_pos_list[0][1]] = 3
            self.current_state = self.get_current_twelve_boolean_state()

            score_list.append(self.score)

            optimal_runs += 1

        return score_list


    def start_game_for_steps(self, display_on):
        self.snake_alive = True
        # creates a int array of zeroes based on the grid size
        self.grid_array = np.zeros((self.grid_size, self.grid_size))
        self.player_head_pos = (4, 4)  # sets the player to start at 4,
        self.grid_array[self.player_head_pos[0]][self.player_head_pos[1]] = 1  # sets player position in grid array
        tail_segment_one = (self.player_head_pos[0] - 1, self.player_head_pos[1])
        tail_segment_two = (self.player_head_pos[0] - 2, self.player_head_pos[1])
        self.player_pos_list = [self.player_head_pos]  # sets the head to be the start of the tail
        self.player_pos_list.append(tail_segment_one)
        self.player_pos_list.append(tail_segment_two)
        self.add_snake_body_to_grid()
        self.apple_spawned = False
        self.spawn_apple_randomly()
        self.current_reward = 0
        self.current_state = self.get_current_twelve_boolean_state()
        self.score = 0
        if display_on:
            self.display.draw_grid(self.grid_array)

    def refresh_after_step(self, display_on):
        if not self.apple_spawned:
            self.spawn_apple_randomly()
        if display_on:
            self.display.draw_grid(self.grid_array)

    def get_current_twelve_boolean_state(self):
        temp_string = ""
        player_pos_x = self.player_pos_list[0][0]
        player_pos_y = self.player_pos_list[0][1]
        apple_pos_x = self.apple_pos[0]
        apple_pos_y = self.apple_pos[1]

        # apple above
        if apple_pos_y > player_pos_y:
            temp_string += "1"
        elif apple_pos_y <= player_pos_y:
            temp_string += "0"

        # apple below
        if apple_pos_y < player_pos_y:
            temp_string += "1"
        elif apple_pos_y >= player_pos_y:
            temp_string += "0"

        # apple left
        if apple_pos_x < player_pos_x:
            temp_string += "1"
        elif apple_pos_x >= player_pos_x:
            temp_string += "0"

        # apple left
        if apple_pos_x > player_pos_x:
            temp_string += "1"
        elif apple_pos_x <= player_pos_x:
            temp_string += "0"

        # wall up
        if player_pos_y - 1 <= 0:
            temp_string += '1'
        elif self.grid_array[player_pos_x][player_pos_y - 1] == 1:
            temp_string += "1"
        else:
            temp_string += "0"

        # wall down
        if player_pos_y + 1 >= 10:
            temp_string += "1"
        elif self.grid_array[player_pos_x][player_pos_y + 1] == 1:
            temp_string += "1"
        else:
            temp_string += "0"

        # wall left
        if player_pos_x - 1 <= 0:
            temp_string += "1"
        elif self.grid_array[player_pos_x - 1][player_pos_y] == 1:
            temp_string += "1"
        else:
            temp_string += "0"

        # wall right
        if player_pos_x + 1 >= 10:
            temp_string += "1"
        elif self.grid_array[player_pos_x + 1][player_pos_y] == 1:
            temp_string += "1"
        else:
            temp_string += "0"

        # get position difference to see how snake is facing
        position_difference = (self.player_pos_list[0][0] - self.player_pos_list[1][0],
                               self.player_pos_list[0][1] - self.player_pos_list[1][1])

        # snake facing up
        if position_difference == (0, -1):
            temp_string += "1"
        else:
            temp_string += "0"

        # snake facing down
        if position_difference == (0, 1):
            temp_string += "1"
        else:
            temp_string += "0"

        # snake facing left
        if position_difference == (-1, 0):
            temp_string += "1"
        else:
            temp_string += "0"

        # snake facing right
        if position_difference == (1, 0):
            temp_string += "1"
        else:
            temp_string += "0"

        return temp_string

    def event_handler(self):
        button_pushed = False
        while not button_pushed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        self.move_player_step('right')
                        button_pushed = True
                        return 'right'
                    elif event.key == pygame.K_LEFT:
                        self.move_player_step('left')
                        button_pushed = True
                        return 'left'
                    elif event.key == pygame.K_UP:
                        self.move_player_step('up')
                        button_pushed = True
                        return 'up'
                    elif event.key == pygame.K_DOWN:
                        self.move_player_step('down')
                        button_pushed = True
                        return 'down'

    def space_was_pressed(self):
        self.display.draw_grid(self.grid_array)
        button_pressed = False
        while not button_pressed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        print("Space was pressed")
                        button_pressed = True
                        return True
        return False

    def remove_tail_from_grid(self, state_tuple):
        self.grid_array[state_tuple[0]][state_tuple[1]] = 0

    def add_tail_onto_grid(self, state_tuple):
        self.grid_array[state_tuple[0]][state_tuple[1]] = 1

    def spawn_apple_randomly(self):
        spawned_incorrectly = True
        random_int_x = -1
        random_int_y = -1
        while spawned_incorrectly:
            random_int_x = random.randint(0, 9)
            random_int_y = random.randint(0, 9)
            if self.grid_array[random_int_x][random_int_y] != 1:
                spawned_incorrectly = False
        self.apple_pos = (random_int_x, random_int_y)
        self.grid_array[random_int_x][random_int_y] = 2
        self.apple_spawned = True

    def move_player_step(self, action):
        if action == 'right':
            self.move_player_right()
        elif action == 'left':
            self.move_player_left()
        elif action == 'up':
            self.move_player_up()
        elif action == 'down':
            self.move_player_down()

    # the move player functions work by changing the delta for x or y

    def move_player_right(self):
        x = 1
        y = 0
        self.move_tail(x, y)

    def move_player_left(self):
        x = -1
        y = 0
        self.move_tail(x, y)

    def move_player_up(self):
        x = 0
        y = -1
        self.move_tail(x, y)

    def move_player_down(self):
        x = 0
        y = 1
        self.move_tail(x, y)

    def move_tail(self, x, y):
        if self.check_if_on_body_or_wall(x, y):
            self.snake_alive = False
            self.current_reward = -100
        else:
            for segment in self.player_pos_list:
                self.remove_tail_from_grid(segment)

            previous_state = self.player_pos_list[0]
            count = 0
            while count < len(self.player_pos_list):
                if count == 0:
                    temp_state_tuple_x = self.player_pos_list[count][0] + x
                    temp_state_tuple_y = self.player_pos_list[count][1] + y
                    self.player_pos_list[count] = (temp_state_tuple_x, temp_state_tuple_y)
                else:
                    temp_prev_state = self.player_pos_list[count]
                    self.player_pos_list[count] = previous_state
                    previous_state = temp_prev_state
                count += 1

            if self.check_if_on_apple():
                self.player_pos_list.append(previous_state)
                self.score += 1
                self.apple_spawned = False
                self.add_snake_body_to_grid()
                self.current_reward = 100
            else:
                self.add_snake_body_to_grid()
                self.current_reward = 0

    def check_if_on_apple(self):
        if self.player_pos_list[0] == self.apple_pos:
            return True
        else:
            return False

    def check_if_on_body_or_wall(self, x, y):
        temp_player_pos_x = self.player_pos_list[0][0] + x
        temp_player_pos_y = self.player_pos_list[0][1] + y
        if temp_player_pos_x < 0 or temp_player_pos_x > 9 or temp_player_pos_y < 0 or temp_player_pos_y > 9:
            return True
        elif self.grid_array[temp_player_pos_x][temp_player_pos_y] == 1:
            return True
        else:
            return False
