# program created by Shawn Ringler
# For CS4080 Homework Assignment 1
# 9-29-2021

import pygame
import numpy


# maze class to hold values of the maze for the display grid
class MazeGenerator:
    # initialize the maze with the size given
    def __init__(self, size_input):
        self.size = size_input
        self.maze_grid = numpy.zeros((self.size, self.size))
        self.player_position = [0, 0]

    # get the grid value of the coords given
    def get_grid_value(self, x, y):
        return self.maze_grid[x][y]

    # change the gridvalue of (x, y) with the value given
    def change_grid_value(self, x, y, value):
        self.maze_grid[x][y] = value

    # get the size of the grid
    def get_grid_size(self):
        return self.size

    # get the grid itself
    def get_grid(self):
        return self.maze_grid

    # change where the player position is
    def change_player_position(self, state_tuple):
        self.change_grid_value(self.player_position[0], self.player_position[1], 0)
        self.player_position[0] = state_tuple[0]
        self.player_position[1] = state_tuple[1]
        self.change_grid_value(self.player_position[0], self.player_position[1], 1)

    # set the player position to the start
    def set_player_position_to_start(self):
        self.player_position = [0, 0]
        self.change_grid_value(self.player_position[0], self.player_position[1], 1)


# colors class to hold color RGB values
class Colors:
    def __init__(self):
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.gray = (100, 100, 100)
        self.purple = (255, 0, 255)
        self.dark_green = (0, 150, 0)


# display grid class displays the grid of the grid to the screen
class DisplayGrid:
    # DisplayGrid uses the pygame library to display the grid onto the screen
    def __init__(self, size_in, block_size, player_starting_pos):
        self.block_size = block_size
        self.grid_size = size_in
        pygame.init()
        self.size = 1000, 500
        self.colors = Colors()
        self.screen = pygame.display.set_mode(self.size)
        self.screen.fill(self.colors.black)
        pygame.display.set_caption("Snake Reinforcement Learning")  # player
        # walls
        self.q_was_pressed = False
        self.x_was_pressed = False

    def draw_grid(self, grid):
        # draw the empty grid first
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                maze_test_value = grid[x][y]
                # wall test value
                if maze_test_value == 0:
                    rect = pygame.Rect(
                        x * self.block_size,
                        y * self.block_size,
                        self.block_size,
                        self.block_size
                    )
                    pygame.draw.rect(self.screen, self.colors.black, rect)

        # note: draw order is important for layering
        # first draw the empty grid first, then the walls and goal, then the player, then the grid overlay last

        # then draw the player
        self.draw_player_and_apple(grid)

        # finally draw the white grid on top of the grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    x * self.block_size,
                    y * self.block_size,
                    self.block_size,
                    self.block_size)
                pygame.draw.rect(self.screen, self.colors.white, rect, 1)

        # self.draw_game_stats(state_string)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                maze_test_value = grid[x][y]
                if maze_test_value == 3:
                    rect = pygame.Rect(
                        x * self.block_size,
                        y * self.block_size,
                        self.block_size,
                        self.block_size
                    )
                    pygame.draw.rect(self.screen, self.colors.purple, rect)

        # update the display
        pygame.display.update()

    def draw_game_stats(self, state_string):
        offset = 525
        new_block_size = 35

        count = 0
        for item in range(1, 13):
            rect = pygame.Rect(
                525,
                new_block_size * item,
                new_block_size,
                new_block_size
            )
            color_choice = self.colors.red

            if state_string[count] == "0":
                color_choice = self.colors.black
            elif state_string[count] == "1":
                color_choice = self.colors.green

            pygame.draw.rect(self.screen, color_choice, rect)

            rect = pygame.Rect(
                525,
                new_block_size * item,
                new_block_size,
                new_block_size
            )

            pygame.draw.rect(self.screen, self.colors.white, rect, 1)
            count += 1

        font = pygame.font.SysFont('arial', 16)

        list_of_strings = ["Apple Down",
                           "Apple Up",
                           "Apple Left",
                           "Apple Right",
                           "Wall Up",
                           "Wall Down",
                           "Wall Left",
                           "Wall Right",
                           "Snake Facing Up",
                           "Snake Facing Down",
                           "Snake Facing Left",
                           "Snake Facing Right"]

        count = 1
        for string in list_of_strings:
            self.screen.blit(font.render(string, True, self.colors.white),
                             (580, new_block_size * count + 5))
            count += 1

    # draw the player object onto the grid
    def draw_player_and_apple(self, grid):
        # find where the player object is a and make a green box
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                maze_test_value = grid[x][y]
                if maze_test_value == 1:
                    rect = pygame.Rect(
                        x * self.block_size,
                        y * self.block_size,
                        self.block_size,
                        self.block_size
                    )
                    pygame.draw.rect(self.screen, self.colors.dark_green, rect)
                elif maze_test_value == 2:
                    rect = pygame.Rect(
                        x * self.block_size,
                        y * self.block_size,
                        self.block_size,
                        self.block_size
                    )
                    pygame.draw.rect(self.screen, self.colors.red, rect)
                elif maze_test_value == 4:
                    rect = pygame.Rect(
                        x * self.block_size,
                        y * self.block_size,
                        self.block_size,
                        self.block_size
                    )
                    pygame.draw.rect(self.screen, self.colors.green , rect)

    # look to see if the space bar is pressed down or not
    def is_space_pressed_down(self):
        if self.q_was_pressed:
            return True
        else:
            button_pressed = False
            while not button_pressed:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            button_pressed = True
                            return True
                        if event.key == pygame.K_q:
                            self.q_was_pressed = True
                            button_pressed = True
                            return True
                        if event.key == pygame.K_x:
                            self.x_was_pressed = True
                            button_pressed = True
                            return False
        return False

    def was_x_pressed_down(self):
        if self.x_was_pressed:
            return True
        else:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_x:
                        self.x_was_pressed = True
                        return True
        return False

    def close_window_and_restart(self):
        pygame.display.quit()
        self.size = 500, 500
        self.colors = Colors()
        self.screen = pygame.display.set_mode(self.size)
        self.screen.fill(self.colors.black)
        pygame.display.set_caption("Shawn Ringler CS4080 RL HW1")
        self.maze = MazeGenerator(5)
        self.maze.change_grid_value(0, 0, 1)  # player
        self.maze.change_grid_value(4, 4, 2)  # goal
        # walls
        self.maze.change_grid_value(1, 2, 3)
        self.maze.change_grid_value(2, 2, 3)
        self.maze.change_grid_value(3, 2, 3)
        self.maze.change_grid_value(4, 2, 3)
        self.maze.change_grid_value(2, 4, 3)
