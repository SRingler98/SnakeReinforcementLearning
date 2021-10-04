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


# display grid class displays the grid of the grid to the screen
class DisplayGrid:
    # DisplayGrid uses the pygame library to display the grid onto the screen
    def __init__(self, size_in, block_size):
        self.block_size = block_size
        self.grid_size = size_in
        pygame.init()
        self.size = 500, 500
        self.colors = Colors()
        self.screen = pygame.display.set_mode(self.size)
        self.screen.fill(self.colors.black)
        pygame.display.set_caption("Snake Reinforcement Learning")
        self.maze = MazeGenerator(self.grid_size)
        self.maze.change_grid_value(4, 4, 1)  # player
        # walls
        self.q_was_pressed = False
        self.x_was_pressed = False

    def draw_grid(self):
        # draw the empty grid first
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                maze_test_value = self.maze.get_grid_value(x, y)
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
        self.draw_player()

        # finally draw the white grid on top of the grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    x * self.block_size,
                    y * self.block_size,
                    self.block_size,
                    self.block_size)
                pygame.draw.rect(self.screen, self.colors.white, rect, 1)

        # update the display
        pygame.display.update()

    # draw the player object onto the grid
    def draw_player(self):
        # find where the player object is a and make a green box
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                maze_test_value = self.maze.get_grid_value(x, y)
                if maze_test_value == 1:
                    rect = pygame.Rect(
                        x * self.block_size,
                        y * self.block_size,
                        self.block_size,
                        self.block_size
                    )
                    pygame.draw.rect(self.screen, self.colors.green, rect)

    # update where the player is on the grid
    def update_player_state(self, state_tuple):
        self.maze.change_player_position(state_tuple)

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

    def event_handler(self, state_tuple):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    x = state_tuple[0] + 1
                    y = state_tuple[1]
                    self.update_player_state((x, y))
                    state_tuple = (x, y)


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
