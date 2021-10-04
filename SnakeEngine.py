import sys, random
import numpy as np
import pygame
from DisplayGrid import DisplayGrid as DG


class SnakeEngine:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid_array = np.zeros((grid_size, grid_size))
        self.player_head_pos = (4, 4)
        self.grid_array[self.player_head_pos[0]][self.player_head_pos[1]] = 1
        self.player_pos_list = [self.player_head_pos]
        self.display = DG(grid_size, 50, self.player_head_pos)
        self.apple_spawned = False

    def add_snake_body_to_grid(self):
        for snake_part in self.player_pos_list:
            self.grid_array[snake_part[0]][snake_part[1]] = 1

    def run_game(self):
        while 1:
            if not self.apple_spawned:
                self.spawn_apple_randomly()
            self.display.draw_grid(self.grid_array)
            self.event_handler()

    def event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    self.move_player('right')
                elif event.key == pygame.K_LEFT:
                    self.move_player('left')
                elif event.key == pygame.K_UP:
                    self.move_player('up')
                elif event.key == pygame.K_DOWN:
                    self.move_player('down')

    def remove_tail_from_grid(self, state_tuple):
        self.grid_array[state_tuple[0]][state_tuple[1]] = 0

    def add_tail_onto_grid(self, state_tuple):
        self.grid_array[state_tuple[0]][state_tuple[1]] = 1

    def spawn_apple_randomly(self):
        random_int_x = random.randint(0, 9)
        random_int_y = random.randint(0, 9)
        self.grid_array[random_int_x][random_int_y] = 2
        self.apple_spawned = True

    def move_player(self, action):
        if action == 'right':
            self.move_player_right()
        elif action == 'left':
            self.move_player_left()
        elif action == 'up':
            self.move_player_up()
        elif action == 'down':
            self.move_player_down()

    def move_player_right(self):
        x = self.player_head_pos[0]
        y = self.player_head_pos[1]
        self.remove_tail_from_grid((x,y))
        x += 1
        self.add_tail_onto_grid((x, y))
        self.player_head_pos = (x, y)

    def move_player_left(self):
        x = self.player_head_pos[0]
        y = self.player_head_pos[1]
        self.remove_tail_from_grid((x, y))
        x -= 1
        self.add_tail_onto_grid((x, y))
        self.player_head_pos = (x, y)

    def move_player_up(self):
        x = self.player_head_pos[0]
        y = self.player_head_pos[1]
        self.remove_tail_from_grid((x, y))
        y -= 1
        self.add_tail_onto_grid((x, y))
        self.player_head_pos = (x, y)

    def move_player_down(self):
        x = self.player_head_pos[0]
        y = self.player_head_pos[1]
        self.remove_tail_from_grid((x, y))
        y += 1
        self.add_tail_onto_grid((x, y))
        self.player_head_pos = (x, y)

