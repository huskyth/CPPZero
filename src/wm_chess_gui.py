# -*- coding: utf-8 -*-
import pygame
import os
import numpy as np
import json
from common import ROOT_PATH

BLACK = 1
WHITE = -1
SCREEN_WIDTH = 580
SCREEN_HEIGHT = 580
CHESSMAN_WIDTH = 20
CHESSMAN_HEIGHT = 20
MAPPATH = str(ROOT_PATH / "assets/pointPos.txt")
DISTANCEPATH = str(ROOT_PATH / "assets/distance.txt")


def get_map():
    with open(MAPPATH, 'rb') as f:
        point_pos = json.loads(f.read())
        return point_pos


def get_distance():
    with open(DISTANCEPATH, 'rb') as f:
        distance = json.loads(f.read())
        return distance


GAME_MAP = get_map()
DISTANCE = get_distance()


class WMChessGUI:
    def __init__(self, n, human_color=1, fps=3):

        # screen
        self.board = None
        self.width = 580
        self.height = 580

        self.n = n
        self.fps = fps

        # human color
        self.human_color = human_color

        # reset status
        self.reset_status()

    def __del__(self):
        # close window
        self.is_running = False

    def init_point_status(self):
        self.board = []
        black = [0, 1, 2, 3, 4, 8]
        white = [7, 11, 12, 13, 14, 15]
        for x in range(21):
            self.board.append(0)
        for x in black:
            self.board[x] = BLACK
        for x in white:
            self.board[x] = WHITE
        self.board = np.array(self.board)

    # reset status
    def reset_status(self):
        self.init_point_status()
        self.k = 1  # step number

        self.is_human = False
        self.human_move = -1

    # human play
    def set_is_human(self, value=True):
        self.is_human = value

    def get_is_human(self):
        return self.is_human

    def get_human_move(self):
        return self.human_move

    def get_human_color(self):
        return self.human_color

    # execute move
    def execute_move(self, color, move):
        from_id, to_id = move
        print(f"exec {from_id} to {to_id}")
        assert self.board[from_id] == color
        assert self.board[to_id] == 0
        assert (DISTANCE[from_id][to_id] == 1)
        self.board[from_id] = 0
        self.board[to_id] = color

        self.k += 1

    # main loop
    def loop(self):
        # set running
        self.is_running = True

        # init
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("WmChess")

        # timer
        self.clock = pygame.time.Clock()

        # background image
        base_folder = os.path.dirname(__file__)
        self.background_img = pygame.image.load(
            os.path.join(base_folder, '../assets/watermelon.png')).convert()

        # font
        self.font = pygame.font.SysFont('Arial', 16)

        while self.is_running:
            # timer
            self.clock.tick(self.fps)

            # handle event
            for event in pygame.event.get():
                # close window
                if event.type == pygame.QUIT:
                    self.is_running = False

                # human play
                if self.is_human and event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_y, mouse_x = event.pos
                    position = (int(mouse_x / self.grid_width + 0.5) - 2,
                                int(mouse_y / self.grid_width + 0.5) - 2)

                    if position[0] in range(0, self.n) and position[1] in range(0, self.n) \
                            and self.board[position[0]][position[1]] == 0:
                        self.human_move = position[0] * self.n + position[1]
                        self.execute_move(self.human_color, self.human_move)
                        self.set_is_human(False)

            # draw
            self._draw_background()
            self._draw_chessman()

            # refresh
            pygame.display.flip()

    def _draw_background(self):
        # load background
        self.screen.blit(self.background_img, (0, 0))

    def fix_xy(self, target):
        x = GAME_MAP[target][0] * \
            SCREEN_WIDTH - CHESSMAN_WIDTH * 0.5
        y = GAME_MAP[target][1] * \
            SCREEN_HEIGHT - CHESSMAN_HEIGHT * 1
        return x, y

    def _draw_chessman(self):
        for index, point in enumerate(self.board):
            if point == 0:
                continue
            (x, y) = self.fix_xy(index)
            if point == BLACK:
                pygame.draw.circle(self.screen, (0, 0, 0), (int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
                                   int(CHESSMAN_HEIGHT // 2 * 1.5))
            elif point == WHITE:
                pygame.draw.circle(self.screen, (255, 0, 0),
                                   (int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
                                   int(CHESSMAN_HEIGHT // 2 * 1.5))
