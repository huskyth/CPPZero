import copy
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from config import config
from src.neural_network import NeuralNetWorkWrapper
from src.wm_chess_gui import WMChessGUI, BLACK, WHITE, CHESSMAN_WIDTH, CHESSMAN_HEIGHT

BLACK_COLOR = (0, 0, 0)
from common import write_image, ROOT_PATH, read_image, ANALYSIS_PATH, create_directory, from_array_to_input_tensor, \
    MOVE_LIST

BACKGROUND = ROOT_PATH / "assets/watermelon.png"


class AnalysisTool:
    def __init__(self):
        self.board = None
        self.nnet = NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'],
                                         config['num_channels'], config['n'], 72, False, False)
        self.stamp = time.time()
        self.work_path = ANALYSIS_PATH / str(self.stamp)
        create_directory(self.work_path)
        self.current_step = None

    @staticmethod
    def draw_circle(image, x, y, color):
        cv2.circle(image, (int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)), int(CHESSMAN_HEIGHT // 2 * 1.5),
                   color, -1)

    def draw_chessmen(self, board, image, is_write, name):
        image = copy.deepcopy(image)
        for index, point in enumerate(board):
            (x, y) = WMChessGUI.fix_xy(index)
            if point == BLACK:
                AnalysisTool.draw_circle(image, x, y, BLACK_COLOR)
            elif point == WHITE:
                AnalysisTool.draw_circle(image, x, y, (255, 0, 0))
            cv2.putText(image, str(index), (int(x - 1), int(y - 1)), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 5)

        if is_write:
            write_image(self.work_path / name, image)
        return image

    def _set_origin_board(self, board):
        self.board = np.array(board)
        assert len(board) == 21

    def _display(self):
        image = read_image(BACKGROUND)

        self.draw_chessmen(self.board, image, True, f"image_{self.current_step}")

    def _load_model(self):
        self.nnet.load_model()

    def _bin_display(self, y):
        max_y = np.max(y)
        x = [str(x) for x in MOVE_LIST]
        assert len(y) == len(x)
        plt.bar(x, y, width=0.5, color='C0', edgecolor='grey')
        plt.xticks(rotation=80, fontsize=4)

        plt.ylim(0, max_y)

        plt.tight_layout()
        plt.savefig(str(self.work_path / f'bar_{self.current_step}.png'), dpi=600)
        plt.show()

    def _infer(self, last_move, current_player, policy=None, value=None):
        assert policy is None and value is None or policy is not None and value is not None
        if policy is None and value is None:
            input = from_array_to_input_tensor(self.board)
            p, v = self.nnet.infer([(input, last_move, current_player)])
        else:
            p, v = policy, value
        with open(str(self.work_path / f"value_{self.current_step}.txt"), 'w') as f:
            f.write(f"value is {v}")
        self._bin_display(p.squeeze(0))

    def analysis(self, board, last_move, current_player, current_step, policy=None, value=None):
        self._set_origin_board(board)
        self.current_step = current_step
        # self._load_model()
        self._infer(last_move, current_player, policy, value)
        self._display()


def init_board():
    board = []
    black = [0, 1, 2, 3, 4, 8]
    white = [7, 11, 12, 13, 14, 15]
    for x in range(21):
        board.append(0)
    for x in black:
        board[x] = BLACK
    for x in white:
        board[x] = WHITE
    return board


if __name__ == '__main__':
    at = AnalysisTool()
    input_board = init_board()
    cp_b = copy.deepcopy(input_board)
    at.analysis(input_board, (0, 1), 1, 0)
    print(input_board == cp_b)
