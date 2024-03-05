# coding: utf-8
import sys

from common import ROOT_PATH

sys.path.append('..')
sys.path.append(str(ROOT_PATH / 'src'))
sys.path.append(str(ROOT_PATH / "build"))

import learner
import config

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ["train", "play"]:
        print("[USAGE] python leaner_test.py train|play")
        exit(1)

    alpha_zero = learner.Leaner(config.config)

    if sys.argv[1] == "train":
        alpha_zero.learn()
    elif sys.argv[1] == "play":
        for i in range(10):
            print("GAME: {}".format(i + 1))
            alpha_zero.play_with_human(human_first=i % 2)
    elif sys.argv[1] == "self-play":
        alpha_zero.only_self_play()
