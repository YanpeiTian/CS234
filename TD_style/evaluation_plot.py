from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet, Net
# from policy_value_net_numpy import PolicyValueNetNumpy as PolicyValueNet
import sys
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch

N=5
SIZE=8
N_GAMES=10
MODEL_1='models/iter_50.model'
MODEL_2='models/iter_50.model'
PLAYOUT=1000
MCTS_PURE=True
HUMAN=False

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)

def policy_evaluate(player1,player2,n_games=N_GAMES):

    win_cnt = defaultdict(int)
    for i in range(n_games):
        board = Board(width=SIZE, height=SIZE, n_in_row=N)
        game = Game(board)
        winner = game.start_play(player1,player2,start_player=i % 2,is_shown=1)
        win_cnt[winner] += 1
    win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games

    return win_ratio

def run():
    n = N
    width, height = SIZE,SIZE

    if MCTS_PURE:
        player_2 = MCTS_Pure(c_puct=5, n_playout=PLAYOUT)
        # print ("Benchmarking the following two models:"+MODEL_1+" Pure MCTS")
    elif HUMAN:
        player_2=Human()
        # print ("Benchmarking the following two models:"+MODEL_1+" Human")
    else:
        pass
        # print ("Benchmarking the following two models:"+MODEL_1+" "+MODEL_2)

    #
    # best_policy_2 = PolicyValueNet(width, height, model_file=MODEL_2)
    # player_2 = MCTSPlayer(best_policy_2.policy_value_fn,
    #                          c_puct=5,
    #                          n_playout=400)  # set larger n_playout for better performance
    # player_1=Human()


    win_ratios = []
    game_batchs = range(50,1501,100)
    for game_batch in game_batchs:
        model = './models/iter_'+str(game_batch)+'.model'
        print(model)

        policy = PolicyValueNet(width, height, model_file=model)
        player_1 = MCTSPlayer(policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance
        win_ratio = policy_evaluate(player_1,player_2)
        win_ratios.append(win_ratio)
        print("The win ratio for "+model+" is: ",str(100*win_ratio)+"%")

    print(zip(win_ratios,game_batchs))

    fig, ax = plt.subplots()
    ax.plot(game_batchs, win_ratios)

    ax.set(xlabel='iterations', ylabel='win ratios',
       title='Win ratio of models trained by 5 input states vs. MCTS player')
    ax.grid()

    fig.savefig("win_ratio.png")

if __name__ == '__main__':
    run()
