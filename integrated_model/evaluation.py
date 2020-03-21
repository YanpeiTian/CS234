from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet, Net
# from policy_value_net_numpy import PolicyValueNetNumpy as PolicyValueNet
import sys
from collections import defaultdict

import torch

N=5
SIZE=8
N_GAMES=1
MODEL_1='best.model'
# MODEL_2='../starter/models_original_2_24/best.model'
PLAYOUT=1000
MCTS_PURE=True
HUMAN=True

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

    # if MCTS_PURE:
    #     player_1 = MCTS_Pure(c_puct=5, n_playout=PLAYOUT)
    #     print ("Benchmarking the following two models:"+MODEL_1+" Pure MCTS")
    # elif HUMAN:
    #     player_2=Human()
    #     print ("Benchmarking the following two models:"+MODEL_1+" Human")
    # else:
    #     print ("Benchmarking the following two models:"+MODEL_1+"  vs  "+MODEL_2)
    #     policy_2= PolicyValueNet(width, height, model_file=MODEL_2,state_representation_channel = 4)
    #     player_2 = MCTSPlayer(policy_2.policy_value_fn,c_puct=5,n_playout=400)  # set larger n_playout for better performance


    #
    policy_1= PolicyValueNet(width, height, model_file=MODEL_1,in_channel = 11,n_resnet=1)
    player_1 = MCTSPlayer(policy_1.policy_value_fn,
                             c_puct=5,
                             n_playout=400)  # set larger n_playout for better performance

    # player_1 = Human()
    player_2 = Human()


    win_ratio = policy_evaluate(player_1,player_2)
    print("The win ratio for "+MODEL_1+" is: ",str(100*win_ratio)+"%")


if __name__ == '__main__':
    run()
