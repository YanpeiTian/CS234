from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet, Net
# from policy_value_net_numpy import PolicyValueNetNumpy as PolicyValueNet
import sys
from collections import defaultdict
from human_play import Human
import torch

N=4
SIZE=6
N_GAMES=3
MODEL_1='current_policy.model'
MODEL_2='current_policy.model'

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
    print ("Benchmarking the following two models:"+MODEL_1+" "+MODEL_2)

    n = N
    width, height = SIZE,SIZE


    best_policy_1 = PolicyValueNet(width, height, model_file=MODEL_2)
    player_1 = MCTSPlayer(best_policy_1.policy_value_fn,
                             c_puct=5,
                             n_playout=400)  # set larger n_playout for better performance

    best_policy_2 = PolicyValueNet(width, height, model_file=MODEL_2)

    player_2 = MCTSPlayer(best_policy_2.policy_value_fn,
                             c_puct=5,
                             n_playout=400)  # set larger n_playout for better performance


    mcts_player = MCTS_Pure(c_puct=5, n_playout=400)
    human=Human()

    result=policy_evaluate(mcts_player,player_2)
    print("The win ratio for "+MODEL_1+" is: ",str(100*result)+"%")



if __name__ == '__main__':

    run()
