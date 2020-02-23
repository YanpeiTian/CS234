from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
# from policy_value_net_pytorch import PolicyValueNet
from policy_value_net_numpy import PolicyValueNetNumpy as PolicyValueNet
import sys
from collections import defaultdict

N=4
SIZE=6
N_GAMES=100

def policy_evaluate(player1,player2,n_games=N_GAMES):

    board = Board(width=SIZE, height=SIZE, n_in_row=N)
    game = Game(board)

    win_cnt = defaultdict(int)
    for i in range(n_games):
        winner = game.start_play(player1,player2,start_player=i % 2,is_shown=0)
        win_cnt[winner] += 1
    win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games

    return win_ratio

def run(model_file_1='best_policy.model',model_file_2='current_policy.model'):
    n = N
    width, height = SIZE,SIZE

    try:
        policy_param_1 = pickle.load(open(model_file_1, 'rb'))
    except:
        policy_param_1 = pickle.load(open(model_file_1, 'rb'),encoding='bytes')  # To support python3

    best_policy_1 = PolicyValueNet(width, height, policy_param_1)
    player_1 = MCTSPlayer(best_policy_1.policy_value_fn,
                             c_puct=5,
                             n_playout=400)  # set larger n_playout for better performance

    try:
        policy_param_2 = pickle.load(open(model_file_2, 'rb'))
    except:
        policy_param_2 = pickle.load(open(model_file_2, 'rb'),encoding='bytes')  # To support python3

    best_policy_2 = PolicyValueNet(width, height, policy_param_2)
    player_2 = MCTSPlayer(best_policy_2.policy_value_fn,
                             c_puct=5,
                             n_playout=400)  # set larger n_playout for better performance

    result=policy_evaluate(player_1,player_2)
    print("The win ratio for "+str(sys.argv[1])+" is %f",result)

if __name__ == '__main__':
    try:
        print ("Benchmarking the following two models:"+str(sys.argv[1])+" "+str(sys.argv[2]))
    except:
        print("Usage: Please give two model to benchmark.")
    run(sys.argv[1],sys.argv[2])
