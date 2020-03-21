from __future__ import print_function
import numpy as np
from time import sleep
from collections import deque
import copy

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.max_state_representation_layer = 51
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player

        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states_buffer = deque(maxlen = self.max_state_representation_layer -1)
        for i in range(self.max_state_representation_layer -1):
            # board states stored as a dict,
            # key: move as location on the board,
            # value: player as pieces type
            state = {}
            self.states_buffer.append(state)

        assert (len(self.states_buffer)==self.max_state_representation_layer -1)
        # print('states_buffer should all be empty dicts: ',self.states_buffer)
        # print('states_buffer length: ',len(self.states_buffer))

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self,state_representation_channel):
        """return the board state from the perspective of the current player.
        state shape: state_representation_channel * width * height
        """
        square_state = np.zeros((state_representation_channel, self.width, self.height))
        assert (state_representation_channel % 2 == 1)
        each_player_layers = (state_representation_channel-1)/2

        black_layer = 0
        for i in range(1,1+int(2*each_player_layers),2):
            # print('i = ',i)
            state = self.states_buffer[-i]
            if state:

                white_layer = int(each_player_layers + black_layer)
                moves, players = np.array(list(zip(*state.items())))
                move_1 = moves[players == self.players[0]]
                move_2 = moves[players == self.players[1]]
                square_state[black_layer][move_1 // self.width,move_1 % self.height] = 1.0
                square_state[white_layer][move_2 // self.width,move_2 % self.height] = 1.0

                black_layer+=1
            else:
                break

        if self.current_player==self.players[1]:
            square_state[state_representation_channel - 1][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        #rightmost--> most current state of board
        #leftmost --> oldest item state of board
        top = copy.deepcopy(self.states_buffer[-1])
        top[move] = self.current_player
        self.states_buffer.append(top)

        # print(self.states_buffer)

        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )

    def has_a_winner(self):
        width = self.width
        height = self.height
        # states = self.states
        states = self.states_buffer[-1]
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board
        self.state_representation_channel = int(kwargs.get('state_representation_channel', 11))

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states_buffer[-1].get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()

            # print(self.board.current_state(state_representation_channel = 11))

            if end:
                if winner != -1:
                    print("Game end. Winner is", players[winner])
                else:
                    print("Game end. Tie")
                return winner

    # def start_self_play(self, player, is_shown=0, temp=1e-3):
    #     """ start a self-play game using a MCTS player, reuse the search tree,
    #     and store the self-play data: (state, mcts_probs, z) for training
    #     """
    #     self.board.init_board()
    #     p1, p2 = self.board.players
    #     states, mcts_probs, current_players = [], [], []
    #     while True:
    #         move, move_probs = player.get_action(self.board,
    #                                              temp=temp,
    #                                              return_prob=1)
    #         # store the data
    #         states.append(self.board.current_state(self.state_representation_channel))
    #         mcts_probs.append(move_probs)
    #         current_players.append(self.board.current_player)
    #         # perform a move
    #         self.board.do_move(move)
    #         if is_shown:
    #             self.graphic(self.board, p1, p2)
    #         end, winner = self.board.game_end()
    #         if end:
    #             # winner from the perspective of the current player of each state
    #             winners_z = np.zeros(len(current_players))
    #             if winner != -1:
    #                 winners_z[np.array(current_players) == winner] = 1.0
    #                 winners_z[np.array(current_players) != winner] = -1.0
    #             # reset MCTS root node
    #             player.reset_player()
    #             if is_shown:
    #                 if winner != -1:
    #                     print("Game end. Winner is player:", winner)
    #                 else:
    #                     print("Game end. Tie")
    #             return winner, zip(states, mcts_probs, winners_z)

    def start_self_play_TD(self, player, policy_value_net, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        step=0
        while True:
            step+=1
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state(self.state_representation_channel))
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

            # predict_value = policy_value_net.policy_value_fn(self.board)[-1].item()
            # if step > (self.board.width * self.board.height)/2 and (predict_value>0.99 or predict_value<-0.99):
            # # if predict_value>0.99 or predict_value<-0.99:
            #
            #     print("Predicted value: "+str(predict_value))
            #
            #     # winner from the perspective of the current player of each state
            #     winners_z = np.zeros(len(current_players))
            #     winners_z[np.array(current_players) == self.board.current_player] = -predict_value
            #     winners_z[np.array(current_players) != self.board.current_player] = predict_value
            #
            #     # reset MCTS root node
            #     player.reset_player()
            #
            #     return -1, zip(states, mcts_probs, winners_z)
