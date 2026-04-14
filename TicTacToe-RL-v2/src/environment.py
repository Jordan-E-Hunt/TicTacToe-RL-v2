import numpy as np


class TicTacToe:
    def __init__(self):
        self.size = 3
        self.board = np.zeros((self.size, self.size))
        self.players = ["X", "O"]
        self.current = self.players[0]
        self.winner = None
        self.game_over = False
        self.last_rotation = 0
        self.check_winner()

    def reset(self):
        self.__init__()

    def avail_moves(self):
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def get_state(self):
        p_idx = self.players.index(self.current) + 1
        norm_board = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    norm_board[i][j] = 0
                elif self.board[i][j] == p_idx:
                    norm_board[i][j] = 1
                else:
                    norm_board[i][j] = -1
        canonical, self.last_rotation = self.canonical_board(norm_board)
        return canonical

    def make_move(self, move):
        reward = -0.1
        if self.board[move[0]][move[1]] != 0:
            return self.get_state(), -1.0, True, False
        self.board[move[0]][move[1]] = self.players.index(self.current) + 1
        self.check_winner()
        if self.game_over:
            if self.winner == self.current:
                reward = 3.0
            elif self.winner == "Draw":
                reward = 0.3
            else:
                reward = -2.0
        self.switch_player()
        return self.get_state(), reward, self.game_over, False

    def switch_player(self):
        self.current = self.players[1] if self.current == self.players[0] else self.players[0]

    def check_winner(self):
        for i in range(self.size):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                self.winner = self.players[int(self.board[i][1] - 1)]
                self.game_over = True
        for j in range(self.size):
            if self.board[0][j] == self.board[1][j] == self.board[2][j] != 0:
                self.winner = self.players[int(self.board[1][j] - 1)]
                self.game_over = True
        if (self.board[0][0] == self.board[1][1] == self.board[2][2] != 0 or
                self.board[0][2] == self.board[1][1] == self.board[2][0] != 0):
            self.winner = self.players[int(self.board[1][1] - 1)]
            self.game_over = True
        if np.all(self.board != 0) and not self.game_over:
            self.winner = "Draw"
            self.game_over = True

    def print_board(self):
        print("--------------")
        for i in range(self.size):
            print("|", end=" ")
            for j in range(self.size):
                print(self.players[int(self.board[i][j] - 1)]
                      if self.board[i][j] != 0 else " ", end=" | ")
            print()
            print("--------------")

    def canonical_board(self, board):
        """Return the lex-smallest tuple among all rotational symmetries."""
        candidates = []
        current = board
        for r in range(4):
            candidates.append(tuple(current.flatten()))
            current = np.rot90(current)
        if self.size > 3:
            candidates.append(tuple(np.fliplr(board).flatten()))
            candidates.append(tuple(np.flipud(board).flatten()))
            candidates.append(tuple(np.transpose(board).flatten()))
            candidates.append(tuple(np.fliplr(np.flipud(board)).flatten()))
        min_idx = candidates.index(min(candidates))
        return min(candidates), min_idx

    def rotate_action(self, action, rotation):
        """Rotate an action into canonical space."""
        r, c = action
        for _ in range(rotation):
            r, c = 2 - c, r
        return (r, c)

    def unrotate_action(self, action, rotation):
        """Inverse-rotate an action back to real board space."""
        r, c = action
        for _ in range(rotation):
            r, c = c, 2 - r
        return (r, c)
