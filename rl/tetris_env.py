import numpy as np
import copy

from game import TetrisEngine
from gym import spaces


piece_map = {
    "I": 1,  # Example: I-piece -> 1
    "O": 2,  # Example: O-piece -> 2
    "T": 3,  # Example: T-piece -> 3
    "L": 4,  # Example: L-piece -> 4
    "J": 5,  # Example: J-piece -> 5
    "S": 6,  # Example: S-piece -> 6
    "Z": 7,  # Example: Z-piece -> 7
    0: 0   # Empty space -> 0
}

class TetrisEnv():
    def __init__(self):
        self.engine = TetrisEngine.GameState()
        #self.action_map = ['L', 'R', 'd', 'D', 'r', 'C']
        #self.action_space = spaces.Discrete(len(self.action_map))

        # PAIR (rotations (%4), end_column)
        self.high_level_actions = [(rot, c) for rot in range(4) for c in range(self.engine.cols)]
        self.high_level_actions.append(('C', None))
        self.action_space = spaces.Discrete(len(self.high_level_actions))

    def reset_game(self):
        self.engine.reset_game()
        return self.get_state()

    def get_state(self):
        board = np.array([[piece_map[cell] for cell in row] for row in self.engine.board]).flatten()

        #piece = self.engine.currentPiece
        current_piece = piece_map[self.engine.currentPiece.type]

        next_pieces = [piece_map[piece.type] for piece in self.engine.nextPieces]
        hold_piece = 0 if not self.engine.holdPiece else piece_map[self.engine.holdPiece.type]

        #current_score = self.engine.score

        state = np.concatenate([board, [current_piece], next_pieces, [hold_piece]]).astype(np.float32)
        return state
    
    def step(self, action, gpx):
        # action_str = self.action_map[action]
        lines_cleared = 0

        prev_board = copy.deepcopy(self.engine.board)

        for r,c in self.engine.currentPiece.get_cells():
            prev_board[r][c] = 0

        rotation, target_col = self.high_level_actions[action]

        if rotation == 'C':
            self.engine.hold_Piece()
            reward = 0 # 0.1
        else:
            lines_cleared = self.engine.place_piece(rotation, target_col, gpx)

            # reward, done = self.engine.perform_action(action_str, prev_board)
            reward = self.engine.calculate_reward(lines_cleared, prev_board)
            #reward = self.engine.calculate_reward()

        done = self.engine.game_ended

        next_state = self.get_state()

        return next_state, reward, done, {'lines_cleared': lines_cleared}