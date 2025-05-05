import time
import random
import copy
#import numpy as np

from .Piece import Piece

class GameState(): #10x20
    def __init__(self):
        self.board = [ #Tablero / grid [[0]*10 for _ in range(20)]
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
        ]
        self.prev_board = None
        self.rows = 20
        self.cols = 10
        self.log = [] # For training purposes log[-1]
        self.images = ['I', 'O', 'T', 'L', 'J', 'S', 'Z']
        self.currentPiece = None
        self.projected_coords = []

        # Piece placing criteria
        self.lock_delay = 0
        self.LOCK_LIMIT = 3

        # Flags
        self.score = 0
        self.last_move_time = time.time()
        self.hold_used = False
        self.is_paused = False
        self.game_ended = False
        self.AI_playing = False
        self.move_time = time.time()
        
        # Next Pieces
        self.nextPieces = [] # 'K' 'L' 'O'
        self.nextPiecesGrid = [
            [0,0,0,0], # Piece 1
            [0,0,0,0],
            [0,0,0,0], 
            [0,0,0,0], # Piece 2
            [0,0,0,0],
            [0,0,0,0], 
            [0,0,0,0], # Piece 3
            [0,0,0,0],
            [0,0,0,0]
        ]

        # Hold piece
        self.holdPiece = None
        self.holdPieceGrid = [
            [0,0,0,0],
            [0,0,0,0]
        ]

        # Difficulty increment
        self.SPEED_FACTOR = 1

        self.spawnPieces()

    # DQN SETUP
    def reset_game(self):
        # Reset the board and game over flag
        self.prev_board = None
        self.board = self.initialize_board()
        self.currentPiece = Piece('O')
        self.projected_coords = []
        self.score = 0
        self.drop_height = 0
        self.last_move_time = time.time()
        self.hold_used = False
        self.is_paused = False
        self.game_ended = False
        self.AI_playing = False

        # Next Pieces
        self.nextPieces = [] # 'K' 'L' 'O'
        self.nextPiecesGrid = [[0] * 4 for _ in range(9)]

        # Hold piece
        self.holdPiece = None
        self.holdPieceGrid = [
            [0,0,0,0],
            [0,0,0,0]
        ]
        self.update()
        self.updateHoldGrid()
        self.spawnPieces()

        self.game_ended = False

    def initialize_board(self):
        # Logic to initialize a new game board
        return [[0] * 10 for _ in range(20)]  # Example: 20 rows, 10 columns
    
    def get_board(self):
        # Return the current board
        return self.board
    
    def perform_action(self, action, board):
        # Implement your action logic here (e.g., move left, rotate, etc.)
        if action == 'L':
            self.moveLeft()
        elif action == 'R':
            self.moveRight()
        elif action == 'd':
            self.moveDown()
        elif action == 'D':
            self.dropPiece()
        elif action == 'r':
            self.rotatePiece()
        elif action == 'C':
            self.hold_Piece()
        
        # Calculate reward and done flag
        reward = self.calculate_reward(board)  # Example: a function that calculates the reward
        done = self.check_game_over()  # Check if the game is over

        return reward, done
    
    def place_piece(self, rotation, col):
        for _ in range(rotation % 4):
            self.rotatePiece()
        while self.currentPiece.col < col:
            if self.moveRight() == False: break
        while self.currentPiece.col > col:
            if self.moveLeft() == False: break
        self.dropPiece()
    
    # def calculate_reward(self):
    #     # Example logic for reward: +1 for clearing a line, etc.
    #     #return 1 if self.check_line_clear() else 0
    #     lines_cleared = self.check_line_clear()
    #     if lines_cleared > 0:
    #         return 100 * lines_cleared  # big positive reward
    #     elif self.game_ended:
    #         return -100  # strong penalty for losing
    #     else:
    #         return -1

    # def calculate_reward(self, lines_cleared, previous_height, new_height, holes_created):
    #     reward = 0
    #     reward += lines_cleared * 10
    #     if lines_cleared > 0:
    #         reward += 5  # bonus for clearing at all
    #     reward -= (new_height - previous_height) * 0.5  # discourage stacking up
    #     reward -= holes_created * 2  # heavily penalize holes
    #     if self.game_ended:
    #         reward -= 50
    #     return reward
    
    def calculate_reward(self, prev_board):
        lines_cleared = self.check_line_clear()
        new_board = self.board

        prev_height = self.get_aggregate_height(prev_board) if prev_board is not None else 0
        new_height = self.get_aggregate_height(new_board)
        height_penalty = new_height - prev_height

        prev_holes = self.get_holes(prev_board) if prev_board is not None else 0
        new_holes = self.get_holes(new_board)
        hole_penalty = new_holes - prev_holes

        balance_penalty = self.balance_penalty()
        center_alignment = self.center_alignment_bonus()

        bumpiness = self.get_bumpiness()

        reward = 0
        #reward += [0, 10, 30, 60, 100][lines_cleared]
        reward += lines_cleared * 20
        reward += self.drop_height * 0.5
        reward -= height_penalty * 3
        reward -= hole_penalty * 2
        reward -= bumpiness * 0.3
        reward += balance_penalty
        reward += center_alignment

        if self.game_ended:
            reward -= 50

        #reward = max(-1.0, min(1.0, reward / 10.0))
        reward = max(-50, min(50, reward))

        return reward

    def get_aggregate_height(self, board):
        heights = []
        for col in zip(*board):  # transpose to iterate columns
            height = 0
            for i, cell in enumerate(col):
                if cell != 0:
                    height = len(col) - i
                    break
            heights.append(height)
        return sum(heights)
    
    def get_holes(self, board):
        holes = 0
        for col in zip(*board):
            block_found = False
            for cell in col:
                if cell != 0:
                    block_found = True
                elif block_found and cell == 0:
                    holes += 1
        return holes

    def get_bumpiness(self):
        heights = []
        for col in zip(*self.board):
            height = 0
            for i, cell in enumerate(col):
                if cell != 0:
                    height = len(col) - i
                    break
            heights.append(height)
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
        return bumpiness

    def check_game_over(self):
        # Check if the game is over
        return self.game_ended
    
    def check_line_clear(self):
        # curr_score = self.score
        # self.clearFullRows()
        # if curr_score != self.score:
        #     return True
        # return False
        # count how many lines you cleared this step:
        lines_cleared = self.clearFullRows(return_count=True)
        if lines_cleared > 0:
            print("*****\nLINES CLEARED\n*****\a")
            time.sleep(2)
        return lines_cleared
    
    def balance_penalty(self):
        column_heights = self.get_column_heights()  # Call once and reuse

        # Split the heights into left and right halves
        left_heights = column_heights[:self.cols // 2]  # First half
        right_heights = column_heights[self.cols // 2:]  # Second half
        
        left_avg = sum(left_heights) / len(left_heights)
        right_avg = sum(right_heights) / len(right_heights)

        # Penalize if one side is significantly higher than the other
        balance_diff = abs(left_avg - right_avg)
        # print(f"Left avg: {left_avg:.2f}, Right avg: {right_avg:.2f}, Penalty: {-(right_avg - left_avg) * 10:.2f}")
        # if right_avg > left_avg:
        #     return -(right_avg - left_avg) * 10
        # else:
        #     return (right_avg - left_avg) * 10
        return -balance_diff * 0.5  # Tune multiplier as needed
    
    def center_alignment_bonus(self):
        heights = self.get_column_heights()
        center = self.cols // 2
        weighted_sum = sum(h * abs(center - i) for i, h in enumerate(heights))
        return -weighted_sum * 0.05  # penalize distance from center

    def get_column_heights(self):
        # Initialize a list to store the heights of each column
        heights = [0] * self.cols

        # Iterate over each column
        for col in range(self.cols):
            # Iterate over the rows in this column, starting from the bottom
            for row in range(self.rows - 1, -1, -1):
                if self.board[row][col] != 0:  # If the cell is filled
                    heights[col] = self.rows - row  # Height is distance from the bottom
                    break  # No need to check above this row, we've found the height

        return heights

    # GAME LOGIC
    def spawnPieces(self):
        while len(self.nextPieces) < 4:
            new_piece = random.choice(self.images)
            self.nextPieces.append(Piece(new_piece))

        self.currentPiece = self.nextPieces.pop(0) # 'K' 'L' 'O'
        
        # Debugging
        # print(self.currentPiece.type, end = " // ")
        # print([k.type for k in self.nextPieces])
        if any(self.board[r][c] != 0 for r,c in self.currentPiece.get_cells()):
            print("Game Over!")
            self.game_ended = True # pygame.quit()
            return

        for r, c in self.currentPiece.get_cells():
            self.board[r][c] = self.currentPiece.type

        self.init_time = time.time()
        self.init_board = self.board

        # Spawn next pieces
        # Clear preview grid
        self.nextPiecesGrid = [[0] * 4 for _ in range(9)]  # Now 9 rows for spacing

        # Spawn next pieces in the preview grid
        for i, piece in enumerate(self.nextPieces[:3]):  # Only display the next 3 pieces
            type = piece.type
            s_r = 3 * i # Each piece starts at row `3*i`, column 1 for centering
            s_c = 0

            if type in {'I', 'J', 'Z'}:
                s_c = 0  # I-piece is wider, so shift left
            elif type in {'O', 'T'}:
                s_c = 1  # Default center for 3-wide pieces
            else:  # L and J
                s_c = 2 

            # Adjust the shape to fit within the section
            pseudo_shape = [(r + s_r, c + s_c) for r, c in piece.shape]

            # Ensure the piece stays within bounds
            for r, c in pseudo_shape:
                if 0 <= r < 9 and 0 <= c < 4:
                    self.nextPiecesGrid[r][c] = type

        # Debugging output
        # print([p.type for p in self.nextPieces])
        # for row in self.nextPiecesGrid:
        #     print(row)
        self.getProjection()

    def update(self):
        current_positions = self.currentPiece.get_cells()
        new_positions = [(r + 1, c) for r, c in current_positions]
        
        self.getProjection()

        if self.game_ended:
            return

        last_speed = self.SPEED_FACTOR
        self.SPEED_FACTOR = 1 - 0.1*(self.score // 1000)

        if self.SPEED_FACTOR <= 0.10: self.SPEED_FACTOR = 0.10
        if last_speed != self.SPEED_FACTOR: print(f"New gravity speed: {self.SPEED_FACTOR}")

        if time.time() - self.last_move_time > 1 * self.SPEED_FACTOR:
            self.last_move_time = time.time()

            if all((r < self.rows and self.board[r][c] == 0) or ((r,c) in current_positions) for r, c in new_positions): 
                # Clear old position
                for r, c in current_positions:
                    self.board[r][c] = 0
                                
                # Move piece down
                self.currentPiece.row += 1 
                self.lock_delay = 1
                
                # Place at new position
                for r, c in self.currentPiece.get_cells():
                    self.board[r][c] = self.currentPiece.type  
            else:
                self.lock_delay += 1
                # Lock the piece and spawn a new one
                # print("Locking piece")
                if self.lock_delay >= self.LOCK_LIMIT:
                    self.placePiece()

    def placePiece(self):
        for r, c in self.currentPiece.get_cells():
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.board[r][c] = self.currentPiece.type

        # Spawn a new piece
        self.drop_height = 0
        self.clearFullRows()
        self.spawnPieces()
        self.log.clear()
        self.hold_used = False

    def getProjection(self):
        # Clone the current piece to avoid modifying the original
        piece_copy = Piece(self.currentPiece.type)
        piece_copy.row = self.currentPiece.row
        piece_copy.col = self.currentPiece.col
        piece_copy.shape = self.currentPiece.shape

        while True:
            # Calculate new projected positions
            new_positions = [(r + 1, c) for r, c in piece_copy.get_cells()]

            # Check if moving down is possible
            if all((r < self.rows and self.board[r][c] == 0)  or ((r, c) in piece_copy.get_cells()) for r, c in new_positions):
                piece_copy.row += 1  # Move the copied piece down
            else:
                break  # Stop when it collides

        # Store the final projected coordinates
        piece_copy_cells = set(piece_copy.get_cells())
        original_piece_cells = set(self.currentPiece.get_cells())

        # Find the difference
        self.projected_coords = list(piece_copy_cells - original_piece_cells)

    def clearFullRows(self, return_count = False):
        count = 0
        for row in range(self.rows):
            if all(self.board[row][col] != 0 for col in range(self.cols)):  # Row is full
                self.board.pop(row)  # Remove the full row
                self.board.insert(0, [0] * self.cols)  # Insert an empty row at the top
                count += 1
        # LINES TO UNCOMMENT
        # if count == 4:
        #     TetrisSFX.play()
        # elif count >= 1:
        #     ClearSFX.play()
        self.score += 100*count

        if return_count:
            return count

    def rotatePiece(self):
        new_shape = []
        new_positions = []
        #print(self.currentPiece.cells, "**************")

        pivot = self.currentPiece.shape[2]
        
        if self.currentPiece.type == 'O':
            return
        elif self.currentPiece.type == 'I':
            if self.currentPiece.shape == [(0,0), (0,1), (0,2), (0,3)]:  # Horizontal → Vertical
                new_shape = [(-1,2), (0,2), (1,2), (2,2)]
            elif self.currentPiece.shape == [(-1,2), (0,2), (1,2), (2,2)]:  # Vertical → 180°
                new_shape = [(1,0), (1,1), (1,2), (1,3)]
            elif self.currentPiece.shape == [(1,0), (1,1), (1,2), (1,3)]:  # 180° → 270°
                new_shape = [(-1,1), (0,1), (1,1), (2,1)]
            elif self.currentPiece.shape == [(-1,1), (0,1), (1,1), (2,1)]:  # 270° → Back to horizontal
                new_shape = [(0,0), (0,1), (0,2), (0,3)]
        else:
            new_shape = [((c - pivot[1]) + pivot[0], -(r - pivot[0]) + pivot[1]) 
                     for r, c in self.currentPiece.shape]
        
        new_positions = self.currentPiece.get_cells(new_shape)

        if self.validate_rotation(new_positions):
            # LINES TO UNCOMMENT
            #RotationSFX.play()
            self.currentPiece.cells = new_positions
            self.currentPiece.shape = new_shape
            self.log.append('r') # as R is for Right
    
    def validate_rotation(self, new_pos):
        current_positions = self.currentPiece.get_cells()

        # Check if the new positions are valid: within bounds and not colliding
        for r, c in new_pos:
            # Check if the position is out of bounds
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                # print("Rotation out of bounds.")
                return False  # Rotation is not valid, out of bounds
            
            # Check if the position is occupied by another piece (not the current piece's cells)
            if self.board[r][c] != 0 and (r, c) not in current_positions:
                # print("Spot occupied by another piece.")
                return False  # Rotation is not valid, occupied spot

        # Clear the current piece's old positions on the board
        for r, c in current_positions:
            self.board[r][c] = 0

        # print("Rotating piece")

        # Update the board with the new rotated positions
        self.currentPiece.cells = new_pos
        
        for r, c in new_pos:
            self.board[r][c] = self.currentPiece.type
        return True

    def moveLeft(self):
        current_positions = self.currentPiece.get_cells()
        new_positions = [(r, c - 1) for r, c in current_positions]

        if all((c >= 0 and self.board[r][c] == 0) or ((r,c) in current_positions) for r, c in new_positions): 
            # Clear old position
            for r, c in current_positions:
                self.board[r][c] = 0
            
            # print("Moving piece left")
            
            # Move piece down
            self.currentPiece.col -= 1  
            
            # Place at new position
            for r, c in self.currentPiece.get_cells():
                self.board[r][c] = self.currentPiece.type

            self.log.append('L')
        else:
            # Lock the piece and spawn a new one
            for r, c in current_positions:
                self.board[r][c] = self.currentPiece.type
            return False

    def moveRight(self):
        current_positions = self.currentPiece.get_cells()
        new_positions = [(r, c + 1) for r, c in current_positions]

        if all((c < self.cols  and self.board[r][c] == 0) or ((r,c) in current_positions) for r, c in new_positions): 
            # Clear old position
            for r, c in current_positions:
                self.board[r][c] = 0
            
            # print("Moving piece right")
            
            # Move piece down
            self.currentPiece.col += 1  
            
            # Place at new position
            for r, c in self.currentPiece.get_cells():
                self.board[r][c] = self.currentPiece.type

            self.log.append('R')
        else:
            # Lock the piece and spawn a new one
            for r, c in current_positions:
                self.board[r][c] = self.currentPiece.type
            return False

    def moveDown(self):
        current_positions = self.currentPiece.get_cells()
        new_positions = [(r + 1, c) for r, c in current_positions]

        if all((r < self.rows and self.board[r][c] == 0) or ((r,c) in current_positions) for r, c in new_positions): 
            # Clear old position
            for r, c in current_positions:
                self.board[r][c] = 0
            
            # print("Moving piece down")
            
            # Move piece down
            self.currentPiece.row += 1  
            
            # Place at new position
            for r, c in self.currentPiece.get_cells():
                self.board[r][c] = self.currentPiece.type  

            self.log.append('d') # since D is for Drop
            self.score += 1
        else:
            # Lock the piece and spawn a new one
            # print("Locking piece")
            self.placePiece()

    def hold_Piece(self):
        if self.hold_used:
            # Can't change twice in a row with the same piece
            return

        for r,c in self.currentPiece.get_cells():
            self.board[r][c] = 0

        # If there's no held piece, store the current one and spawn a new piece
        if self.holdPiece is None:
            self.holdPiece = self.currentPiece
            self.spawnPieces()  # Spawn new piece after first hold
        else:
            # for r, c in self.holdPiece.get_cells():
            #     self.holdPieceGrid[r][c] = 0
            # Swap current piece with the held one
            self.currentPiece, self.holdPiece = self.holdPiece, self.currentPiece
            self.spawnHoldPiece()

        # Update the hold piece preview
        self.updateHoldGrid()
        self.log.append('C')
        self.hold_used = True

    def updateHoldGrid(self):
        """ Updates the hold piece grid for previewing the held piece. """
        self.holdPieceGrid = [[0] * 4 for _ in range(2)]  # 2-row display

        if not self.holdPiece:
            return  # No piece held yet

        type = self.holdPiece.type

        # Centering offsets based on piece shape
        center_offsets = {
            'I': 0, 'J': 0, 'Z': 0,
            'O': 1, 'T': 1,
            'L': 2, 'S': 2
        }
        s_c = center_offsets.get(type, 1)  # Default to 1

        # Generate a pseudo-positioned shape
        self.holdPiece.shape = Piece.piece_shapes[self.holdPiece.type]
        pseudo_shape = [(r, c + s_c) for r, c in self.holdPiece.shape]

        # Ensure piece is placed within the 2-row grid
        for r, c in pseudo_shape:
            if 0 <= r < 2 and 0 <= c < 4:  # Prevent index errors
                self.holdPieceGrid[r][c] = type

        # Debugging output
        # print("Hold Piece:", self.holdPiece.type)
        # for row in self.holdPieceGrid:
        #     print(row)

    def spawnHoldPiece(self):
        self.currentPiece = Piece(self.currentPiece.type)
        for r, c in self.currentPiece.get_cells():
            if self.board[r][c] != 0:  # Game over condition (spawn area occupied)
                print("Game Over!")
                self.game_ended = True #pygame.quit()
                # exit()
            else:
                self.board[r][c] = self.currentPiece.type

    def dropPiece(self):
        count = 0
        current_positions = self.currentPiece.get_cells()
        while True:
            new_positions = [(r + 1, c) for r, c in current_positions]

            # Check if the piece can still move down
            if all((r < self.rows and self.board[r][c] == 0) or ((r, c) in current_positions) for r, c in new_positions):
                # Move the piece down by 1 row
                for r, c in current_positions:
                    self.board[r][c] = 0  # Clear the current position
                
                # Update the row and piece position
                self.currentPiece.row += 1
                count += 1
                
                # Set the piece in the new position
                for r, c in self.currentPiece.get_cells():
                    self.board[r][c] = self.currentPiece.type
                current_positions = self.currentPiece.get_cells()  # Update current_positions to the new one
            else:
                # Lock the piece in place and spawn a new piece
                # print("Locking piece")
                self.log.append('D')
                self.drop_height = count
                self.score += 2*count
                self.placePiece()  # Lock the piece and spawn a new one
                break  # Exit the loop since the piece has been locked