import time
import random
import copy
import pygame
#import numpy as np

from .Piece import Piece

pygame.mixer.init()
RotationSFX = pygame.mixer.Sound("assets/audio/sfx/Rotation.mp3")
TetrisSFX = pygame.mixer.Sound("assets/audio/sfx/TetrisClear.mp3")
ClearSFX = pygame.mixer.Sound("assets/audio/sfx/ClearLine.mp3")

class GameState(): #10x20
    def __init__(self):
        self.board = [[0]*10 for _ in range(20)]
        self.prev_board = [[0] * 10 for _ in range(20)]
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
        self.nextPiecesGrid = [[0]*4 for _ in range(9)]

        # Hold piece
        self.holdPiece = None
        self.holdPieceGrid = [[0]*4 for _ in range(2)]

        # Difficulty increment
        self.SPEED_FACTOR = 1

        self.spawnPieces()

    # DQN SETUP
    def reset_game(self):
        # Reset the board and game over flag
        self.prev_board = [[0] * 10 for _ in range(20)]
        self.board = [[0]*10 for _ in range(20)]
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
    
    def place_piece(self, rotation, col, gpx):
        # clock = pygame.time.Clock()
        # clock.tick(3)
        for _ in range(rotation % 4):
            self.rotatePiece()
            if gpx is not None:
                self.update()
                gpx.drawBoard(None, self)
                pygame.display.flip()
                pygame.time.delay(50)
        while self.currentPiece.col < col:
            if self.moveRight() == False: break
            if gpx is not None:
                self.update()
                gpx.drawBoard(None, self)
                pygame.display.flip()
                pygame.time.delay(50)
        while self.currentPiece.col > col:
            if self.moveLeft() == False: break
            if gpx is not None:
                self.update()
                gpx.drawBoard(None, self)
                pygame.display.flip()
                pygame.time.delay(50)
        lines_cleared = self.dropPiece(return_state=True)
        if gpx is not None:
            self.update()
            gpx.drawBoard(None, self)
            pygame.display.flip()

        return lines_cleared

    def calculate_reward(self, ep, lines_cleared, prev_board):
        reward = 0

        new_board = self.board

        prev_holes = self.get_holes(prev_board)
        new_holes = self.get_holes(new_board)

        prev_bumpiness, prev_height = self.get_bumpiness_and_heights(prev_board)
        new_bumpiness, new_height = self.get_bumpiness_and_heights(new_board)

        factor = min(1.0, ep / 7000)
        # hole_weight = 0.2 * factor
        # bumpiness_weight = 0.05 * factor
        # height_weight = 0.1 * factor

        # REMOVE HEIGHT PUNISHMENT AS IT FLATTENS ALL 'I'

        # TO TRY
        hole_weight = 0.04 * factor
        bumpiness_weight = 0.0125 * factor
        #height_weight = 0.025 * factor

        # reward_factor = 5 + (min(10, ep/200))
        # TO TRY
        reward_factor = 15 + (min(25, ep/500))
        reward += (lines_cleared**2) * reward_factor

        reward -= hole_weight * (new_holes - prev_holes)
        reward -= bumpiness_weight * (new_bumpiness - prev_bumpiness)
        #reward -= height_weight * (new_height - prev_height)

        if self.game_ended:
            reward -= 5

        reward += 0.01

        return reward

        # reward = 0
        # reward += lines_cleared*2.5 #[0, 10, 20, 30, 70][lines_cleared]
        # #reward -= holes_pen*0.5

        # return reward
    
    def get_holes(self, board):
        holes = 0
        for col in range(len(board[0])):
            block_found = False
            for row in range(len(board)):
                if board[row][col] != 0:
                    block_found = True
                elif block_found:
                    holes += 1

        return holes

    def get_bumpiness_and_heights(self, board):
        heights = []
        for col in range(len(board[0])):
            col_height = 0
            for row in range(len(board)):
                if board[row][col] != 0:
                    col_height = len(board) - row
                    break
            heights.append(col_height)
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))

        return bumpiness, max(heights)

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

    def placePiece(self, return_state=False):
        for r, c in self.currentPiece.get_cells():
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.board[r][c] = self.currentPiece.type

        # Spawn a new piece
        self.drop_height = 0
        lines_cleared = self.clearFullRows(return_count=True)
        self.spawnPieces()
        self.log.clear()
        self.hold_used = False
        if return_state: return lines_cleared

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

    def clearFullRows(self, return_count=False):
        count = 0
        for row in range(self.rows):
            if all(self.board[row][col] != 0 for col in range(self.cols)):  # Row is full
                #BUSCAR LA FORMA DE LLAMAR DRAW_BOARD PARA VER EL TABLERO AL MOMENTO DE ELIMINAR LAS FILAS
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
            if count:
                print(f"*****\n{count} LINES CLEARED\n*****")
                #time.sleep(2)
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
            # RotationSFX.play()
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

    def dropPiece(self, return_state=False):
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
                lines_cleared = self.placePiece(return_state=True)  # Lock the piece and spawn a new one
                if return_state: return lines_cleared
                break  # Exit the loop since the piece has been locked