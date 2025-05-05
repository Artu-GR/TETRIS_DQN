class Piece():
    start_positions = {  # Initial spawn locations
        'I': (0, 3), 'O': (0, 4), 'T': (0, 4),
        'L': (0, 5), 'J': (0, 3), 'S': (0, 5), 'Z': (0, 3)
    }

    piece_shapes = {  # Tetris shapes in relative (row, col) positions
        'I': [(0, 0), (0, 1), (0, 2), (0, 3)],
        'O': [(0, 0), (0, 1), (1, 0), (1, 1)],
        'T': [(0, 0), (1, -1), (1, 0), (1, 1)],
        'L': [(0, 0), (1, 0), (1, -1), (1, -2)],
        'J': [(0, 0), (1, 0), (1, 1), (1, 2)],
        'S': [(0, 0), (0, 1), (1, 0), (1, -1)],
        'Z': [(0, 0), (0, 1), (1, 1), (1, 2)]
    }

    def __init__(self, piece_type):
        self.type = piece_type
        self.shape = Piece.piece_shapes[piece_type]
        self.row, self.col = Piece.start_positions[piece_type]
        self.cells = self.get_cells()

    def get_cells(self, cells=None):
        if cells==None:
            return [(self.row + r, self.col + c) for r, c in self.shape]
        else:
            return [(self.row + r, self.col + c) for r, c in cells]