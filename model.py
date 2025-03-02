import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import random
import numpy as np

class ChessNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 12 channels (6 piece types x 2 colors)
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)  # Output: position evaluation
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten and fully connected layers
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.tanh(self.fc2(x))  # Output between -1 and 1
        return x

class DeepCarlsen:
    def __init__(self, difficulty='intermediate'):
        self.difficulty = difficulty
        self.model = ChessNN()
        # Try to load the trained model if it exists
        try:
            self.model.load_state_dict(torch.load('chess_model.pth'))
            self.model.eval()
            print("Loaded trained model")
        except:
            print("Using untrained model")
            
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        # Search depth based on difficulty
        self.depth_by_difficulty = {
            'easy': 1,
            'intermediate': 2,
            'hard': 3
        }
    
    def board_to_tensor(self, board):
        """Convert chess board to tensor representation"""
        pieces = [
            chess.PAWN, chess.KNIGHT, chess.BISHOP,
            chess.ROOK, chess.QUEEN, chess.KING
        ]
        tensor = torch.zeros(12, 8, 8)  # 12 planes: 6 piece types x 2 colors
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                piece_idx = pieces.index(piece.piece_type)
                if not piece.color:  # Black pieces in second half of channels
                    piece_idx += 6
                tensor[piece_idx][7-rank][file] = 1
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def evaluate_position(self, board):
        """Evaluate chess position using neural network and material count"""
        # Material counting
        material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color:
                    material += value
                else:
                    material -= value
        
        # Neural network evaluation
        with torch.no_grad():
            board_tensor = self.board_to_tensor(board)
            nn_eval = self.model(board_tensor).item()
        
        # Combine material and positional evaluation
        total_eval = material / 30 + nn_eval  # Normalize material score
        
        # Add randomness based on difficulty
        if self.difficulty == 'easy':
            total_eval += random.uniform(-1, 1)
        elif self.difficulty == 'intermediate':
            total_eval += random.uniform(-0.3, 0.3)
        
        return total_eval if board.turn else -total_eval
    
    def minimax(self, board, depth, alpha, beta):
        """Minimax algorithm with alpha-beta pruning"""
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board)
        
        value = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            value = max(value, -self.minimax(board, depth - 1, -beta, -alpha))
            board.pop()
            
            alpha = max(alpha, value)
            if alpha >= beta:
                break
                
        return value
    
    def get_best_move_simple(self, board):
        """Get the best move using minimax with alpha-beta pruning"""
        depth = self.depth_by_difficulty[self.difficulty]
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        
        # Randomize move order for more variety
        random.shuffle(legal_moves)
        
        for move in legal_moves:
            # Make the move
            board.push(move)
            
            # Get value of resulting position
            value = -self.minimax(board, depth - 1, -beta, -alpha)
            
            # Undo the move
            board.pop()
            
            # Update best move if needed
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, value)
            
        return best_move
    
    def get_best_move(self, board):
        """Get best move using minimax search"""
        depth = self.depth_by_difficulty[self.difficulty]
        _, best_move = self.minimax(board, depth, float('-inf'), float('inf'), True)
        return best_move
    
    def set_difficulty(self, difficulty):
        """Change the AI difficulty level"""
        if difficulty in self.depth_by_difficulty:
            self.difficulty = difficulty
            print(f"Difficulty set to: {difficulty} (search depth: {self.depth_by_difficulty[difficulty]})")
        else:
            raise ValueError(f"Invalid difficulty level. Choose from: {list(self.depth_by_difficulty.keys())}")
