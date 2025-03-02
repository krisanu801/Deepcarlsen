import pygame
import chess
import os
from model import DeepCarlsen

class ChessGame:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("DeepCarlsen Chess")
        
        # Initialize game parameters
        self.square_size = 60
        self.board_offset = (50, 50)
        
        # Load chess pieces images
        self.piece_images = {}
        self.load_pieces()
        
        # Chess board and AI
        self.board = chess.Board()
        self.ai = DeepCarlsen()
        
        # Game state
        self.selected_square = None
        self.player_color = chess.WHITE
        self.game_over = False
        
        # Colors
        self.colors = {
            'light_square': (240, 217, 181),
            'dark_square': (181, 136, 99),
            'selected': (130, 151, 105),
            'highlight': (205, 210, 106),
            'text': (50, 50, 50)
        }
        
        # Fonts
        self.ui_font = pygame.font.SysFont('Arial', 36)
        self.small_font = pygame.font.SysFont('Arial', 24)
    
    def load_pieces(self):
        """Load chess piece images"""
        pieces = ['p', 'n', 'b', 'r', 'q', 'k']
        colors = ['b', 'w']
        for piece in pieces:
            for color in colors:
                name = color + piece
                image_path = os.path.join('assets', f'{name}.png')
                if os.path.exists(image_path):
                    image = pygame.image.load(image_path)
                    self.piece_images[name] = pygame.transform.scale(
                        image, 
                        (self.square_size, self.square_size)
                    )
    
    def get_square_from_pos(self, pos):
        """Convert screen position to chess square"""
        x, y = pos
        x -= self.board_offset[0]
        y -= self.board_offset[1]
        if 0 <= x < 8 * self.square_size and 0 <= y < 8 * self.square_size:
            file = x // self.square_size
            rank = 7 - (y // self.square_size)
            return chess.square(file, rank)
        return None
    
    def get_pos_from_square(self, square):
        """Convert chess square to screen position"""
        file = chess.square_file(square)
        rank = 7 - chess.square_rank(square)
        x = self.board_offset[0] + file * self.square_size
        y = self.board_offset[1] + rank * self.square_size
        return (x, y)
    
    def draw_board(self):
        """Draw the chess board"""
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)
                x = self.board_offset[0] + file * self.square_size
                y = self.board_offset[1] + rank * self.square_size
                color = self.colors['light_square'] if (rank + file) % 2 == 0 else self.colors['dark_square']
                
                # Draw square
                pygame.draw.rect(self.screen, color, 
                               (x, y, self.square_size, self.square_size))
                
                # Draw selected square highlight
                if square == self.selected_square:
                    pygame.draw.rect(self.screen, self.colors['selected'],
                                   (x, y, self.square_size, self.square_size), 3)
                    
                    # Highlight legal moves from this square
                    piece = self.board.piece_at(square)
                    if piece and piece.color == self.player_color:
                        for move in self.board.legal_moves:
                            if move.from_square == square:
                                to_x = self.board_offset[0] + (move.to_square % 8) * self.square_size
                                to_y = self.board_offset[1] + (7 - move.to_square // 8) * self.square_size
                                pygame.draw.circle(self.screen, self.colors['highlight'],
                                                 (to_x + self.square_size // 2, to_y + self.square_size // 2), 8)
                
                # Draw piece
                piece = self.board.piece_at(square)
                if piece:
                    color = 'w' if piece.color else 'b'
                    piece_name = color + piece.symbol().lower()
                    if piece_name in self.piece_images:
                        self.screen.blit(self.piece_images[piece_name], (x, y))
        
        # Draw rank and file labels
        for i in range(8):
            # Draw rank numbers (1-8)
            rank_text = self.small_font.render(str(8 - i), True, self.colors['text'])
            x = self.board_offset[0] - 25
            y = self.board_offset[1] + i * self.square_size + self.square_size // 2 - rank_text.get_height() // 2
            self.screen.blit(rank_text, (x, y))
            
            # Draw file letters (a-h)
            file_text = self.small_font.render(chr(97 + i), True, self.colors['text'])
            x = self.board_offset[0] + i * self.square_size + self.square_size // 2 - file_text.get_width() // 2
            y = self.board_offset[1] + 8 * self.square_size + 10
            self.screen.blit(file_text, (x, y))
    
    def draw_ui(self):
        """Draw UI elements"""
        # Draw difficulty selector
        difficulties = ['easy', 'intermediate', 'hard']
        for i, diff in enumerate(difficulties):
            x = self.width - 150
            y = 50 + i * 40
            # Draw button background
            button_rect = pygame.Rect(x, y, 100, 30)
            button_color = (100, 200, 100) if self.ai.difficulty == diff else (200, 200, 200)
            pygame.draw.rect(self.screen, button_color, button_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), button_rect, 1)  # border
            
            # Draw button text
            text = self.ui_font.render(diff.title(), True, (50, 50, 50))
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)
        
        # Draw game status
        status = "Your turn" if self.board.turn == self.player_color else "AI thinking..."
        if self.game_over:
            if self.board.is_checkmate():
                winner = "You" if self.board.turn != self.player_color else "AI"
                status = f"{winner} won!"
            elif self.board.is_stalemate():
                status = "Draw - Stalemate"
            elif self.board.is_insufficient_material():
                status = "Draw - Insufficient Material"
            elif self.board.is_fifty_moves():
                status = "Draw - Fifty Move Rule"
            elif self.board.is_repetition():
                status = "Draw - Repetition"
            else:
                status = "Game Over"
        
        status_text = self.ui_font.render(status, True, self.colors['text'])
        self.screen.blit(status_text, (self.width - 200, self.height - 50))
        
        # Draw instructions
        instructions = [
            "Click to select and move pieces",
            "Press 'R' to reset game",
            "ESC to quit"
        ]
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.colors['text'])
            self.screen.blit(text, (self.width - 200, self.height - 150 + i * 25))
    
    def handle_click(self, pos):
        """Handle mouse click events"""
        x, y = pos
        
        # Check if difficulty button was clicked
        if self.width - 150 <= x <= self.width - 50:
            if 50 <= y <= 90:
                self.ai.set_difficulty('easy')
                return
            elif 90 <= y <= 130:
                self.ai.set_difficulty('intermediate')
                return
            elif 130 <= y <= 170:
                self.ai.set_difficulty('hard')
                return
        
        # Handle board clicks
        square = self.get_square_from_pos(pos)
        if square is None:
            return
            
        if self.selected_square is None:
            # Select piece
            piece = self.board.piece_at(square)
            if piece and piece.color == self.player_color and not self.game_over:
                self.selected_square = square
        else:
            # Try to make move
            move = chess.Move(self.selected_square, square)
            
            # Check for pawn promotion
            if (self.board.piece_at(self.selected_square).piece_type == chess.PAWN and 
                ((self.player_color and square >= 56) or (not self.player_color and square <= 7))):
                move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
            
            if move in self.board.legal_moves:
                print(f"Making move: {move.uci()}")  # Debug print
                self.board.push(move)
                self.selected_square = None
                
                # Update display after player move
                self.screen.fill((255, 255, 255))
                self.draw_board()
                self.draw_ui()
                pygame.display.flip()
                pygame.time.wait(200)  # Small delay for smooth animation
                
                if not self.board.is_game_over():
                    # AI's turn
                    print("AI is thinking...")
                    ai_move = self.ai.get_best_move_simple(self.board)
                    if ai_move:
                        print(f"AI move: {ai_move.uci()}")
                        self.board.push(ai_move)
                        # Update display after AI move
                        self.screen.fill((255, 255, 255))
                        self.draw_board()
                        self.draw_ui()
                        pygame.display.flip()
                        pygame.time.wait(200)  # Small delay for smooth animation
                    
                self.game_over = self.board.is_game_over()
            else:
                print(f"Invalid move: {move.uci()}")  # Debug print
                self.selected_square = None
    
    def run(self):
        """Main game loop"""
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset game
                        self.board = chess.Board()
                        self.game_over = False
                        self.selected_square = None
                    elif event.key == pygame.K_ESCAPE:  # Quit game
                        running = False
            
            # Draw everything
            self.screen.fill((255, 255, 255))
            self.draw_board()
            self.draw_ui()
            pygame.display.flip()
            
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    game = ChessGame()
    game.run()
