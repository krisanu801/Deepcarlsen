import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import io
import random
from model import ChessNN, DeepCarlsen
from torch.utils.data import Dataset, DataLoader

class ChessDataset(Dataset):
    def __init__(self, pgn_data, max_positions=10000):
        self.positions = []
        self.evaluations = []
        
        game_count = 0
        while len(self.positions) < max_positions:
            pgn = chess.pgn.read_game(io.StringIO(pgn_data))
            if pgn is None:
                break
                
            board = pgn.board()
            result = pgn.headers["Result"]
            
            # Convert result to numerical evaluation
            if result == "1-0":
                base_eval = 1.0
            elif result == "0-1":
                base_eval = -1.0
            else:
                base_eval = 0.0
            
            # Add positions from the game
            for move in pgn.mainline_moves():
                board.push(move)
                
                # Skip early game positions
                if len(self.positions) >= max_positions:
                    break
                
                # Add position and evaluation
                dc = DeepCarlsen()
                tensor = dc.board_to_tensor(board)
                self.positions.append(tensor)
                
                # Evaluation gradually approaches the game result
                moves_left = max(30 - board.fullmove_number, 1)
                current_eval = base_eval / moves_left
                self.evaluations.append(current_eval)
            
            game_count += 1
            if game_count % 10 == 0:
                print(f"Processed {game_count} games, {len(self.positions)} positions")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        # Remove the extra dimension and convert to float32
        position = self.positions[idx].squeeze(0).float()
        return position, torch.tensor(self.evaluations[idx], dtype=torch.float32)

def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()  # Set model to training mode
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_positions, batch_evals in train_loader:
            # Move tensors to device and ensure float32
            batch_positions = batch_positions.to(device).float()
            batch_evals = batch_evals.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(batch_positions)
            loss = criterion(outputs, batch_evals.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

def main():
    # Sample chess games for training
    sample_games = """
    [Event "Training Game"]
    [Result "1-0"]
    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 1-0

    [Event "Training Game"]
    [Result "0-1"]
    1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 Ne7 0-1

    [Event "Training Game"]
    [Result "1/2-1/2"]
    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O 1/2-1/2
    """
    
    # Create dataset and dataloader
    dataset = ChessDataset(sample_games)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train model
    model = ChessNN()
    trained_model = train_model(model, train_loader)
    
    # Save trained model
    torch.save(trained_model.state_dict(), 'chess_model.pth')
    print("Model saved to chess_model.pth")

if __name__ == "__main__":
    main()
