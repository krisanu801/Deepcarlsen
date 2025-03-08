# DeepCarlsen Chess AI

DeepCarlsen is a chess AI that can play at different difficulty levels using deep learning. The model is inspired by AlphaZero's architecture and can adapt its playing strength based on the selected difficulty level.

## Features

- Three difficulty levels: Easy, Intermediate, and Hard
- Neural network-based evaluation and move selection
- Interactive GUI using Pygame
- Trained on chess game databases
- Adjustable search depth and randomness per difficulty level

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepcarlsen.git
cd deepcarlsen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download chess piece images:
Create an `assets` folder and add chess piece images named as follows:
- wp.png (white pawn)
- wn.png (white knight)
- wb.png (white bishop)
- wr.png (white rook)
- wq.png (white queen)
- wk.png (white king)
- bp.png (black pawn)
- bn.png (black knight)
- bb.png (black bishop)
- br.png (black rook)
- bq.png (black queen)
- bk.png (black king)

## Usage

1. To play against the AI:
```bash
python game.py
```

2. To train the model on your own dataset:
```bash
python train.py
```

## Project Structure

- `model.py`: Contains the neural network architecture
- `game.py`: Implements the chess GUI and game logic
- `train.py`: Training script for the model
- `requirements.txt`: List of Python dependencies
- `assets/`: Directory for chess piece images

## Model Architecture

DeepCarlsen uses a neural network with three main components:
1. **Board Encoder**: Processes the chess board state using convolutional layers
2. **Value Head**: Evaluates the position (-1 to 1)
3. **Policy Head**: Predicts move probabilities

## Difficulty Levels

- **Easy**: 
  - Shallow search depth (2 moves)
  - High temperature (1.0)
  - High randomness (0.3)

- **Intermediate**:
  - Medium search depth (4 moves)
  - Medium temperature (0.5)
  - Medium randomness (0.1)

- **Hard**:
  - Deep search depth (6 moves)
  - Low temperature (0.1)
  - No randomness (0.0)

## Controls

- Left click to select and move pieces
- Press 'R' to reset the game
- Click difficulty buttons to change AI strength

## Training

To train the model on your own dataset:
1. Place your PGN files in the `data` directory
2. Update the `pgn_files` list in `train.py`
3. Run `python train.py`

The trainer will:
- Split data into training and validation sets
- Train for the specified number of epochs
- Save the best model based on validation loss

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)  file for details.

## Acknowledgments

- Inspired by AlphaZero and Stockfish
- Chess piece images should be attributed to their respective creators
- Thanks to the python-chess and Pygame communities
# Deepcarlsen
