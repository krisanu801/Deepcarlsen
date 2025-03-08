import os
import matplotlib.pyplot as plt

# Unicode symbols for chess pieces
chess_pieces = {
    "wk": "♔",  # White King
    "wq": "♕",  # White Queen
    "wr": "♖",  # White Rook
    "wb": "♗",  # White Bishop
    "wn": "♘",  # White Knight
    "wp": "♙",  # White Pawn
    "bk": "♚",  # Black King
    "bq": "♛",  # Black Queen
    "br": "♜",  # Black Rook
    "bb": "♝",  # Black Bishop
    "bn": "♞",  # Black Knight
    "bp": "♟",  # Black Pawn
}

# Create assets folder if it doesn't exist
os.makedirs("assets", exist_ok=True)

def save_chess_piece(filename, symbol):
    """Generate and save a chess piece image with no white background."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=300)  # High resolution
    ax.text(0.5, 0.5, symbol, fontsize=120, ha='center', va='center')  # Bigger size
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_facecolor("none")  # Transparent background

    # Save image without padding & with transparency
    filepath = os.path.join("assets", f"{filename}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"Saved: {filepath}")

# Generate and save all chess pieces
for filename, symbol in chess_pieces.items():
    save_chess_piece(filename, symbol)

print("All chess pieces saved in the 'assets' folder without white areas!")
