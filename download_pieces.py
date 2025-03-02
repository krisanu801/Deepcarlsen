import os
import requests

def download_piece(piece_name, url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {piece_name}")
    else:
        print(f"Failed to download {piece_name}")

def main():
    # Create assets directory if it doesn't exist
    if not os.path.exists('assets'):
        os.makedirs('assets')

    # Chess piece images from Wikimedia Commons (standard chess pieces)
    pieces = {
        'wk': 'https://upload.wikimedia.org/wikipedia/commons/4/42/Chess_klt45.svg',
        'wq': 'https://upload.wikimedia.org/wikipedia/commons/1/15/Chess_qlt45.svg',
        'wr': 'https://upload.wikimedia.org/wikipedia/commons/7/72/Chess_rlt45.svg',
        'wb': 'https://upload.wikimedia.org/wikipedia/commons/b/b1/Chess_blt45.svg',
        'wn': 'https://upload.wikimedia.org/wikipedia/commons/7/70/Chess_nlt45.svg',
        'wp': 'https://upload.wikimedia.org/wikipedia/commons/4/45/Chess_plt45.svg',
        'bk': 'https://upload.wikimedia.org/wikipedia/commons/f/f0/Chess_kdt45.svg',
        'bq': 'https://upload.wikimedia.org/wikipedia/commons/4/47/Chess_qdt45.svg',
        'br': 'https://upload.wikimedia.org/wikipedia/commons/f/ff/Chess_rdt45.svg',
        'bb': 'https://upload.wikimedia.org/wikipedia/commons/9/98/Chess_bdt45.svg',
        'bn': 'https://upload.wikimedia.org/wikipedia/commons/e/ef/Chess_ndt45.svg',
        'bp': 'https://upload.wikimedia.org/wikipedia/commons/c/c7/Chess_pdt45.svg',
    }

    for piece, url in pieces.items():
        save_path = os.path.join('assets', f'{piece}.png')
        download_piece(piece, url, save_path)

if __name__ == "__main__":
    main()
