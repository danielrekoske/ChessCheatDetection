import re
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import math
from dataclasses import dataclass
import Bigram

def tokenize_pgn(pgn_text):
    pgn_regexes = {
        'comment': r'(;.*?)\n',           # Line comment
        'tag': r'\[.*?\]',                # Tag pair
        'comment_open': r'\{',            # Inline comment start
        'comment_close': r'\}',           # Inline comment end
        'move_number': r'\d+\.+',         # Move number
        'symbol': r'[-+\w./]+',           # Chess move symbols
        'newline': r'\n',                 # Newline
        'whitespace': r'\s+'              # Whitespace
    }
    
    master_regex = '|'.join(f'(?P<{name}>{regex})' for name, regex in pgn_regexes.items())
    
    tokens = []
    for match in re.finditer(master_regex, pgn_text):
        for name, regex in pgn_regexes.items():
            if match.group(name):
                tokens.append((name, match.group(name)))
                break
    
    filtered_tokens = [token for token in tokens if token[0] not in ('whitespace', 'newline')]
    
    move_tokens = [token[1] for token in filtered_tokens if token[0] == 'symbol']
    
    return move_tokens

def parse_and_tokenize_pgn(file_path):
    with open(file_path, 'r') as file:
        raw_data = file.read()
    
    games = re.split(r'\n\n(?=\[Event )', raw_data)
    games = [game for game in games if game.strip()]

    all_moves = []
    unique_moves = set()
    
    for game in games:
        moves = tokenize_pgn(game)
        all_moves.append(moves)
        unique_moves.update(moves)
    
    vocab = {move: idx for idx, move in enumerate(unique_moves)}
    return all_moves, vocab

file_path = 'comp_chess_games.pgn'

all_moves, vocab = parse_and_tokenize_pgn(file_path)

all_token_indices = [[vocab[move] for move in moves] for moves in all_moves]

voab_size = len(vocab)