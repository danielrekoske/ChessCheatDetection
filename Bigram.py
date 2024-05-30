import re
from collections import defaultdict
import pandas as pd
import numpy as np

def parse_pgn(file_path):
    with open(file_path, 'r') as file:
        raw_data = file.read()
    
    games = re.split(r'\n\n(?=\[Event )', raw_data)
    
    games = [game for game in games if game.strip()]
    
    print(f'Number of games: {len(games)}')
    return games

def create_bigrams(tokenized_moves):
    result_tokens = {"1-0", "0-1", "1/2-1/2"}
    bigrams = [
        (tokenized_moves[i], tokenized_moves[i+1])
        for i in range(len(tokenized_moves) - 1)
        if tokenized_moves[i] not in result_tokens and tokenized_moves[i+1] not in result_tokens
    ]
    return bigrams

def process_pgn(file_path):
    games = parse_pgn(file_path)
    all_bigrams = defaultdict(int)
    
    for game in games:
        tokenized_moves = tokenize_pgn(game)
        bigrams = create_bigrams(tokenized_moves)
        for bigram in bigrams:
            all_bigrams[bigram] += 1
    
    return all_bigrams

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

def compute_bigram_probabilities(bigram_counts):
    total_bigrams = sum(bigram_counts.values())
    bigram_probabilities = {bigram: count / total_bigrams for bigram, count in bigram_counts.items()}
    return bigram_probabilities

def create_bigram_dataframe(bigram_probabilities):
    data = {
        'First Move': [bigram[0] for bigram in bigram_probabilities.keys()],
        'Second Move': [bigram[1] for bigram in bigram_probabilities.keys()],
        'Probability': list(bigram_probabilities.values())
    }
    df = pd.DataFrame(data)
    return df

def predict_next_move(df, current_move):
    filtered_df = df[df['First Move'] == current_move]
    if filtered_df.empty:
        return None
    return filtered_df.sample(weights=filtered_df['Probability']).iloc[0]['Second Move']

def save_bigram_dataframe(df, file_path):
    df.to_csv(file_path, index=False)

def load_bigram_dataframe(file_path):
    return pd.read_csv(file_path)
