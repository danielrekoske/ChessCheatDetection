import Bigram
# Load the DataFrame from the CSV file (for future use)
loaded_bigram_df = Bigram.load_bigram_dataframe("bigram_probabilities.csv")

# Example of predicting the next move
current_move = 'e4'  # Example starting move
predicted_move = Bigram.predict_next_move(loaded_bigram_df, current_move)
print(f"Predicted next move after '{current_move}': {predicted_move}")
