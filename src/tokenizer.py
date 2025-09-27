# -*- coding: utf-8 -*-
import os
import argparse
import json
from tqdm import tqdm

class Tokenizer:
    """
    A simple and efficient character-level tokenizer.
    Handles vocabulary creation, encoding/decoding, and special tokens.
    """
    def __init__(self, special_tokens=None):
        self.char_to_index = {}
        self.index_to_char = {}
        self.vocab_size = 0
        
        # Define default special tokens
        self.special_tokens = special_tokens if special_tokens is not None else {
            'pad': '<pad>',  # Padding token
            'sos': '<sos>',  # Start of Sequence token
            'eos': '<eos>',  # End of Sequence token
            'unk': '<unk>'   # Unknown token
        }

    def _add_char(self, char):
        """Adds a character to the vocabulary if it's not already present."""
        if char not in self.char_to_index:
            self.index_to_char[self.vocab_size] = char
            self.char_to_index[char] = self.vocab_size
            self.vocab_size += 1
            
    def fit_on_file(self, filepath):
        """
        Builds the vocabulary by streaming a text file line-by-line.
        This is memory-efficient for very large files.
        """
        print("Building tokenizer vocabulary from file...")
        
        # Add special tokens first to ensure they have consistent indices (0, 1, 2...)
        for token_name, token_char in self.special_tokens.items():
            self._add_char(token_char)

        # Use a set to efficiently find all unique characters
        char_vocab = set()
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in tqdm(f, desc="Scanning for unique characters"):
                    char_vocab.update(line.strip())
        except FileNotFoundError:
            print(f"Error: Training file not found at {filepath}")
            raise

        # Add the unique characters from the file to our vocabulary
        for char in sorted(list(char_vocab)): # Sort for consistent mapping
            self._add_char(char)
            
        print(f"Vocabulary built. Total unique characters: {self.vocab_size}")

    def encode(self, text, add_sos_eos=False):
        """Converts a string of text into a list of integer tokens."""
        unk_token = self.char_to_index.get(self.special_tokens['unk'])
        encoded = [self.char_to_index.get(char, unk_token) for char in text]
        
        if add_sos_eos:
            sos_token = self.char_to_index.get(self.special_tokens['sos'])
            eos_token = self.char_to_index.get(self.special_tokens['eos'])
            encoded = [sos_token] + encoded + [eos_token]
            
        return encoded

    def decode(self, tokens, skip_special_tokens=True):
        """Converts a list of integer tokens back into a string."""
        decoded_chars = []
        special_token_chars = set(self.special_tokens.values())
        
        for token in tokens:
            char = self.index_to_char.get(token, self.special_tokens['unk'])
            if skip_special_tokens and char in special_token_chars:
                continue
            decoded_chars.append(char)
            
        return "".join(decoded_chars)

    def save(self, filepath):
        """Saves the tokenizer's vocabulary and configuration to a JSON file."""
        config = {
            'char_to_index': self.char_to_index,
            'index_to_char': self.index_to_char,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size
        }
        print(f"Saving tokenizer to {filepath}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print("Tokenizer saved successfully.")

    @classmethod
    def load(cls, filepath):
        """Loads a tokenizer from a JSON file."""
        print(f"Loading tokenizer from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        tokenizer = cls()
        tokenizer.char_to_index = config['char_to_index']
        # JSON saves integer keys as strings, so we must convert them back
        tokenizer.index_to_char = {int(k): v for k, v in config['index_to_char'].items()}
        tokenizer.special_tokens = config['special_tokens']
        tokenizer.vocab_size = config['vocab_size']
        print("Tokenizer loaded successfully.")
        return tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build and save a character-level tokenizer.")
    parser.add_argument(
        '--input-file',
        type=str,
        default= r'C:\Users\AIDSHPCLAB08\Desktop\dpiit-project\Passowrd_decryption_and_extractor\data\processed\stage1_train.txt',
        help="Path to the processed training password file to build vocabulary from."
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default= r'C:\Users\AIDSHPCLAB08\Desktop\dpiit-project\Passowrd_decryption_and_extractor\models/tokenizer.json',
        help="Path to save the output tokenizer JSON file."
    )
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize and build the tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_file(args.input_file)
    
    # Save the tokenizer
    tokenizer.save(args.output_file)

    # --- A quick test to demonstrate functionality ---
    print("\n--- Tokenizer Test ---")
    test_password = "Password123!"
    encoded = tokenizer.encode(test_password, add_sos_eos=True)
    decoded = tokenizer.decode(encoded)
    print(f"Original: '{test_password}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
