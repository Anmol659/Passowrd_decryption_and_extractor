# -*- coding: utf-8 -*-
import os
import argparse
import json
from collections import Counter
from tqdm import tqdm
import re

def password_to_pattern(password):
    """
    Converts a password string into its structural pattern representation.
    Example: 'Password123!' -> 'UL7D3S1'
    
    L: Lowercase
    U: Uppercase
    D: Digit
    S: Special Character
    """
    if not password:
        return ""
        
    pattern = []
    
    # Define character types using regex for efficiency
    char_types = {
        'U': re.compile(r'[A-Z]'),
        'L': re.compile(r'[a-z]'),
        'D': re.compile(r'[0-9]'),
        'S': re.compile(r'[^A-Za-z0-9]') # Anything not a letter or digit
    }

    # First, convert password to a sequence of type characters (e.g., "ULLLDDS")
    type_sequence = []
    for char in password:
        found = False
        for type_char, regex in char_types.items():
            if regex.match(char):
                type_sequence.append(type_char)
                found = True
                break
        if not found:
            # Fallback for any unexpected characters, though our filter should prevent this
            type_sequence.append('S') 

    if not type_sequence:
        return ""

    # Now, compress the sequence (e.g., "ULLLDDS" -> "U1L3D2S1")
    current_char = type_sequence[0]
    count = 1
    for next_char in type_sequence[1:]:
        if next_char == current_char:
            count += 1
        else:
            pattern.append(f"{current_char}{count}")
            current_char = next_char
            count = 1
    pattern.append(f"{current_char}{count}") # Append the last sequence
            
    return "".join(pattern)

def induce_grammar(args):
    """
    Reads a large password file, induces a probabilistic grammar (counts patterns),
    and saves the result as a JSON file.
    """
    print("--- Starting PCFG Grammar Induction ---")
    print(f"Input training file: {args.input_file}")
    print(f"Output grammar file: {args.output_file}")
    
    # Use collections.Counter for highly efficient counting
    pattern_counts = Counter()
    total_passwords = 0

    try:
        with open(args.input_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="Analyzing password structures"):
                password = line.strip()
                if password:
                    pattern = password_to_pattern(password)
                    pattern_counts[pattern] += 1
                    total_passwords += 1
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.input_file}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    if total_passwords == 0:
        print("No passwords found in the input file. Aborting.")
        return

    print(f"\nAnalyzed {total_passwords:,} passwords.")
    print(f"Found {len(pattern_counts):,} unique patterns.")

    # Convert counts to probabilities
    print("Calculating pattern probabilities...")
    pattern_probabilities = {
        pattern: count / total_passwords
        for pattern, count in pattern_counts.items()
    }

    # Prepare final data structure for saving
    output_data = {
        "total_passwords": total_passwords,
        "total_unique_patterns": len(pattern_counts),
        "pattern_probabilities": pattern_probabilities
    }

    # Save the grammar to a JSON file
    print(f"Saving grammar to {args.output_file}...")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            json.dump(output_data, f_out, indent=4)
    except Exception as e:
        print(f"Failed to save JSON file: {e}")
        return

    print("\n--- Grammar Induction Complete ---")
    # Show the top 10 most common patterns as a sanity check
    print("Top 10 most common patterns found:")
    for pattern, count in pattern_counts.most_common(10):
        prob = pattern_probabilities[pattern]
        print(f"  - Pattern: {pattern:<20} | Count: {count:<15,} | Probability: {prob:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Induce a PCFG from a password training set.")
    parser.add_argument(
        '--input-file', 
        type=str, 
        default=r"C:\Users\AIDSHPCLAB08\Desktop\dpiit-project\Passowrd_decryption_and_extractor\data\processed\stage1_train.txt",
        help="Path to the processed training password file."
    )
    parser.add_argument(
        '--output-file', 
        type=str, 
        default= r'C:/Users/AIDSHPCLAB08/Desktop/dpiit-project/Passowrd_decryption_and_extractor/models/pcfg/pcfg_grammar.json',
        help="Path to save the output JSON grammar file."
    )
    
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    induce_grammar(args)
