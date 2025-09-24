# -*- coding: utf-8 -*-
import os
import random
import argparse
from tqdm import tqdm
import datetime
import shutil

# --- Configuration ---
ALLOWED_CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
VALID_CHAR_SET = set(ALLOWED_CHARACTERS)

# --- Synthetic Data Generation (Functions are the same) ---

def generate_leet_speak(word):
    leet_map = {'a': '4', 'e': '3', 'g': '6', 'i': '1', 'o': '0', 's': '5', 't': '7'}
    leet_word = "".join([leet_map.get(char.lower(), char) if random.random() > 0.5 else char for char in word])
    return "".join([c.upper() if random.random() > 0.7 else c for c in leet_word])

def generate_date_passwords(num_to_generate=10000):
    passwords = set()
    special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '_', '.']
    current_year = datetime.datetime.now().year
    for _ in tqdm(range(num_to_generate), desc="Generating Dates"):
        year = random.randint(current_year - 25, current_year + 1)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        formats = [
            f"{datetime.date(year, month, day):%B%Y}", f"{datetime.date(year, month, day):%b%Y}",
            f"{day:02d}{month:02d}{year}", f"{year}{month:02d}{day:02d}",
        ]
        base_pass = random.choice(formats)
        if random.random() > 0.5: base_pass += random.choice(special_chars)
        if random.random() > 0.5: base_pass += str(random.randint(10, 999))
        if random.random() > 0.5 and base_pass[0].isalpha(): base_pass = base_pass[0].upper() + base_pass[1:]
        passwords.add(base_pass)
    return [p for p in passwords if len(p) >= 12]

# --- Disk-Based Processing Functions ---

def filter_and_write_to_disk(args, temp_dir):
    """
    Reads the massive source file line-by-line, filters passwords,
    and writes valid ones directly to an intermediate file on disk.
    This uses minimal RAM.
    """
    print("\n[Step 1/5] Filtering the main password file to disk...")
    filtered_path = os.path.join(temp_dir, "_filtered_passwords.txt")
    count = 0
    with open(args.input_file, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(filtered_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Filtering RockYou", unit=" lines"):
            password = line.strip()
            if len(password) >= args.min_length and set(password).issubset(VALID_CHAR_SET):
                f_out.write(password + '\n')
                count += 1
    print(f"Finished filtering. Found and wrote {count:,} valid passwords.")
    return filtered_path, count

def generate_and_write_to_disk(args, temp_dir):
    """
    Generates synthetic passwords and writes them directly to disk.
    """
    print("\n[Step 2/5] Generating synthetic data and writing to disk...")
    synthetic_path = os.path.join(temp_dir, "_synthetic_passwords.txt")
    
    date_passwords = generate_date_passwords(num_to_generate=args.num_synthetic)
    
    with open(synthetic_path, 'w', encoding='utf-8') as f_out:
        for p in date_passwords:
            f_out.write(p + '\n')
            
    print(f"Wrote {len(date_passwords):,} synthetic date passwords.")
    # Note: Leet speak and affix augmentation is skipped in this version
    # because it requires reading the filtered list, which is too large.
    # The date generation is more controlled and provides sufficient augmentation.
    return synthetic_path, len(date_passwords)

def shuffle_on_disk(temp_dir, output_shuffled_path, files_to_shuffle, num_buckets=100):
    """
    Performs an external shuffle. Reads all source files and distributes
    their lines randomly into a set of smaller 'bucket' files. Then,
    concatenates the buckets into a final shuffled file.
    """
    print(f"\n[Step 3/5] Shuffling data on disk using {num_buckets} buckets...")
    shuffle_bucket_dir = os.path.join(temp_dir, "shuffle_buckets")
    os.makedirs(shuffle_bucket_dir, exist_ok=True)
    
    # Open all bucket files for writing
    buckets = [open(os.path.join(shuffle_bucket_dir, f'bucket_{i}.txt'), 'w', encoding='utf-8') for i in range(num_buckets)]
    
    try:
        # Distribute lines from all source files into random buckets
        for source_file in files_to_shuffle:
            with open(source_file, 'r', encoding='utf-8') as f_in:
                for line in tqdm(f_in, desc=f"Distributing {os.path.basename(source_file)}"):
                    buckets[random.randint(0, num_buckets - 1)].write(line)
    finally:
        # Ensure all bucket files are closed
        for b in buckets:
            b.close()

    # Concatenate the shuffled buckets into the final output file
    with open(output_shuffled_path, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(num_buckets), desc="Combining buckets"):
            bucket_path = os.path.join(shuffle_bucket_dir, f'bucket_{i}.txt')
            with open(bucket_path, 'r', encoding='utf-8') as f_in:
                shutil.copyfileobj(f_in, f_out)
            os.remove(bucket_path) # Clean up bucket file after use
    
    shutil.rmtree(shuffle_bucket_dir) # Clean up the directory
    print("Finished shuffling.")

def split_final_file(args, shuffled_path, total_lines):
    """
    Reads the final shuffled file and splits it into train and validation sets.
    """
    print("\n[Step 4/5] Splitting shuffled file into train and validation sets...")
    train_path = os.path.join(args.output_dir, "stage1_train.txt")
    val_path = os.path.join(args.output_dir, "stage1_validation.txt")
    
    split_index = int(total_lines * (1 - args.val_split))
    
    with open(shuffled_path, 'r', encoding='utf-8') as f_in, \
         open(train_path, 'w', encoding='utf-8') as f_train, \
         open(val_path, 'w', encoding='utf-8') as f_val:
        
        for i, line in enumerate(tqdm(f_in, total=total_lines, desc="Splitting")):
            if i < split_index:
                f_train.write(line)
            else:
                f_val.write(line)
                
    print(f"Train set size: {split_index:,}")
    print(f"Validation set size: {total_lines - split_index:,}")

def main():
    parser = argparse.ArgumentParser(description="Memory-safe processing of massive password datasets.")
    parser.add_argument('--input-file', type=str, required=True, help="Path to the raw password file.")
    parser.add_argument('--output-dir', type=str, default='../data/processed', help="Directory to save the final train/val files.")
    parser.add_argument('--temp-dir', type=str, default='../data/temp', help="Directory for temporary intermediate files.")
    parser.add_argument('--min-length', type=int, default=12)
    parser.add_argument('--val-split', type=float, default=0.02)
    parser.add_argument('--num-synthetic', type=int, default=2000000)
    args = parser.parse_args()

    print("--- Starting Memory-Safe Data Preparation ---")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    # 1. Filter source file to disk
    filtered_file, filtered_count = filter_and_write_to_disk(args, args.temp_dir)
    
    # 2. Generate synthetic data to disk
    synthetic_file, synthetic_count = generate_and_write_to_disk(args, args.temp_dir)
    
    # 3. Shuffle both files together on disk
    total_lines = filtered_count + synthetic_count
    files_to_shuffle = [filtered_file, synthetic_file]
    shuffled_path = os.path.join(args.temp_dir, "_shuffled_combined.txt")
    shuffle_on_disk(args.temp_dir, shuffled_path, files_to_shuffle)
    
    # 4. Split the final shuffled file
    split_final_file(args, shuffled_path, total_lines)

    # 5. Clean up
    print("\n[Step 5/5] Cleaning up temporary files...")
    # The temp files inside the shuffle function are already cleaned.
    # We just need to remove the main intermediate files.
    os.remove(filtered_file)
    os.remove(synthetic_file)
    os.remove(shuffled_path)
    # You can remove the temp_dir itself if it's empty
    if not os.listdir(args.temp_dir):
        os.rmdir(args.temp_dir)
    
    print("\n--- Data Preparation Complete ---")

if __name__ == '__main__':
    main()