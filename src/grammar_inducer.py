# -*- coding: utf-8 -*-
import os
import argparse
import json
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def password_to_pattern(password: str) -> str:
    if not password:
        return ""

    seq = []
    for c in password:
        if c.isupper():
            seq.append("U")
        elif c.islower():
            seq.append("L")
        elif c.isdigit():
            seq.append("D")
        else:
            seq.append("S")

    if not seq:
        return ""

    result = []
    current = seq[0]
    count = 1
    for nxt in seq[1:]:
        if nxt == current:
            count += 1
        else:
            result.append(f"{current}{count}")
            current, count = nxt, 1
    result.append(f"{current}{count}")
    return "".join(result)


def process_chunk(lines, top_k=1_000_000):
    """Process a chunk of lines into a Counter, pruned to top-K patterns."""
    c = Counter()
    for line in lines:
        pw = line.strip()
        if pw:
            c[password_to_pattern(pw)] += 1

    # keep only top_k to avoid massive return payload
    return Counter(dict(c.most_common(top_k)))


def merge_counters(c1: Counter, c2: Counter, top_k=1_000_000) -> Counter:
    """Merge two Counters and keep only top-K patterns."""
    c1.update(c2)
    if len(c1) > top_k * 2:  # prune aggressively
        c1 = Counter(dict(c1.most_common(top_k)))
    return c1


def induce_grammar(input_file, output_file, chunk_size=1_000_000, top_k=10_000_000):
    print("--- Starting PCFG Grammar Induction (Multiprocessing Top-K mode) ---")
    print(f"Input file : {input_file}")
    print(f"Output file: {output_file}\n")

    total_passwords = 0
    final_counts = Counter()

    with Pool(processes=cpu_count()) as pool:
        with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
            chunk = []
            futures = []
            chunk_id = 0

            for line in f:
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    chunk_id += 1
                    print(f"[INFO] Queuing chunk {chunk_id:,} ({len(chunk):,} lines)")
                    futures.append(pool.apply_async(process_chunk, (chunk, top_k)))
                    chunk = []

                    # Merge results early to free memory
                    if len(futures) >= cpu_count() * 2:
                        print("[INFO] Merging intermediate results...")
                        for ft in futures:
                            final_counts = merge_counters(final_counts, ft.get(), top_k)
                        futures.clear()

            # leftover
            if chunk:
                chunk_id += 1
                print(f"[INFO] Queuing final chunk {chunk_id:,} ({len(chunk):,} lines)")
                futures.append(pool.apply_async(process_chunk, (chunk, top_k)))

            print("\n[INFO] Waiting for workers to finish...")
            for ft in tqdm(futures, desc="Merging results"):
                final_counts = merge_counters(final_counts, ft.get(), top_k)

    total_passwords = sum(final_counts.values())

    print(f"\nAnalyzed {total_passwords:,} passwords.")
    print(f"Kept top {len(final_counts):,} patterns (pruned from billions).")

    # compute probabilities for top patterns only
    pattern_probabilities = {
        p: c / total_passwords for p, c in final_counts.items()
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump({
            "total_passwords": total_passwords,
            "top_k": top_k,
            "patterns": pattern_probabilities,
        }, f_out, indent=4)

    print("\n--- Grammar Induction Complete ---")
    print("Top 10 most common patterns:")
    for p, c in final_counts.most_common(10):
        print(f"  - {p:<20} | Count: {c:<15,} | Prob: {pattern_probabilities[p]:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Induce a PCFG from a password training set.")
    parser.add_argument(
        "--input-file",
        type=str,
        default=r"C:\Users\AIDSHPCLAB08\Desktop\dpiit-project\Passowrd_decryption_and_extractor\data\processed\stage1_train.txt",
        help="Path to the processed training password file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=r"C:\Users\AIDSHPCLAB08\Desktop\dpiit-project\Passowrd_decryption_and_extractor\models\pcfg\pcfg_grammar.json",
        help="Path to save the output JSON grammar file."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1_000_000,
        help="Number of most frequent patterns to keep."
    )
    args = parser.parse_args()

    induce_grammar(args.input_file, args.output_file, top_k=args.top_k)
