# -*- coding: utf-8 -*-
import os
import argparse
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import struct
import numpy as np
import mmap
import psutil

# Helper function to print memory usage
def print_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{stage}] Current RAM usage: {mem_info.rss / 1024 ** 3:.2f} GB")

def build_index(data_file, index_file):
    """
    Scans a large text file and creates a binary index file of line offsets.
    This is done once and saves a huge amount of RAM on subsequent runs.
    """
    print(f"Index file '{index_file}' not found. Building it now...")
    with open(data_file, 'rb') as f_data:
        # Get total file size for tqdm progress bar
        total_size = os.fstat(f_data.fileno()).st_size
        pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Indexing {os.path.basename(data_file)}")
        
        with open(index_file, 'wb') as f_idx:
            offset = 0
            # We write the first offset (0)
            f_idx.write(struct.pack('<Q', offset))
            
            while True:
                line = f_data.readline()
                if not line:
                    break
                offset = f_data.tell()
                f_idx.write(struct.pack('<Q', offset))
                pbar.update(len(line))
        pbar.close()
    print("Index built successfully.")

class PasswordDataset(Dataset):
    """
    A highly memory-efficient Dataset using a memory-mapped index file.
    RAM usage is negligible regardless of the number of passwords.
    """
    def __init__(self, file_path, tokenizer, max_seq_len=256):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        index_file = file_path + ".idx"
        if not os.path.exists(index_file):
            build_index(file_path, index_file)

        print(f"Loading memory-mapped index from: {index_file}")
        self.index_file = open(index_file, 'rb')
        # Use mmap to treat the file on disk as if it were a byte array in memory
        self.mmap = mmap.mmap(self.index_file.fileno(), 0, access=mmap.ACCESS_READ)
        # Interpret the mmap bytes as an array of 64-bit unsigned integers
        # We slice off the last entry because it points to the end of the file
        self.line_offsets = np.frombuffer(self.mmap, dtype=np.uint64)[:-1]
        
        print(f"Found {len(self.line_offsets)} passwords.")

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        offset = self.line_offsets[idx]
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(offset)
            line = f.readline().strip()

        if len(line) > self.max_seq_len - 2:
            line = line[:self.max_seq_len - 2]
        
        tokens = self.tokenizer.encode(line, add_sos_eos=True)
        
        src = tokens[:-1]
        tgt = tokens[1:]
        
        return torch.tensor(src), torch.tensor(tgt)

    def close(self):
        """Close the memory-map and file handles."""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'index_file'):
            self.index_file.close()

def collate_fn(batch, pad_token_id):
    src_batch, tgt_batch = [], []
    for src_item, tgt_item in batch:
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)

    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_token_id)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_token_id)
    
    return src_padded, tgt_padded

def train(args):
    """Main training function."""
    print_memory_usage("Start of Training Script")
    
    # Import necessary modules inside the function
    from model import PassFormer
    from tokenizer import Tokenizer
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available.")
        return
        
    device = torch.device("cuda")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    tokenizer = Tokenizer.load(args.tokenizer_path)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.char_to_index[tokenizer.special_tokens['pad']]
    
    print("\nLoading datasets...")
    train_dataset = PasswordDataset(args.train_data, tokenizer, args.max_seq_len)
    val_dataset = PasswordDataset(args.val_data, tokenizer, args.max_seq_len)
    print_memory_usage("After Datasets Initialized")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=lambda b: collate_fn(b, pad_token_id), pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=lambda b: collate_fn(b, pad_token_id), pin_memory=True
    )

    print("\nInitializing model...")
    model = PassFormer(
        vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, max_seq_len=args.max_seq_len
    )
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    print_memory_usage("After Model Initialized")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Training]")
            for src, tgt in pbar:
                src, tgt = src.to(device), tgt.to(device)
                src_padding_mask = (src == pad_token_id)
                tgt_padding_mask = (tgt == pad_token_id)
                optimizer.zero_grad()
                output = model(src, tgt, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
                loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_train_loss = train_loss / len(train_loader)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Validation]")
                for src, tgt in pbar_val:
                    src, tgt = src.to(device), tgt.to(device)
                    src_padding_mask = (src == pad_token_id)
                    tgt_padding_mask = (tgt == pad_token_id)
                    output = model(src, tgt, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
                    loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
                    val_loss += loss.item()
                    pbar_val.set_postfix({'val_loss': f"{loss.item():.4f}"})

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            print(f"\nEpoch {epoch} Summary: Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}\n")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                print(f"Validation loss improved. Saving best model to {checkpoint_path}")
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(model_state, checkpoint_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Clean up file handles
        train_dataset.close()
        val_dataset.close()
        print("--- Training Finished or Interrupted. Dataset file handles closed. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the PassFormer model.")
    parser.add_argument('--train_data', type=str, default=r'C:\Users\AIDSHPCLAB08\Desktop\dpiit-project\Passowrd_decryption_and_extractor\data\processed\stage1_train.txt')
    parser.add_argument('--val_data', type=str, default=r'C:\Users\AIDSHPCLAB08\Desktop\dpiit-project\Passowrd_decryption_and_extractor\data\processed\stage1_validation.txt')
    parser.add_argument('--tokenizer_path', type=str, default=r'C:\Users\AIDSHPCLAB08\Desktop\dpiit-project\Passowrd_decryption_and_extractor\models\tokenizer.json')
    parser.add_argument('--checkpoint_dir', type=str, default=r'C:\Users\AIDSHPCLAB08\Desktop\dpiit-project\Passowrd_decryption_and_extractor\models\checkpoints')
    parser.add_argument('--max_seq_len', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Import necessary modules from other files
    from tokenizer import Tokenizer
    from model import PassFormer
    
    args = parser.parse_args()
    train(args)

