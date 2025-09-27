# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import argparse
from tokenizer import Tokenizer

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings so that they can be summed.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PassFormer(nn.Module):
    """
    An Encoder-Decoder Transformer model for password generation.
    It's an autoregressive model that predicts the next character in a sequence.
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_seq_len: int = 256):
        super().__init__()
        
        self.d_model = d_model
        
        # Token Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        
        # Core Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: This makes handling batches easier
        )
        
        # Final output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Special masks to prevent the model from "cheating" by looking ahead
        self.tgt_mask = None
        self.max_seq_len = max_seq_len

    def _generate_square_subsequent_mask(self, sz):
        """Generates a square mask for the sequence. Used by the decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_padding_mask: torch.Tensor = None, tgt_padding_mask: torch.Tensor = None):
        """
        Forward pass of the model.
        Args:
            src: the sequence to the encoder (batch_size, src_len)
            tgt: the sequence to the decoder (batch_size, tgt_len)
            src_padding_mask: mask for padding tokens in the source
            tgt_padding_mask: mask for padding tokens in the target
        """
        # Embed and add positional encoding
        src_embed = self.embedding(src) * math.sqrt(self.d_model)
        tgt_embed = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Note: PyTorch Transformer expects [seq_len, batch_size, dim] by default if batch_first=False
        # Since we set batch_first=True, we can keep it as [batch_size, seq_len, dim]
        src_embed = self.pos_encoder(src_embed.transpose(0,1)).transpose(0,1)
        tgt_embed = self.pos_encoder(tgt_embed.transpose(0,1)).transpose(0,1)

        # Generate target mask for the decoder
        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt[0]):
            device = tgt.device
            self.tgt_mask = self._generate_square_subsequent_mask(len(tgt[0])).to(device)

        # Pass through the transformer
        output = self.transformer(
            src_embed, 
            tgt_embed, 
            tgt_mask=self.tgt_mask, 
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Final linear layer to get logits
        return self.fc_out(output)

if __name__ == '__main__':
    # --- A quick test to verify the model architecture and dimensions ---
    print("--- Verifying PassFormer Model Architecture ---")
    
    # 1. Load the tokenizer to get vocab size
    # This assumes you have run tokenizer.py and the file exists
    try:
        tokenizer = Tokenizer.load(r'C:\Users\AIDSHPCLAB08\Desktop\dpiit-project\Passowrd_decryption_and_extractor\models/tokenizer.json')
        VOCAB_SIZE = tokenizer.vocab_size
        print(f"Loaded tokenizer with a vocabulary size of: {VOCAB_SIZE}")
    except FileNotFoundError:
        print("Tokenizer file not found. Using a default vocab size of 100 for test.")
        VOCAB_SIZE = 100

    # 2. Model Hyperparameters (These can be tuned later)
    D_MODEL = 512       # Embedding dimension
    NHEAD = 8           # Number of attention heads
    NUM_ENCODER_LAYERS = 3 # Number of encoder layers
    NUM_DECODER_LAYERS = 3 # Number of decoder layers
    
    # 3. Instantiate the model
    model = PassFormer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS
    )

    # 4. Create some dummy data to test the forward pass
    BATCH_SIZE = 4
    SEQ_LENGTH = 30
    # Simulate a batch of tokenized passwords
    src_tensor = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    tgt_tensor = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))

    print(f"\nInput tensor shape (batch): {src_tensor.shape}")

    # 5. Run the model
    try:
        output = model(src=src_tensor, tgt=tgt_tensor)
        print(f"Output tensor shape (batch): {output.shape}")
        print("Model forward pass successful!")
        
        # Verify output shape -> (batch_size, seq_length, vocab_size)
        assert output.shape == (BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
        print("Output shape is correct.")

    except Exception as e:
        print(f"An error occurred during the model test: {e}")
