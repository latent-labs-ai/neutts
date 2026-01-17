#!/usr/bin/env python3
"""
Extend 1.5B NeuTTS Tokenizer with V2 Special Tokens

This script adds the 8 missing V2 chat template tokens to the 1.5B model:
- <|REF_TEXT_START|>, <|REF_TEXT_END|>
- <|REF_SPEECH_START|>, <|REF_SPEECH_END|>
- <|TARGET_TEXT_START|>, <|TARGET_TEXT_END|>
- <|TARGET_CODES_START|>, <|TARGET_CODES_END|>

It also resizes the model embeddings and initializes new token embeddings.

Usage:
    python3 extend_tokenizer_1.5b.py \
        --model_path pretrained_1.5b \
        --output_path pretrained_1.5b_v2 \
        --device cuda
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')


# V2 Special Tokens to add
V2_SPECIAL_TOKENS = [
    "<|REF_TEXT_START|>",
    "<|REF_TEXT_END|>",
    "<|REF_SPEECH_START|>",
    "<|REF_SPEECH_END|>",
    "<|TARGET_TEXT_START|>",
    "<|TARGET_TEXT_END|>",
    "<|TARGET_CODES_START|>",
    "<|TARGET_CODES_END|>",
]


def check_token_exists(tokenizer, token):
    """Check if a token exists in tokenizer vocabulary."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    encoded = tokenizer.encode(token, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)
    return token in decoded and len(encoded) == 1, token_id


def extend_tokenizer_and_model(model_path: str, output_path: str, device: str = "cpu"):
    """
    Extend tokenizer with V2 tokens and resize model embeddings.
    
    Args:
        model_path: Path to original 1.5B model
        output_path: Path to save extended model
        device: Device for model loading
    """
    print("=" * 80)
    print("1.5B NeuTTS Tokenizer Extension - V2 Special Tokens")
    print("=" * 80)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    original_vocab_size = len(tokenizer)
    print(f"Original vocab size: {original_vocab_size:,}")
    
    # Check which tokens are missing
    print(f"\nChecking V2 special tokens...")
    missing_tokens = []
    for token in V2_SPECIAL_TOKENS:
        exists, token_id = check_token_exists(tokenizer, token)
        status = "EXISTS" if exists else "MISSING"
        print(f"  {token:<30} {status}")
        if not exists:
            missing_tokens.append(token)
    
    if not missing_tokens:
        print("\nAll V2 tokens already present. No extension needed.")
        return
    
    print(f"\nMissing tokens to add: {len(missing_tokens)}")
    
    # Add special tokens to tokenizer
    print(f"\nAdding {len(missing_tokens)} special tokens to tokenizer...")
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": missing_tokens
    })
    print(f"Tokens added: {num_added}")
    
    new_vocab_size = len(tokenizer)
    print(f"New vocab size: {new_vocab_size:,}")
    
    # Verify tokens were added
    print(f"\nVerifying added tokens...")
    for token in missing_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token:<30} -> ID: {token_id}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    print(f"Device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None
    )
    
    original_embed_size = model.get_input_embeddings().weight.shape[0]
    print(f"Original embedding size: {original_embed_size:,}")
    
    # Resize model embeddings
    print(f"\nResizing model embeddings...")
    model.resize_token_embeddings(new_vocab_size)
    
    new_embed_size = model.get_input_embeddings().weight.shape[0]
    print(f"New embedding size: {new_embed_size:,}")
    
    # Initialize new token embeddings
    print(f"\nInitializing new token embeddings...")
    
    # Get the embedding and lm_head weights
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    
    # Calculate mean of existing embeddings for initialization
    # Use mean of special token embeddings for better initialization
    special_token_ids = [
        tokenizer.convert_tokens_to_ids("<|im_start|>"),
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    ]
    special_embeddings = input_embeddings[special_token_ids]
    mean_embedding = special_embeddings.mean(dim=0)
    
    # Initialize new tokens with mean embedding + small noise
    for token in missing_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        # Add small random noise for diversity
        noise = torch.randn_like(mean_embedding) * 0.01
        input_embeddings[token_id] = mean_embedding + noise
        output_embeddings[token_id] = mean_embedding + noise
        print(f"  Initialized {token} (ID: {token_id})")
    
    # Save extended model and tokenizer
    print(f"\nSaving extended model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_path)
    print(f"  Model saved")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    print(f"  Tokenizer saved")
    
    # Final verification
    print(f"\nFinal verification...")
    tokenizer_verify = AutoTokenizer.from_pretrained(output_path)
    print(f"  Loaded tokenizer vocab size: {len(tokenizer_verify):,}")
    
    for token in V2_SPECIAL_TOKENS:
        exists, token_id = check_token_exists(tokenizer_verify, token)
        status = "OK" if exists else "FAILED"
        print(f"  {token:<30} ID: {token_id:<10} {status}")
    
    print(f"\n" + "=" * 80)
    print("Extension Complete!")
    print("=" * 80)
    print(f"Original vocab: {original_vocab_size:,}")
    print(f"New vocab: {new_vocab_size:,}")
    print(f"Added tokens: {num_added}")
    print(f"Output path: {output_path}")
    print("=" * 80)
    
    # Print token ID summary for encoding script
    print(f"\nV2 Token IDs (for encoding script):")
    print("-" * 50)
    for token in V2_SPECIAL_TOKENS:
        token_id = tokenizer_verify.convert_tokens_to_ids(token)
        var_name = token.replace("<|", "").replace("|>", "").upper()
        print(f"{var_name} = {token_id}  # {token}")


def main():
    parser = argparse.ArgumentParser(description="Extend 1.5B tokenizer with V2 tokens")
    parser.add_argument("--model_path", type=str, default="pretrained_1.5b",
                        help="Path to original 1.5B model")
    parser.add_argument("--output_path", type=str, default="pretrained_1.5b_v2",
                        help="Path to save extended model")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "auto"],
                        help="Device for model loading")
    
    args = parser.parse_args()
    
    extend_tokenizer_and_model(
        model_path=args.model_path,
        output_path=args.output_path,
        device=args.device
    )


if __name__ == "__main__":
    main()
