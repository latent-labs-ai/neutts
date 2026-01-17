#!/usr/bin/env python3
"""
Extend 1.5B NeuTTS Tokenizer with V2 + Paralinguistic Tokens

This script adds 16 new tokens to the 1.5B model:

V2 Chat Template (8 tokens):
- <|REF_TEXT_START|>, <|REF_TEXT_END|>
- <|REF_SPEECH_START|>, <|REF_SPEECH_END|>
- <|TARGET_TEXT_START|>, <|TARGET_TEXT_END|>
- <|TARGET_CODES_START|>, <|TARGET_CODES_END|>

Paralinguistic Tags (8 tokens, >10K samples each):
- [laughs], [curious], [excited], [sighs]
- [exhales], [mischievously], [whispers], [sarcastic]

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


# V2 Special Tokens (8 tokens)
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

# Paralinguistic Tokens (8 tokens) - only tags with >10K samples
PARALINGUISTIC_TOKENS = [
    "[laughs]",        # 41K samples
    "[curious]",       # 37K samples
    "[excited]",       # 29K samples
    "[sighs]",         # 26K samples
    "[exhales]",       # 17K samples
    "[mischievously]", # 16K samples
    "[whispers]",      # 13K samples
    "[sarcastic]",     # 10K samples
]

# Combined: All new tokens to add (16 total)
ALL_NEW_TOKENS = V2_SPECIAL_TOKENS + PARALINGUISTIC_TOKENS


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
    print("1.5B NeuTTS Tokenizer Extension - V2 + Paralinguistic Tokens")
    print("=" * 80)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    original_vocab_size = len(tokenizer)
    print(f"Original vocab size: {original_vocab_size:,}")
    
    # Check which tokens are missing
    print(f"\nChecking all {len(ALL_NEW_TOKENS)} tokens...")
    missing_tokens = []
    for token in ALL_NEW_TOKENS:
        exists, token_id = check_token_exists(tokenizer, token)
        status = "EXISTS" if exists else "MISSING"
        print(f"  {token:<30} {status}")
        if not exists:
            missing_tokens.append(token)
    
    if not missing_tokens:
        print("\nAll tokens already present. No extension needed.")
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
    
    for token in ALL_NEW_TOKENS:
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
    for token in ALL_NEW_TOKENS:
        token_id = tokenizer_verify.convert_tokens_to_ids(token)
        var_name = token.replace("<|", "").replace("|>", "").upper()
        print(f"{var_name} = {token_id}  # {token}")
    
    # Run tokenize/detokenize verification
    verify_tokenization(tokenizer_verify)


def verify_tokenization(tokenizer):
    """
    Verify that new tokens are properly tokenized as single tokens.
    Tests tokenize -> detokenize roundtrip with sample sentences.
    """
    print("\n" + "=" * 80)
    print("TOKENIZATION VERIFICATION")
    print("=" * 80)
    
    # Test sentences with V2 template tokens
    v2_test_sentences = [
        "<|REF_TEXT_START|>Hello world<|REF_TEXT_END|>",
        "<|REF_SPEECH_START|>audio codes here<|REF_SPEECH_END|>",
        "<|TARGET_TEXT_START|>Generate this text<|TARGET_TEXT_END|>",
        "<|TARGET_CODES_START|>1234 5678<|TARGET_CODES_END|>",
    ]
    
    # Test sentences with paralinguistic tokens
    para_test_sentences = [
        "[laughs] That was really funny!",
        "I'm [curious] about this topic.",
        "[excited] I can't wait to see it!",
        "[sighs] It's been a long day.",
        "[exhales] Let me think about that.",
        "[mischievously] I have a plan.",
        "[whispers] Don't tell anyone.",
        "[sarcastic] Oh, that's just great.",
    ]
    
    # Combined test
    combined_test = [
        "<|REF_TEXT_START|>[laughs] Hello there!<|REF_TEXT_END|>",
        "<|TARGET_TEXT_START|>[whispers] This is a secret [excited]<|TARGET_TEXT_END|>",
    ]
    
    all_tests = {
        "V2 Template Tokens": v2_test_sentences,
        "Paralinguistic Tokens": para_test_sentences,
        "Combined Tokens": combined_test,
    }
    
    all_passed = True
    
    for category, sentences in all_tests.items():
        print(f"\n{category}:")
        print("-" * 60)
        
        for sentence in sentences:
            # Tokenize
            token_ids = tokenizer.encode(sentence, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            
            # Detokenize
            decoded = tokenizer.decode(token_ids)
            
            # Check roundtrip
            roundtrip_ok = decoded.strip() == sentence.strip()
            
            # Check if special tokens are single tokens
            special_tokens_found = []
            for t in ALL_NEW_TOKENS:
                if t in sentence:
                    t_id = tokenizer.convert_tokens_to_ids(t)
                    is_single = t_id in token_ids
                    special_tokens_found.append((t, t_id, is_single))
            
            # Print results
            status = "PASS" if roundtrip_ok else "FAIL"
            if not roundtrip_ok:
                all_passed = False
            
            print(f"\nInput:    {sentence}")
            print(f"Token IDs: {token_ids}")
            print(f"Tokens:   {tokens}")
            print(f"Decoded:  {decoded}")
            print(f"Roundtrip: {status}")
            
            for t, t_id, is_single in special_tokens_found:
                single_status = "SINGLE TOKEN" if is_single else "SPLIT (BAD)"
                if not is_single:
                    all_passed = False
                print(f"  {t} -> ID {t_id}: {single_status}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL VERIFICATION TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Check output above")
    print("=" * 80)
    
    return all_passed


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
