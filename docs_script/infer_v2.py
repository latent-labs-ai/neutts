#!/usr/bin/env python3
"""
V2 Template Inference Script for Zero-Shot Voice Cloning
Uses NEW V2 template format with explicit delimiters

NEW V2 TEMPLATE FORMAT:
  User turn: <|REF_TEXT_START|>{ref_phones}<|REF_TEXT_END|>
             <|REF_SPEECH_START|>{ref_codes}<|REF_SPEECH_END|>
             <|TARGET_TEXT_START|>{target_phones}<|TARGET_TEXT_END|>
  
  Assistant turn: <|TARGET_CODES_START|>{target_codes}<|TARGET_CODES_END|>

Usage:
    python3 infer_v2.py --speaker Ivy --text "Hello, how are you today?"
    python3 infer_v2.py --speaker Priyanka --text "This is a test of the V2 template."
    
    # Batch generation
    python3 infer_v2.py --speakers Ivy,Priyanka --texts "Test 1" "Test 2" "Test 3"
"""

import os
import sys
import torch
import soundfile as sf
import re
import random
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import time
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from neuttsair import NeuTTSAir
except ImportError:
    print("Error: neuttsair module not found. Please install it first.")
    sys.exit(1)


class V2TemplateInference:
    """
    V2 Template Inference with explicit delimiters.
    
    This uses the NEW training format with 8 special delimiter tokens:
    - <|REF_TEXT_START|>, <|REF_TEXT_END|>
    - <|REF_SPEECH_START|>, <|REF_SPEECH_END|>
    - <|TARGET_TEXT_START|>, <|TARGET_TEXT_END|>
    - <|TARGET_CODES_START|>, <|TARGET_CODES_END|>
    """
    
    def __init__(
        self,
        model_path: str,
        codec_path: str = "neuphonic/neucodec",
        device: str = "cuda",
        speaker_info_path: str = "speaker_info.json"
    ):
        """Initialize V2 inference pipeline."""
        self.device = device
        self.speaker_info_path = speaker_info_path
        self.performance_metrics = []
        
        print(f"\n{'='*80}")
        print(f"V2 Template Voice Cloning Inference Pipeline")
        print(f"{'='*80}")
        print(f"Model: {model_path}")
        print(f"Device: {device}")
        print(f"Template: V2 with explicit delimiters")
        
        # Load speaker information
        self.speaker_info = self.load_speaker_info(speaker_info_path)
        print(f"Speakers: {len(self.speaker_info)} loaded")
        
        # Check GPU
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load NeuTTSAir
        print(f"\nInitializing NeuTTSAir pipeline...")
        self.tts = NeuTTSAir(
            backbone_repo=model_path,
            backbone_device=device,
            codec_repo=codec_path,
            codec_device=device
        )
        print("✓ NeuTTSAir pipeline loaded")
        
        self.tokenizer = self.tts.tokenizer
        self.model = self.tts.backbone
        self.model.eval()
        
        # Verify V2 special tokens exist
        print(f"\nVerifying V2 special tokens...")
        v2_tokens = [
            '<|REF_TEXT_START|>', '<|REF_TEXT_END|>',
            '<|REF_SPEECH_START|>', '<|REF_SPEECH_END|>',
            '<|TARGET_TEXT_START|>', '<|TARGET_TEXT_END|>',
            '<|TARGET_CODES_START|>', '<|TARGET_CODES_END|>'
        ]
        
        for token in v2_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                raise ValueError(f"V2 token not found: {token}. Model may not be trained with V2 format!")
        
        print(f"✓ All 8 V2 delimiter tokens verified")
        print(f"✓ Vocabulary size: {len(self.tokenizer):,}")
        
        # Get special token IDs
        self.target_codes_start = self.tokenizer.convert_tokens_to_ids('<|TARGET_CODES_START|>')
        self.target_codes_end = self.tokenizer.convert_tokens_to_ids('<|TARGET_CODES_END|>')
        
        print(f"{'='*80}\n")
    
    def load_speaker_info(self, speaker_info_path: str) -> dict:
        """Load speaker information from JSON."""
        if not os.path.exists(speaker_info_path):
            raise FileNotFoundError(f"Speaker info not found: {speaker_info_path}")
        
        with open(speaker_info_path, 'r', encoding='utf-8') as f:
            speaker_info = json.load(f)
        
        # Validate
        for speaker_name, info in speaker_info.items():
            required_fields = ['speaker', 'speaker_audio_path', 'speaker_text']
            for field in required_fields:
                if field not in info:
                    raise ValueError(f"Missing field '{field}' for speaker '{speaker_name}'")
            
            if not os.path.exists(info['speaker_audio_path']):
                raise FileNotFoundError(f"Speaker audio not found: {info['speaker_audio_path']}")
            if not os.path.exists(info['speaker_text']):
                raise FileNotFoundError(f"Speaker text not found: {info['speaker_text']}")
        
        return speaker_info
    
    def get_speaker_info(self, speaker_name: str) -> dict:
        """Get speaker information."""
        if speaker_name not in self.speaker_info:
            available = ', '.join(list(self.speaker_info.keys())[:10])
            raise ValueError(f"Speaker '{speaker_name}' not found. Available: {available}...")
        return self.speaker_info[speaker_name]
    
    def encode_reference(self, speaker_name: str) -> tuple:
        """Encode reference audio and get phonemes."""
        speaker_info = self.get_speaker_info(speaker_name)
        audio_path = speaker_info['speaker_audio_path']
        phonemized_text_path = speaker_info['speaker_text']
        
        # Get reference codes
        ref_codes_tensor = self.tts.encode_reference(audio_path)
        if hasattr(ref_codes_tensor, 'tolist'):
            ref_codes = ref_codes_tensor.tolist()
        else:
            ref_codes = [int(x) for x in ref_codes_tensor]
        
        # Get reference phonemes
        with open(phonemized_text_path, 'r', encoding='utf-8') as f:
            ref_phonemes = f.read().strip()
        
        return ref_codes, ref_phonemes
    
    def build_v2_prompt(
        self,
        target_phonemes: str,
        ref_phonemes: str,
        ref_codes: List[int]
    ) -> str:
        """
        Build V2 template prompt with explicit delimiters.
        
        V2 FORMAT:
        user: Convert the text to speech:
        <|REF_TEXT_START|>{ref_phones}<|REF_TEXT_END|>
        <|REF_SPEECH_START|>{ref_codes}<|REF_SPEECH_END|>
        <|TARGET_TEXT_START|>{target_phones}<|TARGET_TEXT_END|>
        assistant:<|TARGET_CODES_START|>
        """
        # Convert ref codes to speech tokens
        ref_codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        
        # Build V2 prompt with explicit delimiters
        user_content = (
            f"<|REF_TEXT_START|>{ref_phonemes}<|REF_TEXT_END|>"
            f"<|REF_SPEECH_START|>{ref_codes_str}<|REF_SPEECH_END|>"
            f"<|TARGET_TEXT_START|>{target_phonemes}<|TARGET_TEXT_END|>"
        )
        
        # Assistant turn starts generation with TARGET_CODES_START delimiter
        prompt = f"user: Convert the text to speech:{user_content}\nassistant:<|TARGET_CODES_START|>"
        
        return prompt
    
    def generate_codes(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 40
    ) -> tuple:
        """Generate speech codes with V2 template."""
        start_time = time.perf_counter()
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Move to device
        if hasattr(self.model, 'device'):
            input_ids = input_ids.to(self.model.device)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            input_ids = input_ids.to(self.model.model.embed_tokens.weight.device)
        else:
            input_ids = input_ids.to(self.device)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.target_codes_end,  # Stop at </TARGET_CODES_END>
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=False
            )
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        generation_time = time.perf_counter() - start_time
        
        # Extract codes
        generated_ids = outputs.sequences[0][input_ids.shape[1]:].cpu().tolist()
        
        speech_codes = []
        for token_id in generated_ids:
            token = self.tokenizer.decode([token_id])
            if token.startswith('<|speech_') and token.endswith('|>'):
                try:
                    code = int(token[9:-2])
                    speech_codes.append(code)
                except ValueError:
                    continue
            elif token_id == self.target_codes_end:
                break
        
        metrics = {
            'generation_time': generation_time,
            'output_tokens': len(generated_ids),
            'speech_codes_count': len(speech_codes),
            'tokens_per_second': len(generated_ids) / generation_time if generation_time > 0 else 0
        }
        
        return speech_codes, metrics
    
    def codes_to_audio(self, codes: List[int]):
        """Decode speech codes to audio."""
        codes_str = "".join([f"<|speech_{i}|>" for i in codes])
        audio = self.tts._decode(codes_str)
        return audio
    
    def synthesize(
        self,
        text: str,
        speaker_name: str,
        output_path: str,
        phonemized_text: Optional[str] = None,
        **generation_kwargs
    ) -> dict:
        """
        Synthesize speech with V2 template.
        
        Args:
            text: Original text (for logging)
            speaker_name: Speaker name
            output_path: Output file path
            phonemized_text: Pre-phonemized text (REQUIRED for V2)
            **generation_kwargs: Generation parameters
        """
        start_total = time.perf_counter()
        
        if not phonemized_text:
            raise ValueError("phonemized_text is required for V2 inference")
        
        # Encode reference
        start_encode = time.perf_counter()
        ref_codes, ref_phonemes = self.encode_reference(speaker_name)
        encode_time = time.perf_counter() - start_encode
        
        # Build V2 prompt
        prompt = self.build_v2_prompt(phonemized_text, ref_phonemes, ref_codes)
        
        # Generate codes
        generated_codes, gen_metrics = self.generate_codes(prompt, **generation_kwargs)
        
        if len(generated_codes) == 0:
            raise ValueError("Model generated 0 speech codes")
        
        # Decode to audio
        start_decode = time.perf_counter()
        audio = self.codes_to_audio(generated_codes)
        decode_time = time.perf_counter() - start_decode
        
        # Save audio
        sample_rate = 24000
        sf.write(output_path, audio, sample_rate)
        
        total_time = time.perf_counter() - start_total
        
        # Metrics
        metrics = {
            'speaker': speaker_name,
            'text': text,
            'output_file': output_path,
            'audio_duration': len(audio) / sample_rate,
            'encode_time': encode_time,
            'generation_time': gen_metrics['generation_time'],
            'decode_time': decode_time,
            'total_time': total_time,
            'tokens_per_second': gen_metrics['tokens_per_second'],
            'speech_codes_count': gen_metrics['speech_codes_count'],
            'rtf': total_time / (len(audio) / sample_rate)
        }
        
        self.performance_metrics.append(metrics)
        return metrics


# Pattern to match special tags like [laugh], [excited], [whispers], [exhales], [sighs], etc.
SPECIAL_TAG_PATTERN = re.compile(r'(\[[a-zA-Z_\s]+\])')


def extract_special_tags(text: str) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Extract special tags like [laugh], [exhales], [sighs] from text.
    
    Args:
        text: Input text with potential special tags
        
    Returns:
        Tuple of (text_without_tags, list of (tag, position) tuples)
    """
    tags = []
    
    # Find all special tags and their positions
    for match in SPECIAL_TAG_PATTERN.finditer(text):
        tag = match.group(1)
        position = match.start()
        tags.append((tag, position))
    
    # Remove tags from text for phonemization
    clean_text = SPECIAL_TAG_PATTERN.sub('', text)
    
    return clean_text, tags


def restore_special_tags(phonemized_text: str, original_text: str, 
                        tags: List[Tuple[str, int]]) -> str:
    """
    Restore special tags to phonemized text at appropriate positions.
    
    Args:
        phonemized_text: Phonemized text without tags
        original_text: Original text with tags
        tags: List of (tag, position) tuples from extract_special_tags
        
    Returns:
        Phonemized text with special tags restored
    """
    if not tags:
        return phonemized_text
    
    # Remove tags from original to get clean original
    clean_original = SPECIAL_TAG_PATTERN.sub('', original_text)
    
    # Build result by inserting tags at relative positions
    result = phonemized_text
    
    # Sort tags by position (reverse order for insertion)
    sorted_tags = sorted(tags, key=lambda x: x[1], reverse=True)
    
    for tag, orig_pos in sorted_tags:
        # Find the position in clean original text (accounting for previously removed tags)
        adjusted_pos = orig_pos
        for other_tag, other_pos in tags:
            if other_pos < orig_pos:
                adjusted_pos -= len(other_tag)
        
        # Calculate relative position (0.0 to 1.0)
        if len(clean_original.strip()) > 0:
            relative_pos = adjusted_pos / len(clean_original)
        else:
            relative_pos = 0.0
        
        # Find insertion point in phonemized text
        insert_pos = int(relative_pos * len(result))
        
        # Try to insert at word boundary (space)
        if insert_pos < len(result):
            # Find nearest space
            space_before = result.rfind(' ', 0, insert_pos)
            space_after = result.find(' ', insert_pos)
            
            if space_before == -1 and space_after == -1:
                # No spaces, insert at calculated position
                pass
            elif space_before == -1:
                insert_pos = space_after
            elif space_after == -1:
                insert_pos = space_before + 1
            else:
                # Choose closest space
                if (insert_pos - space_before) < (space_after - insert_pos):
                    insert_pos = space_before + 1
                else:
                    insert_pos = space_after
        
        # Insert tag with space
        result = result[:insert_pos] + ' ' + tag + ' ' + result[insert_pos:]
    
    # Clean up multiple spaces
    result = re.sub(r' +', ' ', result).strip()
    
    return result


def phonemize_text(text: str, language: str = "en") -> str:
    """
    Phonemize text using EspeakBackend from phonemizer library.
    EXACTLY matches training phonemization from pre_phonemize_dataset.py
    
    Preserves special tags like [laugh], [exhales], [sighs], [whispers].
    
    Configuration matches training:
    - preserve_punctuation=True
    - with_stress=True (includes ˈ primary and ˌ secondary stress markers)
    - words_mismatch="ignore"
    - language_switch="remove-flags"
    
    Args:
        text: Text to phonemize
        language: Language code ('en' for English, 'ar' for Arabic)
        
    Returns:
        Phonemized text string (IPA format with stress markers and special tags preserved)
    """
    try:
        from phonemizer.backend import EspeakBackend
    except ImportError:
        print(f"ERROR: phonemizer not installed! Run: pip install phonemizer")
        return text
    
    if not text.strip():
        return text
    
    # Extract special tags before phonemization
    clean_text, special_tags = extract_special_tags(text)
    
    if not clean_text.strip():
        # Only tags, no text to phonemize
        return text
    
    # Map language codes to espeak language codes
    lang_map = {
        'en': 'en-us',
        'ar': 'ar'
    }
    espeak_lang = lang_map.get(language, language)
    
    try:
        # Create backend with EXACT training configuration
        backend = EspeakBackend(
            language=espeak_lang,
            preserve_punctuation=True,      # Keep punctuation for expressiveness
            with_stress=True,                # Include stress markers (ˈ, ˌ) - CRITICAL!
            words_mismatch="ignore",         # Ignore word mismatches
            language_switch="remove-flags"   # Clean language switches
        )
        
        # Phonemize clean text (without tags)
        phones = backend.phonemize([clean_text])
        
        if not phones or len(phones) == 0:
            print(f"Warning: Empty phoneme output for: {text[:50]}...")
            return text
        
        # Process output: split and rejoin (preserves stress markers and punctuation)
        # This matches pre_phonemize_dataset.py line 127-128
        phones_processed = phones[0].split()
        phones_final = ' '.join(phones_processed)
        
        # Restore special tags
        if special_tags:
            phones_final = restore_special_tags(phones_final, text, special_tags)
        
        return phones_final
        
    except Exception as e:
        print(f"Warning: Phonemization failed ({e}), using raw text")
        return text


def main():
    parser = argparse.ArgumentParser(description="V2 Template Voice Cloning Inference")
    
    # Speaker and text
    parser.add_argument("--speaker", type=str, help="Speaker name (e.g., Ivy, Priyanka)")
    parser.add_argument("--speakers", type=str, help="Comma-separated speaker names")
    parser.add_argument("--text", type=str, help="Single text to synthesize")
    parser.add_argument("--texts", nargs="+", help="Multiple texts to synthesize")
    parser.add_argument("--text_file", type=str, help="Text file with one sentence per line (will be auto-phonemized)")
    parser.add_argument("--phonemized_text_file", type=str, help="Pre-phonemized text file (one phoneme sequence per line, skips phonemization)")
    
    # Paths
    parser.add_argument("--model_path", type=str, default="checkpoint-990000", help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, default="results_v6", help="Output directory")
    parser.add_argument("--speaker_info_path", type=str, default="speaker_info_3.json")
    parser.add_argument("--codec_path", type=str, default="neuphonic/neucodec")
    
    # Generation params
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    
    # Language
    parser.add_argument("--language", type=str, default="en", choices=["en", "ar"])
    
    args = parser.parse_args()
    
    # Parse speakers
    if args.speakers:
        speaker_names = [s.strip() for s in args.speakers.split(',')]
    elif args.speaker:
        speaker_names = [args.speaker]
    else:
        # Default test speakers
        speaker_names = ["Ivy", "Priyanka"]
        print(f"No speakers specified, using defaults: {', '.join(speaker_names)}")
    
    # Parse texts
    phonemized_texts = None  # Will store pre-phonemized text if provided
    
    if args.phonemized_text_file:
        # Load pre-phonemized texts from file
        with open(args.phonemized_text_file, 'r', encoding='utf-8') as f:
            phonemized_texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(phonemized_texts)} pre-phonemized texts from: {args.phonemized_text_file}")
        # Use same texts for display (will be replaced with originals if available)
        texts = phonemized_texts.copy()
    elif args.text_file:
        # Load texts from file (one per line)
        with open(args.text_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(texts)} texts from: {args.text_file}")
        
        # Shuffle speakers for each text line (one speaker per line)
        if len(texts) > 0:
            # Create shuffled speaker assignments (cycle through speakers randomly)
            random.shuffle(speaker_names)  # Shuffle speaker order
            speaker_assignments = []
            for i in range(len(texts)):
                speaker_assignments.append(speaker_names[i % len(speaker_names)])
            
            # Shuffle assignments again for more randomness
            random.shuffle(speaker_assignments)
            
            print(f"\n📝 Text-to-Speaker Assignments (shuffled):")
            for i, (text, speaker) in enumerate(zip(texts, speaker_assignments), 1):
                print(f"  Line {i}: {speaker:12s} → {text[:60]}...")
            
            # Override speaker_names with assignments
            # We'll use this later in generation loop
            text_speaker_map = list(zip(texts, speaker_assignments))
        else:
            text_speaker_map = None
    elif args.texts:
        texts = args.texts
        text_speaker_map = None
    elif args.text:
        texts = [args.text]
        text_speaker_map = None
    else:
        # Default test texts
        texts = [
            "Hello, this is a test of the new V2 template format.",
            "The weather is beautiful today, don't you think?",
            "I really enjoy testing new voice cloning models."
        ]
        text_speaker_map = None
        print(f"No texts specified, using {len(texts)} default test texts")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = V2TemplateInference(
        model_path=args.model_path,
        codec_path=args.codec_path,
        device=args.device,
        speaker_info_path=args.speaker_info_path
    )
    
    # Generation parameters
    gen_params = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'max_new_tokens': args.max_new_tokens
    }
    
    print(f"\n{'='*80}")
    print(f"V2 TEMPLATE BATCH GENERATION")
    print(f"{'='*80}")
    
    # Check if we're using text file with shuffled assignments
    if args.text_file and text_speaker_map:
        print(f"Mode: One line per speaker (shuffled)")
        print(f"Texts: {len(text_speaker_map)} samples")
        print(f"Total generations: {len(text_speaker_map)}")
    else:
        print(f"Mode: All speakers × all texts")
        print(f"Speakers: {', '.join(speaker_names)}")
        print(f"Texts: {len(texts)} samples")
        print(f"Total generations: {len(speaker_names) * len(texts)}")
    
    print(f"Parameters: temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"{'='*80}\n")
    
    total_start = time.perf_counter()
    success_count = 0
    total_count = 0
    
    # Generate based on mode
    if args.text_file and text_speaker_map:
        # ONE LINE PER SPEAKER MODE (shuffled)
        for text_idx, (text, speaker) in enumerate(text_speaker_map, 1):
            total_count += 1
            
            print(f"\n{'='*80}")
            print(f"Sample [{text_idx}/{len(text_speaker_map)}]")
            print(f"Speaker: {speaker}")
            print(f"Text: {text[:80]}...")
            print(f"{'='*80}")
            
            # Auto-detect language for ALL text sources
            detected_lang = args.language
            
            # Detect language based on Arabic Unicode range
            if any('\u0600' <= c <= '\u06FF' for c in text):
                detected_lang = 'ar'
            else:
                detected_lang = 'en'
            
            print(f"  Language: {detected_lang}")
            
            # Phonemize using EspeakBackend (with special tag preservation)
            phonemized_text = phonemize_text(text, detected_lang)
            print(f"  Phonemes: {phonemized_text[:80]}...")
            
            # Output filename
            output_filename = f"{speaker.lower()}_line_{text_idx:03d}.wav"
            output_path = os.path.join(args.output_dir, output_filename)
            
            # Text file
            text_filename = f"{speaker.lower()}_line_{text_idx:03d}.txt"
            text_path = os.path.join(args.output_dir, text_filename)
            
            try:
                # Save text
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # Generate audio
                metrics = pipeline.synthesize(
                    text=text,
                    speaker_name=speaker,
                    output_path=output_path,
                    phonemized_text=phonemized_text,
                    **gen_params
                )
                
                success_count += 1
                print(f"✓ SUCCESS: {output_filename}")
                print(f"  Audio Duration: {metrics['audio_duration']:.2f}s")
                print(f"  Generation Time: {metrics['generation_time']:.3f}s")
                print(f"  Tokens/sec: {metrics['tokens_per_second']:.1f}")
                print(f"  RTF: {metrics['rtf']:.3f}")
                
            except Exception as e:
                print(f"✗ FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
    
    else:
        # TRADITIONAL MODE: All speakers × all texts
        for speaker_idx, speaker in enumerate(speaker_names, 1):
            print(f"\n{'='*80}")
            print(f"SPEAKER [{speaker_idx}/{len(speaker_names)}]: {speaker}")
            print(f"{'='*80}")
            
            for text_idx, text in enumerate(texts, 1):
                total_count += 1
                
                print(f"\n[{text_idx}/{len(texts)}] Text: {text[:80]}...")
                
                # Check if we have pre-phonemized text
                if phonemized_texts and text_idx <= len(phonemized_texts):
                    # Use pre-phonemized text directly (no language detection needed)
                    phonemized_text = phonemized_texts[text_idx - 1]
                    print(f"  Using pre-phonemized text")
                    print(f"  Phonemes: {phonemized_text[:80]}...")
                else:
                    # Auto-detect language
                    detected_lang = args.language
                    
                    # Detect language based on Arabic Unicode range
                    if any('\u0600' <= c <= '\u06FF' for c in text):
                        detected_lang = 'ar'
                    else:
                        detected_lang = 'en'
                    
                    print(f"  Detected language: {detected_lang}")
                    
                    # Phonemize using EspeakBackend (with special tag preservation)
                    phonemized_text = phonemize_text(text, detected_lang)
                    print(f"  Phonemes: {phonemized_text[:80]}...")
                
                # Output filename
                output_filename = f"{speaker.lower()}_sample_{text_idx:03d}.wav"
                output_path = os.path.join(args.output_dir, output_filename)
                
                # Text file
                text_filename = f"{speaker.lower()}_sample_{text_idx:03d}.txt"
                text_path = os.path.join(args.output_dir, text_filename)
                
                try:
                    # Save text
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    # Generate audio
                    metrics = pipeline.synthesize(
                        text=text,
                        speaker_name=speaker,
                        output_path=output_path,
                        phonemized_text=phonemized_text,
                        **gen_params
                    )
                    
                    success_count += 1
                    print(f"✓ SUCCESS: {output_filename}")
                    print(f"  Audio Duration: {metrics['audio_duration']:.2f}s")
                    print(f"  Generation Time: {metrics['generation_time']:.3f}s")
                    print(f"  Tokens/sec: {metrics['tokens_per_second']:.1f}")
                    print(f"  RTF: {metrics['rtf']:.3f}")
                    
                except Exception as e:
                    print(f"✗ FAILED: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    total_time = time.perf_counter() - total_start
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, "performance_metrics_v2.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': vars(args),
            'metrics': pipeline.performance_metrics,
            'summary': {
                'total_samples': total_count,
                'successful': success_count,
                'failed': total_count - success_count,
                'total_time': total_time,
                'avg_time_per_sample': total_time / max(success_count, 1)
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total Samples: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Success Rate: {(success_count/total_count*100):.1f}%")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Time/Sample: {total_time/max(success_count, 1):.2f}s")
    
    if pipeline.performance_metrics:
        avg_tps = sum(m['tokens_per_second'] for m in pipeline.performance_metrics) / len(pipeline.performance_metrics)
        avg_rtf = sum(m['rtf'] for m in pipeline.performance_metrics) / len(pipeline.performance_metrics)
        
        print(f"\nPerformance Metrics:")
        print(f"  Avg Tokens/sec: {avg_tps:.1f}")
        print(f"  Avg RTF: {avg_rtf:.3f}")
    
    print(f"\nOutput Directory: {args.output_dir}")
    print(f"Metrics saved to: {metrics_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
