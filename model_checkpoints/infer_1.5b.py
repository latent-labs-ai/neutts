#!/usr/bin/env python3
"""
Inference Script for 1.5B NeuTTS Model (Direct Text - No Phonemization)

Uses original NeuTTS template format with special tokens.
This model is trained on direct text, not phonemized text.

Usage:
    python3 infer_1.5b.py --speaker Adeline --text "Hello, how are you?"
    python3 infer_1.5b.py --speaker Ivy --text "مرحبا كيف حالك؟"
"""

import os
import sys
import torch
import soundfile as sf
import json
import argparse
import time
from pathlib import Path
from typing import List, Optional
import librosa
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from neucodec import NeuCodec


class NeuTTS15B:
    """
    1.5B NeuTTS Inference (direct text, no phonemization).
    """
    
    def __init__(
        self,
        model_path: str,
        codec_path: str = "neuphonic/neucodec",
        device: str = "cuda",
        speaker_info_path: str = "speaker_info_1.5b.json"
    ):
        self.device = device
        self.speaker_info_path = speaker_info_path
        self.sample_rate = 24000
        
        print(f"\n{'='*60}")
        print(f"1.5B NeuTTS Inference (Direct Text)")
        print(f"{'='*60}")
        print(f"Model: {model_path}")
        print(f"Device: {device}")
        
        # Load speaker info
        self.speaker_info = self._load_speaker_info(speaker_info_path)
        print(f"Speakers: {len(self.speaker_info)} loaded")
        
        # Check device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        if device == "mps" and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            self.device = "cpu"
        
        # Load tokenizer and model
        print(f"\nLoading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32
        ).to(self.device)
        self.model.eval()
        print(f"Model loaded: {self.model.config.hidden_size}d, {self.model.config.num_hidden_layers}L")
        print(f"Vocab size: {len(self.tokenizer):,}")
        
        # Verify special tokens
        self._verify_tokens()
        
        # Load codec
        print(f"\nLoading NeuCodec...")
        self.codec = NeuCodec.from_pretrained(codec_path)
        self.codec.eval().to(self.device if self.device != "mps" else "cpu")
        print(f"NeuCodec loaded")
        
        # Get special token IDs
        self.speech_gen_start = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
        self.speech_gen_end = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        self.text_prompt_start = self.tokenizer.convert_tokens_to_ids('<|TEXT_PROMPT_START|>')
        self.text_prompt_end = self.tokenizer.convert_tokens_to_ids('<|TEXT_PROMPT_END|>')
        
        print(f"{'='*60}\n")
    
    def _load_speaker_info(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Speaker info not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _verify_tokens(self):
        tokens = [
            '<|TEXT_PROMPT_START|>', '<|TEXT_PROMPT_END|>',
            '<|SPEECH_GENERATION_START|>', '<|SPEECH_GENERATION_END|>',
            '<|speech_0|>', '<|speech_1|>'
        ]
        for token in tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Token not found: {token}")
        print(f"Special tokens verified")
    
    def encode_reference(self, audio_path: str) -> List[int]:
        """Encode reference audio to speech codes."""
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
        codec_device = self.device if self.device != "mps" else "cpu"
        wav_tensor = wav_tensor.to(codec_device)
        with torch.no_grad():
            codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return codes.cpu().tolist()
    
    def build_prompt(self, ref_text: str, ref_codes: List[int], target_text: str) -> List[int]:
        """Build prompt using NeuTTS template format (direct text)."""
        # Combine reference text and target text
        input_text = ref_text + " " + target_text
        
        # Get token IDs
        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        
        # Encode input text
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        
        # Build chat template
        chat = "user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"
        ids = self.tokenizer.encode(chat)
        
        # Replace TEXT_REPLACE with actual text
        text_replace_idx = ids.index(text_replace)
        ids = (
            ids[:text_replace_idx]
            + [self.text_prompt_start]
            + input_ids
            + [self.text_prompt_end]
            + ids[text_replace_idx + 1:]
        )
        
        # Replace SPEECH_REPLACE with reference codes
        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes_ids = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [self.speech_gen_start] + codes_ids
        
        return ids
    
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 4096,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 40
    ) -> List[int]:
        """Generate speech codes from prompt."""
        input_ids = torch.tensor(prompt_ids).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.speech_gen_end,
                use_cache=True
            )
        
        generated_ids = outputs[0][input_ids.shape[1]:].cpu().tolist()
        
        speech_codes = []
        for token_id in generated_ids:
            token = self.tokenizer.decode([token_id])
            if token.startswith('<|speech_') and token.endswith('|>'):
                try:
                    code = int(token[9:-2])
                    speech_codes.append(code)
                except ValueError:
                    continue
            elif token_id == self.speech_gen_end:
                break
        
        return speech_codes
    
    def decode_codes(self, codes: List[int]) -> np.ndarray:
        """Decode speech codes to audio waveform."""
        codec_device = self.device if self.device != "mps" else "cpu"
        codes_tensor = torch.tensor(codes, dtype=torch.long)[None, None, :].to(codec_device)
        with torch.no_grad():
            audio = self.codec.decode_code(codes_tensor).cpu().numpy()
        return audio[0, 0, :]
    
    def synthesize(
        self,
        text: str,
        speaker_name: str,
        output_path: str,
        **gen_kwargs
    ) -> dict:
        """Synthesize speech from text using specified speaker."""
        start = time.perf_counter()
        
        if speaker_name not in self.speaker_info:
            raise ValueError(f"Speaker '{speaker_name}' not found")
        
        info = self.speaker_info[speaker_name]
        
        # Get reference text
        with open(info['speaker_text'], 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()
        
        # Encode reference audio
        ref_codes = self.encode_reference(info['speaker_audio_path'])
        
        # Build prompt with direct text (returns token ids)
        prompt_ids = self.build_prompt(ref_text, ref_codes, text)
        
        # Generate
        gen_start = time.perf_counter()
        speech_codes = self.generate(prompt_ids, **gen_kwargs)
        gen_time = time.perf_counter() - gen_start
        
        if len(speech_codes) == 0:
            raise ValueError("No speech codes generated")
        
        # Decode
        audio = self.decode_codes(speech_codes)
        
        # Save
        sf.write(output_path, audio, self.sample_rate)
        
        total_time = time.perf_counter() - start
        duration = len(audio) / self.sample_rate
        
        return {
            'speaker': speaker_name,
            'text': text,
            'output': output_path,
            'duration': duration,
            'gen_time': gen_time,
            'total_time': total_time,
            'codes': len(speech_codes),
            'rtf': total_time / duration
        }


def main():
    parser = argparse.ArgumentParser(description="1.5B NeuTTS Inference")
    parser.add_argument("--model_path", type=str, default="pretrained_1.5b")
    parser.add_argument("--codec_path", type=str, default="neuphonic/neucodec")
    parser.add_argument("--speaker_info_path", type=str, default="speaker_info_1.5b.json")
    parser.add_argument("--speaker", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_1.5b")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=40)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize
    tts = NeuTTS15B(
        model_path=args.model_path,
        codec_path=args.codec_path,
        device=args.device,
        speaker_info_path=args.speaker_info_path
    )
    
    # Generate
    timestamp = int(time.time())
    output_path = os.path.join(args.output_dir, f"{args.speaker.lower()}_{timestamp}.wav")
    
    print(f"Generating: {args.text[:60]}...")
    metrics = tts.synthesize(
        text=args.text,
        speaker_name=args.speaker,
        output_path=output_path,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    print(f"\nResult:")
    print(f"  Output: {metrics['output']}")
    print(f"  Duration: {metrics['duration']:.2f}s")
    print(f"  Gen Time: {metrics['gen_time']:.2f}s")
    print(f"  RTF: {metrics['rtf']:.3f}")


if __name__ == "__main__":
    main()
