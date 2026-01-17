#!/usr/bin/env python3
"""
8x GPU Parallel NeuCodec Encoder for 1.5B NeuTTS Model Training
V2 Chat Template - Direct text (no phonemization) - H100 optimized

IMPORTANT: Run extend_tokenizer_1.5b.py FIRST to add V2 tokens to model!

V2 CHAT TEMPLATE FORMAT:
========================
user: Convert the text to speech:<|REF_TEXT_START|>{ref_text}<|REF_TEXT_END|><|REF_SPEECH_START|>{ref_codes}<|REF_SPEECH_END|><|TARGET_TEXT_START|>{target_text}<|TARGET_TEXT_END|>
assistant:<|TARGET_CODES_START|>{target_codes}<|TARGET_CODES_END|>

LABEL MASKING:
- Mask EVERYTHING until "assistant:" with -100
- Train ONLY on: <|TARGET_CODES_START|>{target_codes}<|TARGET_CODES_END|>

OUTPUT FORMAT:
{
    "__key__": "sample_000001",
    "input_ids": [...],      # Full tokenized input sequence
    "labels": [...],         # Labels with -100 masking for non-target parts
    "text": "...",           # Target text (for reference)
    "ref_text": "..."        # Reference text (for reference)
}

Usage:
    # First extend tokenizer (one-time):
    python3 extend_tokenizer_1.5b.py --model_path pretrained_1.5b --output_path pretrained_1.5b_v2

    # Then encode data with path remapping:
    python3 encode_1.5b_v2_8gpu.py training_el_v1_chatml.json output.json \\
        --tokenizer_path pretrained_1.5b_v2 \\
        --num_gpus 8 \\
        --path_remap "/mnt/weka/home/vikram.solanki/workspace/vs=/scratch/vikram.solanki/workspace/vs"
"""

import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torchaudio
import multiprocessing as mp
import argparse
from typing import List, Tuple, Optional


# V2 Special Tokens
V2_TOKENS = [
    "<|REF_TEXT_START|>",
    "<|REF_TEXT_END|>",
    "<|REF_SPEECH_START|>",
    "<|REF_SPEECH_END|>",
    "<|TARGET_TEXT_START|>",
    "<|TARGET_TEXT_END|>",
    "<|TARGET_CODES_START|>",
    "<|TARGET_CODES_END|>",
]


def remap_path(path: str, path_remap: str) -> str:
    """
    Remap audio paths from old cluster to new cluster.
    
    Args:
        path: Original audio path
        path_remap: Remap string in format "old_prefix=new_prefix"
    
    Returns:
        Remapped path
    """
    if not path_remap:
        return path
    
    old_prefix, new_prefix = path_remap.split("=", 1)
    if path.startswith(old_prefix):
        return path.replace(old_prefix, new_prefix, 1)
    return path


def check_token_exists(tokenizer, token):
    """Check if a token exists in tokenizer vocabulary."""
    encoded = tokenizer.encode(token, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)
    return token in decoded and len(encoded) == 1


def verify_v2_tokens(tokenizer):
    """Verify all V2 tokens exist in tokenizer."""
    missing = []
    for token in V2_TOKENS:
        if not check_token_exists(tokenizer, token):
            missing.append(token)
    return missing


def extract_zero_shot_data(sample):
    """
    Extract zero-shot voice cloning data from ChatML format.
    
    Returns:
        tuple: (ref_audio_path, ref_text, target_audio_path, target_text, sample_id)
    """
    try:
        messages = sample.get("messages", [])
        
        user_msg = None
        assistant_msg = None
        
        for msg in messages:
            if msg.get("role") == "user":
                user_msg = msg
            elif msg.get("role") == "assistant":
                assistant_msg = msg
        
        if not user_msg or not assistant_msg:
            return None
        
        # Extract from user message: ref_audio, ref_text, target_text
        ref_audio_path = None
        ref_text = ""
        target_text_user = ""
        
        text_count = 0
        for item in user_msg.get("content", []):
            if item.get("type") == "audio":
                ref_audio_path = item.get("audio_url")
            elif item.get("type") == "text":
                if text_count == 0:
                    ref_text = item.get("text", "")
                else:
                    target_text_user = item.get("text", "")
                text_count += 1
        
        # Extract from assistant message: target_audio, target_text
        target_audio_path = None
        target_text = None
        
        for item in assistant_msg.get("content", []):
            if item.get("type") == "audio":
                target_audio_path = item.get("audio_url")
            elif item.get("type") == "text":
                target_text = item.get("text")
        
        misc = sample.get("misc", {})
        sample_id = misc.get("sample_id", "unknown")
        
        if not all([ref_audio_path, target_audio_path, target_text]):
            return None
        
        return (ref_audio_path, ref_text, target_audio_path, target_text, sample_id)
        
    except Exception:
        return None


def _extract_chunk(chunk, start_idx):
    """Extract zero-shot data from a chunk of samples."""
    results = []
    for i, sample in enumerate(chunk):
        idx = start_idx + i
        zr_data = extract_zero_shot_data(sample)
        if zr_data is None:
            continue
        
        ref_audio_path, ref_text, target_audio_path, target_text, sample_id = zr_data
        results.append((idx, ref_audio_path, ref_text, target_audio_path, target_text, sample_id))
    
    return results


def gpu_worker(gpu_id, samples, result_queue, path_remap='', tokenizer_path='pretrained_1.5b_v2', codec_path='neucodec_Checkpoints'):
    """GPU worker for encoding audio and tokenizing sequences with V2 template."""
    try:
        import os
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['OMP_NUM_THREADS'] = '1'
        
        import torch
        import torchaudio
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque
        from transformers import AutoTokenizer
        
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        print(f"[GPU {gpu_id}] Initializing (H100: 90 I/O threads + V2 template)...", flush=True)
        if path_remap:
            print(f"[GPU {gpu_id}] Path remap: {path_remap}", flush=True)
        
        # Load NeuCodec from local checkpoint
        try:
            from neucodec import NeuCodec
            codec = NeuCodec.from_pretrained(codec_path, local_files_only=True)
            codec = codec.eval().to(device)
            
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            print(f"[GPU {gpu_id}] NeuCodec loaded (TF32 enabled)", flush=True)
        except Exception as e:
            print(f"[GPU {gpu_id}] NeuCodec failed: {e}", flush=True)
            for idx, _, _, _, _, _ in samples:
                result_queue.put({'index': idx, 'success': False})
            return
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"[GPU {gpu_id}] Tokenizer: vocab_size={len(tokenizer):,}", flush=True)
        except Exception as e:
            print(f"[GPU {gpu_id}] Tokenizer failed: {e}", flush=True)
            for idx, _, _, _, _, _ in samples:
                result_queue.put({'index': idx, 'success': False})
            return
        
        # Verify V2 tokens exist
        missing = verify_v2_tokens(tokenizer)
        if missing:
            print(f"[GPU {gpu_id}] ERROR: Missing V2 tokens: {missing}", flush=True)
            print(f"[GPU {gpu_id}] Run extend_tokenizer_1.5b.py first!", flush=True)
            for idx, _, _, _, _, _ in samples:
                result_queue.put({'index': idx, 'success': False})
            return
        
        # Get V2 token IDs
        REF_TEXT_START = tokenizer.convert_tokens_to_ids('<|REF_TEXT_START|>')
        REF_TEXT_END = tokenizer.convert_tokens_to_ids('<|REF_TEXT_END|>')
        REF_SPEECH_START = tokenizer.convert_tokens_to_ids('<|REF_SPEECH_START|>')
        REF_SPEECH_END = tokenizer.convert_tokens_to_ids('<|REF_SPEECH_END|>')
        TARGET_TEXT_START = tokenizer.convert_tokens_to_ids('<|TARGET_TEXT_START|>')
        TARGET_TEXT_END = tokenizer.convert_tokens_to_ids('<|TARGET_TEXT_END|>')
        TARGET_CODES_START = tokenizer.convert_tokens_to_ids('<|TARGET_CODES_START|>')
        TARGET_CODES_END = tokenizer.convert_tokens_to_ids('<|TARGET_CODES_END|>')
        
        print(f"[GPU {gpu_id}] V2 Token IDs: REF_TEXT_START={REF_TEXT_START}, TARGET_CODES_END={TARGET_CODES_END}", flush=True)
        print(f"[GPU {gpu_id}] Ready - {len(samples)} samples", flush=True)
        
        encoded = 0
        failed = 0
        first_error = None
        
        stream = torch.cuda.Stream(device=device)
        
        def load_audio_fast(audio_path):
            """Fast audio loading with torchaudio."""
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            return waveform.squeeze()
        
        def load_and_prepare_batch(batch_items):
            """Load multiple audio files in parallel with path remapping."""
            loaded_data = []
            for idx, ref_audio_path, ref_text, target_audio_path, target_text, sample_id in batch_items:
                try:
                    # Apply path remapping (old cluster -> new cluster)
                    full_ref = remap_path(ref_audio_path, path_remap)
                    full_target = remap_path(target_audio_path, path_remap)
                    
                    ref_wave = load_audio_fast(full_ref)
                    target_wave = load_audio_fast(full_target)
                    loaded_data.append((idx, ref_wave, ref_text, target_wave, target_text, sample_id, True, None))
                except Exception as e:
                    import traceback
                    loaded_data.append((idx, None, ref_text, None, target_text, sample_id, False, str(e)))
            return loaded_data
        
        def build_v2_training_sample(ref_text: str, ref_codes: List[int], 
                                      target_text: str, target_codes: List[int]) -> Tuple[List[int], List[int]]:
            """
            Build pre-tokenized training sample with V2 template.
            
            V2 Format:
            user: Convert the text to speech:<|REF_TEXT_START|>{ref_text}<|REF_TEXT_END|><|REF_SPEECH_START|>{ref_codes}<|REF_SPEECH_END|><|TARGET_TEXT_START|>{target_text}<|TARGET_TEXT_END|>
            assistant:<|TARGET_CODES_START|>{target_codes}<|TARGET_CODES_END|>
            
            Labels: -100 for everything until "assistant:" 
            Train ONLY on: <|TARGET_CODES_START|>{target_codes}<|TARGET_CODES_END|>
            """
            # Tokenize text parts
            ref_text_ids = tokenizer.encode(ref_text.strip(), add_special_tokens=False)
            target_text_ids = tokenizer.encode(target_text.strip(), add_special_tokens=False)
            
            # Tokenize chat prefixes
            user_prefix = "user: Convert the text to speech:"
            user_prefix_ids = tokenizer.encode(user_prefix, add_special_tokens=False)
            
            assistant_prefix = "\nassistant:"
            assistant_prefix_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)
            
            # Convert codes to speech tokens
            ref_codes_str = "".join([f"<|speech_{c}|>" for c in ref_codes])
            target_codes_str = "".join([f"<|speech_{c}|>" for c in target_codes])
            
            ref_codes_ids = tokenizer.encode(ref_codes_str, add_special_tokens=False)
            target_codes_ids = tokenizer.encode(target_codes_str, add_special_tokens=False)
            
            # Build full input_ids with V2 template
            # user: Convert the text to speech:<|REF_TEXT_START|>{ref_text}<|REF_TEXT_END|><|REF_SPEECH_START|>{ref_codes}<|REF_SPEECH_END|><|TARGET_TEXT_START|>{target_text}<|TARGET_TEXT_END|>
            # assistant:<|TARGET_CODES_START|>{target_codes}<|TARGET_CODES_END|>
            
            input_ids = (
                user_prefix_ids +
                [REF_TEXT_START] + ref_text_ids + [REF_TEXT_END] +
                [REF_SPEECH_START] + ref_codes_ids + [REF_SPEECH_END] +
                [TARGET_TEXT_START] + target_text_ids + [TARGET_TEXT_END] +
                assistant_prefix_ids +
                [TARGET_CODES_START] + target_codes_ids + [TARGET_CODES_END]
            )
            
            # Build labels: -100 for everything EXCEPT target_codes part
            # Mask everything until and including "assistant:"
            # Train on: <|TARGET_CODES_START|>{target_codes}<|TARGET_CODES_END|>
            
            num_masked = (
                len(user_prefix_ids) +
                1 + len(ref_text_ids) + 1 +  # REF_TEXT_START, ref_text, REF_TEXT_END
                1 + len(ref_codes_ids) + 1 +  # REF_SPEECH_START, ref_codes, REF_SPEECH_END
                1 + len(target_text_ids) + 1 +  # TARGET_TEXT_START, target_text, TARGET_TEXT_END
                len(assistant_prefix_ids)  # assistant:
            )
            
            # Labels: -100 for masked, actual token IDs for training part
            labels = (
                [-100] * num_masked +
                [TARGET_CODES_START] + target_codes_ids + [TARGET_CODES_END]
            )
            
            assert len(input_ids) == len(labels), f"Length mismatch: {len(input_ids)} vs {len(labels)}"
            
            return input_ids, labels
        
        def encode_batch(loaded_batch):
            """Encode batch with CUDA stream optimization."""
            results = []
            
            valid_items = []
            for item in loaded_batch:
                idx, ref_wave, ref_text, target_wave, target_text, sample_id, success, error_msg = item
                if not success:
                    results.append((idx, None, None, None, None, None, 0, False, error_msg))
                else:
                    valid_items.append((idx, ref_wave, ref_text, target_wave, target_text, sample_id))
            
            if not valid_items:
                return results
            
            with torch.cuda.stream(stream), torch.no_grad():
                try:
                    ref_codes_batch = []
                    target_codes_batch = []
                    
                    for idx, ref_wave, ref_text, target_wave, target_text, sample_id in valid_items:
                        ref_tensor = ref_wave.unsqueeze(0).unsqueeze(0)
                        target_tensor = target_wave.unsqueeze(0).unsqueeze(0)
                        
                        ref_codes = codec.encode_code(audio_or_path=ref_tensor).squeeze(0).squeeze(0)
                        target_codes = codec.encode_code(audio_or_path=target_tensor).squeeze(0).squeeze(0)
                        
                        ref_codes_batch.append(ref_codes)
                        target_codes_batch.append(target_codes)
                    
                    stream.synchronize()
                    
                    ref_codes_lists = [c.cpu().numpy().tolist() for c in ref_codes_batch]
                    target_codes_lists = [c.cpu().numpy().tolist() for c in target_codes_batch]
                    
                    for i, (idx, _, ref_text, _, target_text, sample_id) in enumerate(valid_items):
                        try:
                            input_ids, labels = build_v2_training_sample(
                                ref_text, ref_codes_lists[i],
                                target_text, target_codes_lists[i]
                            )
                            
                            results.append((
                                idx, ref_text, target_text,
                                input_ids, labels, sample_id,
                                len(target_codes_lists[i]),
                                True, None
                            ))
                        except Exception as e:
                            results.append((idx, None, None, None, None, None, 0, False, str(e)))
                        
                except Exception as e:
                    print(f"[GPU {gpu_id}] Stream error: {str(e)[:100]}", flush=True)
                    for idx, ref_wave, ref_text, target_wave, target_text, sample_id in valid_items:
                        try:
                            with torch.no_grad():
                                ref_codes = codec.encode_code(
                                    audio_or_path=ref_wave.unsqueeze(0).unsqueeze(0)
                                ).squeeze(0).squeeze(0).cpu().numpy().tolist()
                                target_codes = codec.encode_code(
                                    audio_or_path=target_wave.unsqueeze(0).unsqueeze(0)
                                ).squeeze(0).squeeze(0).cpu().numpy().tolist()
                            
                            input_ids, labels = build_v2_training_sample(
                                ref_text, ref_codes, target_text, target_codes
                            )
                            results.append((idx, ref_text, target_text, input_ids, labels, 
                                          sample_id, len(target_codes), True, None))
                        except Exception as e2:
                            results.append((idx, None, None, None, None, None, 0, False, str(e2)))
            
            return results
        
        # Processing pipeline
        BATCH_SIZE = 32
        NUM_IO_THREADS = 90
        
        print(f"[GPU {gpu_id}] Pipeline: {NUM_IO_THREADS} threads, batch {BATCH_SIZE}", flush=True)
        
        with ThreadPoolExecutor(max_workers=NUM_IO_THREADS) as executor:
            batches = [samples[i:i+BATCH_SIZE] for i in range(0, len(samples), BATCH_SIZE)]
            
            futures_queue = deque()
            PREFETCH = min(20, len(batches))
            
            for batch in batches[:PREFETCH]:
                futures_queue.append(executor.submit(load_and_prepare_batch, batch))
            
            batch_idx = PREFETCH
            while futures_queue:
                loaded_batch = futures_queue.popleft().result()
                
                if batch_idx < len(batches):
                    futures_queue.append(executor.submit(load_and_prepare_batch, batches[batch_idx]))
                    batch_idx += 1
                
                for result in encode_batch(loaded_batch):
                    idx, ref_text, target_text, input_ids, labels, sample_id, target_len, success, err = result
                    
                    if success:
                        encoded += 1
                    else:
                        failed += 1
                        if not first_error:
                            first_error = err
                    
                    result_queue.put({
                        'index': idx,
                        'ref_text': ref_text,
                        'text': target_text,
                        'input_ids': input_ids,
                        'labels': labels,
                        'sample_id': sample_id,
                        'target_duration': target_len / 50.0 if target_len else 0,
                        'success': success
                    })
                
                if (encoded + failed) % 1000 == 0:
                    pct = (encoded + failed) / len(samples) * 100
                    print(f"[GPU {gpu_id}] {encoded + failed:,}/{len(samples):,} ({pct:.1f}%)", flush=True)
        
        print(f"[GPU {gpu_id}] DONE: {encoded:,} encoded, {failed} failed", flush=True)
        if first_error:
            print(f"[GPU {gpu_id}] Error: {first_error[:200]}", flush=True)
        
    except Exception as e:
        print(f"[GPU {gpu_id}] FATAL: {e}", flush=True)
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='8-GPU V2 Encoder for 1.5B NeuTTS')
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--tokenizer_path', type=str, default='pretrained_1.5b_v2',
                        help='Path to EXTENDED 1.5B tokenizer (with V2 tokens)')
    parser.add_argument('--codec_path', type=str, default='neucodec_Checkpoints',
                        help='Path to local NeuCodec checkpoint')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--path_remap', type=str, default='',
                        help='Remap paths: "old_prefix=new_prefix" (e.g., "/mnt/weka/home=/scratch")')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("1.5B NeuTTS V2 Encoder - 8x GPU (H100 Optimized)")
    print("=" * 80)
    print("V2 Template Format:")
    print("  user: Convert the text to speech:<|REF_TEXT_START|>...assistant:<|TARGET_CODES_START|>...")
    print("  Labels: Train ONLY on target_codes + TARGET_CODES_END")
    print("  Text: Direct text (NO phonemization)")
    print("=" * 80)
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"GPUs: {args.num_gpus} x H100 (TF32)")
    if args.path_remap:
        print(f"Path remap: {args.path_remap}")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)
    
    print(f"\nGPUs: {torch.cuda.device_count()}")
    for i in range(min(args.num_gpus, torch.cuda.device_count())):
        print(f"  {i}: {torch.cuda.get_device_name(i)}")
    
    # Verify tokenizer has V2 tokens
    print(f"\nVerifying V2 tokenizer: {args.tokenizer_path}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"  Vocab: {len(tok):,}")
    
    missing = verify_v2_tokens(tok)
    if missing:
        print(f"\n*** ERROR: Missing V2 tokens: {missing}")
        print(f"*** Run extend_tokenizer_1.5b.py first!")
        sys.exit(1)
    
    print("  V2 tokens: OK")
    for token in V2_TOKENS:
        tid = tok.convert_tokens_to_ids(token)
        print(f"    {token}: {tid}")
    
    # Load data
    print(f"\nLoading: {args.input_file}")
    chatml_data = []
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            chatml_data = json.load(f)
        print(f"Loaded {len(chatml_data):,} samples")
    except json.JSONDecodeError:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        chatml_data.append(json.loads(line))
                    except:
                        pass
        print(f"Loaded {len(chatml_data):,} (JSONL)")
    
    # Extract data
    print("\nExtracting data...")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    
    num_workers = int(multiprocessing.cpu_count() * 0.7)
    chunk_size = max(1000, len(chatml_data) // num_workers)
    
    valid_samples = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(0, len(chatml_data), chunk_size):
            futures.append(executor.submit(_extract_chunk, chatml_data[i:i+chunk_size], i))
        
        for future in as_completed(futures):
            valid_samples.extend(future.result())
    
    valid_samples.sort(key=lambda x: x[0])
    print(f"Valid: {len(valid_samples):,}")
    
    if not valid_samples:
        print("No valid samples!")
        sys.exit(1)
    
    # Distribute
    print(f"\nDistributing across {args.num_gpus} GPUs...")
    per_gpu = len(valid_samples) // args.num_gpus
    gpu_samples = []
    for i in range(args.num_gpus):
        start = i * per_gpu
        end = len(valid_samples) if i == args.num_gpus - 1 else start + per_gpu
        gpu_samples.append(valid_samples[start:end])
        print(f"  GPU {i}: {len(gpu_samples[i]):,}")
    
    # Start workers
    print(f"\nStarting workers...")
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass
    
    manager = mp.Manager()
    result_queue = manager.Queue()
    
    processes = []
    for gpu_id in range(args.num_gpus):
        p = mp.Process(target=gpu_worker, args=(
            gpu_id, gpu_samples[gpu_id], result_queue, 
            args.path_remap, args.tokenizer_path, args.codec_path
        ))
        p.start()
        processes.append(p)
    
    # Collect results
    print("\nCollecting (streaming)...\n")
    import time, signal, shutil
    
    total = len(valid_samples)
    encoded_indices = set()
    
    if args.resume and os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f:
                for s in json.load(f):
                    key = s.get('__key__', '')
                    if key.startswith('sample_'):
                        encoded_indices.add(int(key.split('_')[1]))
            print(f"[RESUME] {len(encoded_indices):,} existing\n")
        except:
            pass
    
    out = open(args.output_file, 'w', encoding='utf-8')
    out.write('[\n')
    
    all_results = []
    last_report = 0
    start_time = time.time()
    encoded_count = 0
    error_count = 0
    written = 0
    total_duration = 0.0
    interrupted = False
    
    def handler(sig, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, handler)
    
    try:
        while len(all_results) < total and not interrupted:
            try:
                result = result_queue.get(timeout=0.1)
                all_results.append(result)
                
                if result['success'] and result['input_ids']:
                    if result['index'] not in encoded_indices:
                        if written > 0:
                            out.write(',\n')
                        out.write(json.dumps({
                            "__key__": f"sample_{result['index']:06d}",
                            "input_ids": result['input_ids'],
                            "labels": result['labels'],
                            "text": result['text'].strip() if result['text'] else "",
                            "ref_text": result['ref_text'].strip() if result['ref_text'] else ""
                        }, ensure_ascii=False))
                        written += 1
                        encoded_count += 1
                        encoded_indices.add(result['index'])
                        total_duration += result.get('target_duration', 0)
                        if written % 100 == 0:
                            out.flush()
                else:
                    error_count += 1
                
                if len(all_results) - last_report >= 1000:
                    pct = len(all_results) / total * 100
                    elapsed = time.time() - start_time
                    rate = len(all_results) / elapsed
                    eta = (total - len(all_results)) / rate / 60 if rate else 0
                    hrs = total_duration / 3600
                    mb = os.path.getsize(args.output_file) / 1e6 if os.path.exists(args.output_file) else 0
                    print(f"Main: {len(all_results):,}/{total:,} ({pct:.1f}%) {rate:.1f}/s ETA:{eta:.1f}m {hrs:.2f}h {mb:.1f}MB")
                    last_report = len(all_results)
            except:
                if not any(p.is_alive() for p in processes):
                    while not result_queue.empty():
                        result = result_queue.get()
                        all_results.append(result)
                        if result['success'] and result['input_ids'] and result['index'] not in encoded_indices:
                            if written > 0:
                                out.write(',\n')
                            out.write(json.dumps({
                                "__key__": f"sample_{result['index']:06d}",
                                "input_ids": result['input_ids'],
                                "labels": result['labels'],
                                "text": result['text'].strip() if result['text'] else "",
                                "ref_text": result['ref_text'].strip() if result['ref_text'] else ""
                            }, ensure_ascii=False))
                            written += 1
                            encoded_count += 1
                            total_duration += result.get('target_duration', 0)
                    break
    except KeyboardInterrupt:
        interrupted = True
    
    out.write('\n]')
    out.close()
    
    if interrupted:
        for p in processes:
            p.terminate()
        print(f"\n[INTERRUPT] Saved {encoded_count:,} to {args.output_file}")
        print("Use --resume to continue")
        sys.exit(0)
    
    for p in processes:
        p.join()
    
    # Sort
    print(f"\nSorting {encoded_count:,}...")
    with open(args.output_file, 'r') as f:
        dataset = json.load(f)
    dataset.sort(key=lambda x: int(x['__key__'].split('_')[1]))
    
    tmp = args.output_file + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(dataset, f, ensure_ascii=False)
    shutil.move(tmp, args.output_file)
    
    total_time = time.time() - start_time
    mb = os.path.getsize(args.output_file) / 1e6
    avg_len = sum(len(s['input_ids']) for s in dataset) / len(dataset) if dataset else 0
    
    print(f"\n{'='*80}")
    print("V2 Encoding Complete!")
    print(f"{'='*80}")
    print(f"  Encoded: {encoded_count:,}")
    print(f"  Failed: {error_count:,}")
    print(f"  Time: {total_time/60:.1f}m ({len(all_results)/total_time:.1f}/s)")
    print(f"  Audio: {total_duration/3600:.2f}h")
    print(f"  Avg seq: {avg_len:.0f} tokens")
    print(f"  Output: {args.output_file} ({mb:.1f}MB)")
    print(f"{'='*80}")
    
    if dataset:
        print("\nSample (truncated):")
        s = dataset[0].copy()
        if len(s['input_ids']) > 50:
            s['input_ids'] = s['input_ids'][:20] + ['...'] + s['input_ids'][-20:]
        if len(s['labels']) > 50:
            s['labels'] = s['labels'][:20] + ['...'] + s['labels'][-20:]
        print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
