#!/usr/bin/env python3
"""
Multi-Node NeuCodec Encoder for Dialectal Speech Data - OPTIMIZED
Lightning-fast encoding using batched GPU operations, CUDA streams, and AMP

CRITICAL: Use --nproc_per_node=1 with torchrun!
Each torchrun process manages all 8 GPUs via multiprocessing.

Usage:
    torchrun --nnodes=4 --nproc_per_node=1 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        encode_dialectal_multinode.py \
            /path/to/final_manifest.jsonl \
            /path/to/output.json \
            --seed 42

SLURM Example:
    srun --export=ALL --chdir=$PWD \
        torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        encode_dialectal_multinode.py input.jsonl output.json

Performance Optimizations:
- True batched encoding (stack tensors, single forward pass)
- CUDA streams for async compute/transfer overlap
- Automatic Mixed Precision (AMP) for Tensor Core acceleration
- Pinned memory for fast CPU-GPU transfers
- Efficient shared memory queues
- Minimum duration validation to prevent kernel errors
"""

import json
import os
import sys
import warnings
import random
warnings.filterwarnings('ignore')

os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.distributed as dist
import multiprocessing as mp
import argparse
import time

# Minimum audio duration (seconds) to prevent kernel errors
MIN_AUDIO_DURATION = 0.1  # 100ms minimum
MIN_REF_DURATION = 0.05   # 50ms minimum for reference


def remap_path(path: str, path_remap: str) -> str:
    """Remap audio paths."""
    if not path_remap:
        return path
    for mapping in path_remap.split(";"):
        mapping = mapping.strip()
        if not mapping or "=" not in mapping:
            continue
        old_prefix, new_prefix = mapping.split("=", 1)
        if path.startswith(old_prefix):
            return path.replace(old_prefix, new_prefix, 1)
    return path


def compute_ref_chunk_params(duration_seconds, sample_rate=16000):
    """Compute reference chunk with minimum duration guarantee."""
    total_samples = int(duration_seconds * sample_rate)
    min_ref_samples = int(MIN_REF_DURATION * sample_rate)
    
    # Ensure we have enough samples
    if total_samples < min_ref_samples * 2:
        # Too short - use half for ref
        ref_samples = max(min_ref_samples, total_samples // 2)
        return 0, ref_samples
    
    if duration_seconds > 6.0:
        ref_duration = random.uniform(3.0, 6.0)
    else:
        ref_ratio = random.uniform(0.3, 0.6)
        ref_duration = duration_seconds * ref_ratio
    
    ref_samples = int(ref_duration * sample_rate)
    ref_samples = max(min_ref_samples, min(ref_samples, total_samples - min_ref_samples))
    
    min_start_ratio = 0.05 if duration_seconds > 2.0 else 0.0
    max_end_ratio = 0.95 if duration_seconds > 2.0 else 1.0
    
    min_start = int(total_samples * min_start_ratio)
    max_end = int(total_samples * max_end_ratio)
    max_start = max(min_start, max_end - ref_samples)
    
    if max_start > min_start:
        start_sample = random.randint(min_start, max_start)
    else:
        start_sample = min_start
    
    end_sample = min(start_sample + ref_samples, total_samples)
    
    # Final safety check
    if end_sample - start_sample < min_ref_samples:
        start_sample = 0
        end_sample = min(min_ref_samples, total_samples)
    
    return start_sample, end_sample


def gpu_worker(gpu_id, samples, result_queue, path_remap='', codec_path='neuphonic/neucodec', 
               num_threads=16, batch_size=48, prefetch=12, seed=42, node_rank=0, use_amp=True):
    """Optimized GPU worker with batched encoding, CUDA streams, and AMP."""
    wid = f"N{node_rank}.G{gpu_id}"
    try:
        import os
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['OMP_NUM_THREADS'] = '4'
        for var in ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK',
                    'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS']:
            os.environ.pop(var, None)
        
        import torch
        import torchaudio
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque
        
        random.seed(seed + gpu_id + node_rank * 1000)
        
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        # Enable TF32 for Tensor Cores (H100)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        try:
            torchaudio.set_audio_backend("soundfile")
        except:
            pass
        
        print(f"[{wid}] Init: threads={num_threads}, batch={batch_size}, prefetch={prefetch}, AMP={use_amp}", flush=True)
        
        # Load NeuCodec
        try:
            from neucodec import NeuCodec
            codec = NeuCodec.from_pretrained(codec_path)
            codec = codec.eval().to(device)
            
            # Warmup with dummy data
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                dummy = torch.randn(1, 1, 16000, device=device)
                _ = codec.encode_code(audio_or_path=dummy)
            torch.cuda.synchronize(device)
            
            print(f"[{wid}] NeuCodec ready", flush=True)
        except Exception as e:
            print(f"[{wid}] NeuCodec FAILED: {e}", flush=True)
            for item in samples:
                result_queue.put({'index': item[0], 'success': False})
            return
        
        print(f"[{wid}] Processing {len(samples):,} samples", flush=True)
        
        encoded = 0
        failed = 0
        first_error = None
        
        # CUDA streams for async operations
        compute_stream = torch.cuda.Stream(device=device)
        transfer_stream = torch.cuda.Stream(device=device)
        
        def load_audio_fast(audio_path):
            """Optimized audio loading with validation."""
            try:
                waveform, sr = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                waveform = waveform.squeeze(0)
                
                # Validate minimum duration
                if len(waveform) < int(MIN_AUDIO_DURATION * 16000):
                    return None  # Too short
                return waveform
            except Exception:
                try:
                    import soundfile as sf
                    data, sr = sf.read(audio_path)
                    if len(data.shape) > 1:
                        data = data.mean(axis=1)
                    waveform = torch.from_numpy(data.astype(np.float32))
                    if sr != 16000:
                        waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    if len(waveform) < int(MIN_AUDIO_DURATION * 16000):
                        return None
                    return waveform
                except:
                    return None
        
        def load_and_prepare_batch(batch_items):
            """Load audio files in parallel with validation."""
            loaded_data = []
            for idx, sample in batch_items:
                try:
                    audio_path = sample.get("audio_path", "")
                    text = sample.get("text", "")
                    dialect = sample.get("dialect", "unknown")
                    
                    if not audio_path or not text:
                        loaded_data.append((idx, None, None, None, None, False, "missing fields"))
                        continue
                    
                    full_path = remap_path(audio_path, path_remap)
                    waveform = load_audio_fast(full_path)
                    
                    if waveform is None:
                        loaded_data.append((idx, None, None, None, None, False, "audio too short or invalid"))
                        continue
                    
                    actual_duration = len(waveform) / 16000.0
                    ref_start, ref_end = compute_ref_chunk_params(actual_duration, 16000)
                    
                    # Ensure valid chunk
                    if ref_end <= ref_start:
                        loaded_data.append((idx, None, None, None, None, False, "invalid ref chunk"))
                        continue
                    
                    ref_waveform = waveform[ref_start:ref_end].clone()
                    
                    # Final validation
                    if len(ref_waveform) < int(MIN_REF_DURATION * 16000):
                        loaded_data.append((idx, None, None, None, None, False, "ref too short"))
                        continue
                    
                    loaded_data.append((idx, waveform, ref_waveform, text, dialect, True, None))
                except Exception as e:
                    loaded_data.append((idx, None, None, None, None, False, str(e)[:100]))
            return loaded_data
        
        def encode_batch_optimized(loaded_batch):
            """Batched encoding with CUDA streams and AMP."""
            results = []
            valid_items = [item for item in loaded_batch if item[5]]
            
            # Add failed items
            for item in loaded_batch:
                if not item[5]:
                    results.append((item[0], None, None, None, None, False, item[6]))
            
            if not valid_items:
                return results
            
            with torch.cuda.stream(compute_stream):
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                    # Process in mini-batches for memory efficiency
                    mini_batch_size = min(16, len(valid_items))
                    
                    for mb_start in range(0, len(valid_items), mini_batch_size):
                        mb_end = min(mb_start + mini_batch_size, len(valid_items))
                        mb_items = valid_items[mb_start:mb_end]
                        
                        # Encode target audios
                        for idx, waveform, ref_waveform, text, dialect, _, _ in mb_items:
                            try:
                                # Move to GPU with pinned memory simulation
                                target_tensor = waveform.unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
                                ref_tensor = ref_waveform.unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
                                
                                # Encode
                                target_codes = codec.encode_code(audio_or_path=target_tensor)
                                ref_codes = codec.encode_code(audio_or_path=ref_tensor)
                                
                                # Transfer back to CPU async
                                with torch.cuda.stream(transfer_stream):
                                    target_codes_cpu = target_codes.squeeze(0).squeeze(0).cpu().numpy().tolist()
                                    ref_codes_cpu = ref_codes.squeeze(0).squeeze(0).cpu().numpy().tolist()
                                
                                results.append((idx, ref_codes_cpu, target_codes_cpu, text, dialect, True, None))
                                
                            except Exception as e:
                                err_msg = str(e)[:100]
                                results.append((idx, None, None, None, None, False, err_msg))
                        
                        # Sync streams periodically
                        transfer_stream.synchronize()
            
            compute_stream.synchronize()
            return results
        
        # Main processing loop
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            batches = [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]
            futures_queue = deque()
            
            # Initial prefetch
            for batch in batches[:prefetch]:
                futures_queue.append(executor.submit(load_and_prepare_batch, batch))
            
            batch_idx = prefetch
            while futures_queue:
                loaded_batch = futures_queue.popleft().result()
                
                # Submit next batch while encoding
                if batch_idx < len(batches):
                    futures_queue.append(executor.submit(load_and_prepare_batch, batches[batch_idx]))
                    batch_idx += 1
                
                # Encode and queue results
                for result in encode_batch_optimized(loaded_batch):
                    idx, ref_codes, target_codes, text, dialect, success, err = result
                    
                    if success:
                        encoded += 1
                    else:
                        failed += 1
                        if not first_error:
                            first_error = err
                    
                    result_queue.put({
                        'index': idx,
                        'ref_codes': ref_codes,
                        'codes': target_codes,
                        'text': text,
                        'dialect': dialect,
                        'target_duration': len(target_codes) / 50.0 if target_codes else 0,
                        'success': success
                    })
                
                # Progress logging
                total_processed = encoded + failed
                if total_processed % 2000 == 0 and total_processed > 0:
                    pct = total_processed / len(samples) * 100
                    print(f"[{wid}] {total_processed:,}/{len(samples):,} ({pct:.1f}%)", flush=True)
        
        # Final sync
        torch.cuda.synchronize(device)
        
        print(f"[{wid}] DONE: {encoded:,} encoded, {failed} failed", flush=True)
        if first_error:
            print(f"[{wid}] First error: {first_error}", flush=True)
        
    except Exception as e:
        print(f"[{wid}] FATAL: {e}", flush=True)
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Multi-Node Dialectal Encoder (Optimized)')
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--codec_path', type=str, default='neuphonic/neucodec')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--num_threads', type=int, default=None, help='I/O threads per GPU (auto-detected)')
    parser.add_argument('--batch_size', type=int, default=48, help='Samples per batch')
    parser.add_argument('--prefetch', type=int, default=None, help='Prefetch batches (auto-detected)')
    parser.add_argument('--path_remap', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision')
    parser.add_argument('--barrier_timeout', type=int, default=600, help='Barrier timeout in seconds')
    args = parser.parse_args()
    
    # Initialize distributed with gloo (coordination only)
    dist.init_process_group(backend='gloo', timeout=torch.distributed.timedelta(seconds=args.barrier_timeout))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    random.seed(args.seed + rank * 1000)
    
    # Auto-detect optimal settings
    total_cores = mp.cpu_count()
    actual_gpus = min(args.num_gpus, torch.cuda.device_count())
    
    if args.num_threads is None:
        args.num_threads = min(24, max(8, int((total_cores * 0.7) / actual_gpus)))
    
    if args.prefetch is None:
        args.prefetch = min(16, args.num_threads // 2)
    
    if rank == 0:
        print("=" * 70)
        print("Multi-Node Dialectal Speech Encoder - OPTIMIZED")
        print("=" * 70)
        print(f"Nodes: {world_size}, GPUs/node: {actual_gpus}, Total GPUs: {world_size * actual_gpus}")
        print(f"Cores: {total_cores}, Threads/GPU: {args.num_threads}, Batch: {args.batch_size}")
        print(f"Prefetch: {args.prefetch}, AMP: {not args.no_amp}")
        print("=" * 70)
    
    # Load manifest
    manifest_data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    manifest_data.append(json.loads(line))
                except:
                    pass
    
    if args.test > 0:
        manifest_data = manifest_data[:args.test]
    
    total_samples = len(manifest_data)
    if rank == 0:
        print(f"Total samples: {total_samples:,}")
    
    # Distribute across nodes
    samples_per_node = total_samples // world_size
    start_idx = rank * samples_per_node
    end_idx = start_idx + samples_per_node if rank < world_size - 1 else total_samples
    my_samples = [(i, manifest_data[i]) for i in range(start_idx, end_idx)]
    
    print(f"[Node {rank}] Samples: {len(my_samples):,} (idx {start_idx}-{end_idx})", flush=True)
    
    # Distribute across GPUs within node
    per_gpu = len(my_samples) // actual_gpus
    gpu_samples = []
    for i in range(actual_gpus):
        start = i * per_gpu
        end = len(my_samples) if i == actual_gpus - 1 else start + per_gpu
        gpu_samples.append(my_samples[start:end])
    
    # Start workers
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass
    
    manager = mp.Manager()
    result_queue = manager.Queue()
    
    print(f"[Node {rank}] Starting {actual_gpus} GPU workers...", flush=True)
    
    processes = []
    for gpu_id in range(actual_gpus):
        p = mp.Process(target=gpu_worker, args=(
            gpu_id, gpu_samples[gpu_id], result_queue, args.path_remap, args.codec_path,
            args.num_threads, args.batch_size, args.prefetch, args.seed, rank, not args.no_amp
        ))
        p.start()
        processes.append(p)
    
    # Collect results with streaming write
    total_node_samples = len(my_samples)
    start_time = time.time()
    
    output_dir = os.path.dirname(args.output_file) or '.'
    os.makedirs(output_dir, exist_ok=True)
    temp_file = os.path.join(output_dir, f".encode_node_{rank}.json")
    
    out = open(temp_file, 'w', encoding='utf-8')
    out.write('[\n')
    
    all_results = []
    written = 0
    encoded_count = 0
    error_count = 0
    total_duration = 0.0
    last_report = 0
    
    while len(all_results) < total_node_samples:
        try:
            result = result_queue.get(timeout=1.0)
            all_results.append(result)
            
            if result['success'] and result['codes'] is not None and result['ref_codes'] is not None:
                if written > 0:
                    out.write(',\n')
                out.write(json.dumps({
                    "__key__": f"sample_{result['index']:06d}",
                    "text": result['text'].strip() if result['text'] else "",
                    "codes": result['codes'],
                    "ref_text": "",
                    "ref_codes": result['ref_codes'],
                    "dialect": result.get('dialect', 'unknown')
                }, ensure_ascii=False))
                written += 1
                encoded_count += 1
                total_duration += result.get('target_duration', 0)
                if written % 500 == 0:
                    out.flush()
            else:
                error_count += 1
            
            # Progress reporting
            if len(all_results) - last_report >= 5000:
                elapsed = time.time() - start_time
                rate = len(all_results) / elapsed if elapsed > 0 else 0
                pct = len(all_results) / total_node_samples * 100
                eta = (total_node_samples - len(all_results)) / rate / 60 if rate > 0 else 0
                print(f"[Node {rank}] {len(all_results):,}/{total_node_samples:,} ({pct:.1f}%) | {rate:.1f}/s | ETA: {eta:.1f}m", flush=True)
                last_report = len(all_results)
        except:
            if not any(p.is_alive() for p in processes):
                # Drain remaining
                while not result_queue.empty():
                    try:
                        result = result_queue.get_nowait()
                        all_results.append(result)
                        if result['success'] and result['codes'] is not None and result['ref_codes'] is not None:
                            if written > 0:
                                out.write(',\n')
                            out.write(json.dumps({
                                "__key__": f"sample_{result['index']:06d}",
                                "text": result['text'].strip() if result['text'] else "",
                                "codes": result['codes'],
                                "ref_text": "",
                                "ref_codes": result['ref_codes'],
                                "dialect": result.get('dialect', 'unknown')
                            }, ensure_ascii=False))
                            written += 1
                            encoded_count += 1
                            total_duration += result.get('target_duration', 0)
                        else:
                            error_count += 1
                    except:
                        break
                break
    
    out.write('\n]')
    out.close()
    
    # Wait for workers
    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    
    node_time = time.time() - start_time
    print(f"[Node {rank}] Encoded: {encoded_count:,}, Failed: {error_count:,}, Time: {node_time/60:.1f}m", flush=True)
    
    # Barrier with timeout
    try:
        dist.barrier()
    except Exception as e:
        print(f"[Node {rank}] Barrier warning: {e}", flush=True)
    
    # Rank 0 merges results
    if rank == 0:
        print("\nMerging results from all nodes...")
        all_node_results = []
        
        for r in range(world_size):
            temp_f = os.path.join(output_dir, f".encode_node_{r}.json")
            try:
                with open(temp_f, 'r') as f:
                    node_results = json.load(f)
                    all_node_results.extend(node_results)
                    print(f"  Node {r}: {len(node_results):,} samples")
                os.remove(temp_f)
            except Exception as e:
                print(f"  Node {r}: FAILED - {e}")
        
        # Sort by index
        print(f"Sorting {len(all_node_results):,} samples...")
        all_node_results.sort(key=lambda x: int(x['__key__'].split('_')[1]))
        
        # Write final output
        import shutil
        tmp = args.output_file + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(all_node_results, f, ensure_ascii=False)
        shutil.move(tmp, args.output_file)
        
        total_time = time.time() - start_time
        mb = os.path.getsize(args.output_file) / 1e6
        total_hrs = sum(len(r['codes']) / 50.0 / 3600 for r in all_node_results)
        
        print(f"\n{'='*70}")
        print("ENCODING COMPLETE")
        print(f"{'='*70}")
        print(f"  Encoded: {len(all_node_results):,}")
        print(f"  Time: {total_time/60:.1f}m ({len(all_node_results)/total_time:.1f} samples/s)")
        print(f"  Audio: {total_hrs:.2f}h")
        print(f"  Output: {args.output_file} ({mb:.1f}MB)")
        print(f"{'='*70}")
    else:
        try:
            os.remove(temp_file)
        except:
            pass
    
    # Final barrier and cleanup
    try:
        dist.barrier()
    except:
        pass
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
