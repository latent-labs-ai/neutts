#!/usr/bin/env python3
"""
Multi-Node NeuCodec Encoder for Dialectal Speech Data
Uses same architecture as encode_4b_v2_8gpu.py but for multi-node

Each node runs independently with 8 GPUs using multiprocessing.
torchrun is used ONLY for coordination - actual encoding uses the proven
GPU worker pattern from encode_4b_v2_8gpu.py.

Usage:
    torchrun --nnodes=4 --nproc_per_node=1 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        encode_dialectal_multinode.py \
            /path/to/final_manifest.jsonl \
            /path/to/output.json \
            --seed 42
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
    """Compute reference chunk for zero-shot voice cloning."""
    total_samples = int(duration_seconds * sample_rate)
    
    if duration_seconds > 6.0:
        ref_duration = random.uniform(3.0, 6.0)
    else:
        ref_ratio = random.uniform(0.3, 0.6)
        ref_duration = duration_seconds * ref_ratio
    
    ref_samples = int(ref_duration * sample_rate)
    ref_samples = min(ref_samples, total_samples - 1)
    
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
    return start_sample, end_sample


def gpu_worker(gpu_id, samples, result_queue, path_remap='', codec_path='neuphonic/neucodec', 
               num_threads=16, batch_size=32, prefetch=8, seed=42, node_rank=0):
    """GPU worker - same pattern as encode_4b_v2_8gpu.py"""
    wid = f"N{node_rank}.G{gpu_id}"  # Worker ID for logging
    try:
        import os
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['OMP_NUM_THREADS'] = '4'
        # Disable distributed in workers
        for var in ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK',
                    'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS']:
            os.environ.pop(var, None)
        
        import torch
        import torchaudio
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque
        
        # Seed per GPU
        random.seed(seed + gpu_id)
        
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        try:
            torchaudio.set_audio_backend("soundfile")
        except:
            pass
        
        print(f"[{wid}] Init: {num_threads} threads, batch={batch_size}", flush=True)
        
        # Load NeuCodec
        try:
            from neucodec import NeuCodec
            codec = NeuCodec.from_pretrained(codec_path)
            codec = codec.eval().to(device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"[{wid}] NeuCodec loaded", flush=True)
        except Exception as e:
            print(f"[{wid}] NeuCodec FAILED: {e}", flush=True)
            for item in samples:
                result_queue.put({'index': item[0], 'success': False})
            return
        
        print(f"[{wid}] Ready - {len(samples):,} samples", flush=True)
        
        # Print first sample path for debugging
        if samples:
            first_path = samples[0][1].get("audio_path", "")
            if path_remap:
                first_path = remap_path(first_path, path_remap)
            print(f"[{wid}] First: {first_path}", flush=True)
            print(f"[{wid}] Exists: {os.path.exists(first_path)}", flush=True)
        
        encoded = 0
        failed = 0
        first_error = None
        
        def load_audio_fast(audio_path):
            try:
                waveform, sr = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                return waveform.squeeze(0)
            except Exception:
                import soundfile as sf
                data, sr = sf.read(audio_path)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                waveform = torch.from_numpy(data.astype(np.float32))
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                return waveform
        
        def load_and_prepare_batch(batch_items):
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
                    actual_duration = len(waveform) / 16000.0
                    
                    ref_start, ref_end = compute_ref_chunk_params(actual_duration, 16000)
                    ref_waveform = waveform[ref_start:ref_end].clone()
                    
                    loaded_data.append((idx, waveform, ref_waveform, text, dialect, True, None))
                except Exception as e:
                    loaded_data.append((idx, None, None, None, None, False, str(e)))
            return loaded_data
        
        def encode_batch(loaded_batch):
            results = []
            valid_items = [item for item in loaded_batch if item[5]]
            
            for item in loaded_batch:
                if not item[5]:
                    results.append((item[0], None, None, None, None, False, item[6]))
            
            if not valid_items:
                return results
            
            with torch.no_grad():
                for idx, waveform, ref_waveform, text, dialect, _, _ in valid_items:
                    try:
                        target_codes = codec.encode_code(
                            audio_or_path=waveform.unsqueeze(0).unsqueeze(0)
                        ).squeeze(0).squeeze(0).cpu().numpy().tolist()
                        
                        ref_codes = codec.encode_code(
                            audio_or_path=ref_waveform.unsqueeze(0).unsqueeze(0)
                        ).squeeze(0).squeeze(0).cpu().numpy().tolist()
                        
                        results.append((idx, ref_codes, target_codes, text, dialect, True, None))
                    except Exception as e:
                        results.append((idx, None, None, None, None, False, str(e)))
            
            return results
        
        # Main processing loop
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            batches = [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]
            futures_queue = deque()
            
            for batch in batches[:prefetch]:
                futures_queue.append(executor.submit(load_and_prepare_batch, batch))
            
            batch_idx = prefetch
            while futures_queue:
                loaded_batch = futures_queue.popleft().result()
                
                if batch_idx < len(batches):
                    futures_queue.append(executor.submit(load_and_prepare_batch, batches[batch_idx]))
                    batch_idx += 1
                
                for result in encode_batch(loaded_batch):
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
                
                if (encoded + failed) % 1000 == 0 and (encoded + failed) > 0:
                    print(f"[{wid}] {encoded + failed:,}/{len(samples):,}", flush=True)
        
        print(f"[{wid}] DONE: {encoded:,} encoded, {failed} failed", flush=True)
        if first_error:
            print(f"[{wid}] First error: {first_error[:150]}", flush=True)
        
    except Exception as e:
        print(f"[{wid}] FATAL: {e}", flush=True)
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--codec_path', type=str, default='neuphonic/neucodec')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--num_threads', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--prefetch', type=int, default=8)
    parser.add_argument('--path_remap', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test', type=int, default=0)
    args = parser.parse_args()
    
    # Initialize distributed (1 process per node)
    dist.init_process_group(backend='gloo')  # Use gloo for coordination only
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    random.seed(args.seed + rank * 1000)
    
    if rank == 0:
        print("=" * 70)
        print("Multi-Node Dialectal Speech Encoder")
        print(f"Nodes: {world_size}, GPUs per node: {args.num_gpus}")
        print(f"Total GPUs: {world_size * args.num_gpus}")
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
    
    print(f"[Node {rank}] Samples: {len(my_samples):,} ({start_idx} to {end_idx})", flush=True)
    
    # Distribute across GPUs within this node
    actual_gpus = min(args.num_gpus, torch.cuda.device_count())
    per_gpu = len(my_samples) // actual_gpus
    gpu_samples = []
    for i in range(actual_gpus):
        start = i * per_gpu
        end = len(my_samples) if i == actual_gpus - 1 else start + per_gpu
        gpu_samples.append(my_samples[start:end])
    
    # Start GPU workers using multiprocessing
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
            args.num_threads, args.batch_size, args.prefetch, args.seed + rank * 1000, rank
        ))
        p.start()
        processes.append(p)
    
    # Collect results with streaming write (like encode_4b_v2_8gpu.py)
    total_node_samples = len(my_samples)
    start_time = time.time()
    
    # Write to temp file with streaming
    output_dir = os.path.dirname(args.output_file)
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
                if written % 100 == 0:
                    out.flush()
            else:
                error_count += 1
            
            if len(all_results) - last_report >= 5000:
                elapsed = time.time() - start_time
                rate = len(all_results) / elapsed if elapsed > 0 else 0
                print(f"[Node {rank}] {len(all_results):,}/{total_node_samples:,} | {rate:.1f}/s | {encoded_count:,} encoded", flush=True)
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
    
    for p in processes:
        p.join()
    
    print(f"[Node {rank}] Encoded: {encoded_count:,}, Failed: {error_count:,}", flush=True)
    
    dist.barrier()
    
    # Rank 0 merges and sorts
    if rank == 0:
        print("\nMerging and sorting results from all nodes...")
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
        
        # Sort by sample index
        print(f"Sorting {len(all_node_results):,} samples...")
        all_node_results.sort(key=lambda x: int(x['__key__'].split('_')[1]))
        
        # Write sorted results
        import shutil
        tmp = args.output_file + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(all_node_results, f, ensure_ascii=False)
        shutil.move(tmp, args.output_file)
        
        total_time = time.time() - start_time
        mb = os.path.getsize(args.output_file) / 1e6
        total_duration = sum(len(r['codes']) / 50.0 for r in all_node_results)
        
        print(f"\n{'='*70}")
        print("ENCODING COMPLETE")
        print(f"{'='*70}")
        print(f"  Encoded: {len(all_node_results):,}")
        print(f"  Time: {total_time/60:.1f}m ({len(all_node_results)/total_time:.1f} samples/s)")
        print(f"  Audio: {total_duration/3600:.2f}h")
        print(f"  Output: {args.output_file} ({mb:.1f}MB)")
        print(f"{'='*70}")
    else:
        try:
            os.remove(temp_file)
        except:
            pass
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
