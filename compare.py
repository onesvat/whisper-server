#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import subprocess
import time
from pathlib import Path

# Updated Model List (Official + Verified Turbo)
MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "turbo",
    "distil-large-v3"
]

def get_vram_usage():
    """Get current VRAM usage in MB using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip())
    except:
        return 0

def calculate_wer(reference, hypothesis):
    """Simple Word Error Rate calculation."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if not ref_words: return 1.0
    
    # Simple edit distance logic (simplified for bench)
    import difflib
    sm = difflib.SequenceMatcher(None, ref_words, hyp_words)
    return 1.0 - sm.ratio()

def main():
    parser = argparse.ArgumentParser(description="Detailed Whisper Model Benchmark")
    parser.add_argument("--audio-dir", default="/data/benchmarks", help="Dir with .wav files")
    parser.add_argument("--expected-json", help="Path to JSON with filename:text mapping")
    parser.add_argument("--output", default="benchmark_results.csv", help="Output file")
    args = parser.parse_args()

    audio_files = sorted(Path(args.audio_dir).glob("*.wav"))
    expected_texts = {}
    if args.expected_json:
        with open(args.expected_json, "r") as f:
            expected_texts = json.load(f)

    results = []
    device = "cuda" if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0 else "cpu"

    # Test Matrix
    beam_sizes = [1, 5]
    compute_types = ["float16", "int8_float16"] if device == "cuda" else ["int8"]

    for model_alias in MODELS:
        for compute in compute_types:
            for beam in beam_sizes:
                print(f"\n>>> Testing: Model={model_alias}, Compute={compute}, Beam={beam}")
                
                # Pre-load VRAM
                vram_start = get_vram_usage()
                
                try:
                    from faster_whisper import WhisperModel
                    # We use the same logic as our server to resolve names
                    # For simplicity in this script, we'll map them manually or rely on local cache
                    # Mapping to match whisper_server/models.py
                    PREDEFINED = {
                        "tiny": "Systran/faster-whisper-tiny",
                        "base": "Systran/faster-whisper-base",
                        "small": "Systran/faster-whisper-small",
                        "medium": "Systran/faster-whisper-medium",
                        "large-v3": "Systran/faster-whisper-large-v3",
                        "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
                        "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
                    }
                    model_id = PREDEFINED.get(model_alias, model_alias)
                    
                    model = WhisperModel(model_id, device=device, compute_type=compute)
                    vram_loaded = get_vram_usage()
                    load_vram = vram_loaded - vram_start
                except Exception as e:
                    print(f"FAILED to load {model_alias}: {e}")
                    continue

                for audio_file in audio_files:
                    start_time = time.time()
                    try:
                        segments, _ = model.transcribe(str(audio_file), language="tr", beam_size=beam)
                        text = " ".join([s.text.strip() for s in segments])
                        duration = time.time() - start_time
                        
                        # Accuracy
                        wer = 0.0
                        if audio_file.name in expected_texts:
                            wer = calculate_wer(expected_texts[audio_file.name], text)
                        
                        results.append({
                            "model": model_alias,
                            "compute": compute,
                            "beam": beam,
                            "file": audio_file.name,
                            "time_sec": round(duration, 2),
                            "vram_mb": load_vram,
                            "wer": round(wer, 4),
                            "text": text
                        })
                        print(f"  {audio_file.name}: {duration:.2f}s | WER: {wer:.2%}")
                    except Exception as e:
                        print(f"  {audio_file.name}: ERROR {e}")

                # Cleanup
                del model
                gc.collect()
                if device == "cuda":
                    import torch
                    torch.cuda.empty_cache()

    # Save to CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writerow({fn: fn for fn in results[0].keys()})
        writer.writerows(results)

    print(f"\nDone! Results saved to {args.output}")

if __name__ == "__main__":
    main()
