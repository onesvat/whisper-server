#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import subprocess
import time
from pathlib import Path

# Benchmarking models only (Removing Distil for this test)
MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "turbo"
]

def get_vram_usage():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip())
    except:
        return 0

def calculate_wer(reference, hypothesis):
    # Normalize: lower case, remove punctuation
    ref_words = reference.lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "").split()
    hyp_words = hypothesis.lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "").split()
    if not ref_words: return 1.0
    import difflib
    sm = difflib.SequenceMatcher(None, ref_words, hyp_words)
    return 1.0 - sm.ratio()

def main():
    parser = argparse.ArgumentParser(description="Exhaustive Whisper Model Benchmark")
    parser.add_argument("--audio-dir", default="/data/benchmarks", help="Dir with .wav files")
    parser.add_argument("--expected-json", default="/app/benchmark_data.json", help="Expected text JSON")
    parser.add_argument("--output", default="/data/benchmarks/results.csv", help="Output file")
    parser.add_argument("--language", default="tr", help="Language code (tr)")
    args = parser.parse_args()

    audio_files = sorted(Path(args.audio_dir).glob("*.wav"))
    expected_texts = {}
    if Path(args.expected_json).exists():
        with open(args.expected_json, "r") as f:
            expected_texts = json.load(f)

    results = []
    try:
        device = "cuda" if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0 else "cpu"
    except:
        device = "cpu"

    # Testing all beam sizes 1-5 and standard compute types
    beam_sizes = [1, 2, 3, 4, 5]
    compute_types = ["float16", "int8_float16"] if device == "cuda" else ["int8"]

    for model_alias in MODELS:
        for compute in compute_types:
            vram_initial = get_vram_usage()
            
            try:
                from faster_whisper import WhisperModel
                PREDEFINED = {
                    "tiny": "Systran/faster-whisper-tiny",
                    "base": "Systran/faster-whisper-base",
                    "small": "Systran/faster-whisper-small",
                    "medium": "Systran/faster-whisper-medium",
                    "large-v2": "Systran/faster-whisper-large-v2",
                    "large-v3": "Systran/faster-whisper-large-v3",
                    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
                }
                model_id = PREDEFINED.get(model_alias, model_alias)
                model = WhisperModel(model_id, device=device, compute_type=compute)
                vram_loaded = get_vram_usage()
            except Exception as e:
                print(f"FAILED {model_alias}: {e}")
                continue

            for beam in beam_sizes:
                print(f"\n>>> Testing: Model={model_alias}, Compute={compute}, Beam={beam}, VAD=True")
                peak_vram = vram_loaded

                for audio_file in audio_files:
                    start_time = time.time()
                    try:
                        segments, _ = model.transcribe(
                            str(audio_file), 
                            language=args.language, 
                            beam_size=beam,
                            vad_filter=True
                        )
                        text = " ".join([s.text.strip() for s in segments])
                        duration = time.time() - start_time
                        
                        current_vram = get_vram_usage()
                        if current_vram > peak_vram:
                            peak_vram = current_vram
                        
                        wer = calculate_wer(expected_texts.get(audio_file.name, ""), text)
                        
                        results.append({
                            "model": model_alias,
                            "compute": compute,
                            "beam": beam,
                            "file": audio_file.name,
                            "time_ms": int(duration * 1000),
                            "vram_load_mb": vram_loaded - vram_initial if vram_loaded > 0 else 0,
                            "vram_peak_mb": peak_vram - vram_initial if peak_vram > 0 else 0,
                            "wer": round(wer, 4),
                            "text": text
                        })
                        print(f"  {audio_file.name}: {int(duration*1000)}ms | WER: {wer:.1%}")
                    except Exception as e:
                        print(f"  {audio_file.name}: ERROR {e}")

            del model
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

    if not results: return

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBenchmark finished. Results saved to {args.output}")

if __name__ == "__main__":
    main()
