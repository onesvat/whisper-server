#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import subprocess
import time
from pathlib import Path

# Defaults
DEFAULT_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo"]
DEFAULT_BEAMS = [1, 2, 3, 4, 5]
DEFAULT_COMPUTE = ["float16", "int8_float16"]

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
    if not reference: return 0.0
    ref_words = reference.lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "").split()
    hyp_words = hypothesis.lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "").split()
    if not ref_words: return 1.0
    import difflib
    sm = difflib.SequenceMatcher(None, ref_words, hyp_words)
    return 1.0 - sm.ratio()

def main():
    parser = argparse.ArgumentParser(description="Customizable Whisper Model Benchmark")
    parser.add_argument("--audio-dir", required=True, help="Dir with .wav files")
    parser.add_argument("--expected-json", help="Expected text JSON")
    parser.add_argument("--output", default="results.csv", help="Output file")
    parser.add_argument("--language", default="tr", help="Language code (default: tr)")
    parser.add_argument("--download-dir", default="/data/models", help="Persistent model storage")
    parser.add_argument("--local-files-only", action="store_true", help="Don't check HF Hub for updates")
    
    # Selection arguments
    parser.add_argument("--models", help=f"Comma-separated models")
    parser.add_argument("--beams", help=f"Comma-separated beam sizes")
    parser.add_argument("--compute-types", help=f"Comma-separated compute types")

    args = parser.parse_args()

    # Parse selections
    models_to_test = args.models.split(",") if args.models else DEFAULT_MODELS
    beams_to_test = [int(b) for b in args.beams.split(",")] if args.beams else DEFAULT_BEAMS
    compute_to_test = args.compute_types.split(",") if args.compute_types else DEFAULT_COMPUTE

    audio_files = sorted(Path(args.audio_dir).glob("*.wav"))
    if not audio_files:
        print(f"Error: No .wav files found in {args.audio_dir}")
        return

    expected_texts = {}
    if args.expected_json and Path(args.expected_json).exists():
        with open(args.expected_json, "r") as f:
            expected_texts = json.load(f)

    results = []
    try:
        device = "cuda" if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0 else "cpu"
    except:
        device = "cpu"

    for model_alias in models_to_test:
        for compute in compute_to_test:
            vram_initial = get_vram_usage()
            
            try:
                from faster_whisper import WhisperModel
                PREDEFINED = {
                    "tiny": "Systran/faster-whisper-tiny",
                    "base": "Systran/faster-whisper-base",
                    "small": "Systran/faster-whisper-small",
                    "medium": "Systran/faster-whisper-medium",
                    "large-v1": "Systran/faster-whisper-large-v1",
                    "large-v2": "Systran/faster-whisper-large-v2",
                    "large-v3": "Systran/faster-whisper-large-v3",
                    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
                }
                model_id = PREDEFINED.get(model_alias, model_alias)
                print(f"\n>>> Loading {model_alias} ({compute})...")
                
                # Forced local files if flag is set
                model = WhisperModel(
                    model_id, 
                    device=device, 
                    compute_type=compute, 
                    download_root=args.download_dir,
                    local_files_only=args.local_files_only
                )
                vram_loaded = get_vram_usage()
            except Exception as e:
                print(f"  FAILED to load {model_alias}: {e}")
                continue

            for beam in beams_to_test:
                print(f"  Beam Size: {beam}")
                peak_vram = vram_loaded

                for audio_file in audio_files:
                    start_time = time.time()
                    try:
                        segments, _ = model.transcribe(str(audio_file), language=args.language, beam_size=beam, vad_filter=True)
                        text = " ".join([s.text.strip() for s in segments])
                        duration = time.time() - start_time
                        
                        current_vram = get_vram_usage()
                        if current_vram > peak_vram: peak_vram = current_vram
                        
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
                        print(f"    {audio_file.name}: {int(duration*1000)}ms | WER: {wer:.1%} | '{text[:50]}...'")
                    except Exception as e:
                        print(f"    ERROR transcribing {audio_file.name}: {e}")

            del model
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

    if results:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nBenchmark finished. Results saved to {args.output}")

if __name__ == "__main__":
    main()
