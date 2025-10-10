"""
Generate 300 audio samples using Stable Audio Open for comparison with MusicGen.
Uses diffusion-based architecture (different from MusicGen's autoregressive approach).
"""
import torch
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import scipy.io.wavfile as wavfile
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm

def generate_stableaudio_full(num_prompts=100, num_seeds=3):
    """Generate full Stable Audio dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("="*60)
    print(f"STABLE AUDIO GENERATION: {num_prompts} prompts × {num_seeds} seeds")
    print("="*60)

    # Load model
    print("\nLoading Stable Audio Open 1.0...")
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model.to(device)
    print("✓ Model loaded")

    # Load prompts
    with open("data/prompts/prompts_text.json") as f:
        prompts_data = json.load(f)[:num_prompts]

    output_dir = Path("runs/artifacts/wav/stableaudio_full")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = num_prompts * num_seeds
    print(f"\nGenerating {total_samples} samples...")

    results = []
    start_time = time.time()

    with tqdm(total=total_samples, desc="Stable Audio Generation") as pbar:
        for prompt_data in prompts_data:
            for seed in range(num_seeds):
                try:
                    # Set seed
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                    # Generate
                    output = generate_diffusion_cond(
                        model,
                        steps=100,
                        cfg_scale=7,
                        conditioning={
                            "prompt": prompt_data['text'],
                            "seconds_start": 0,
                            "seconds_total": 10
                        },
                        sample_size=480000,  # 10s at 48kHz
                        sigma_min=0.3,
                        sigma_max=500,
                        sampler_type="dpmpp-3m-sde",
                        device=device,
                        seed=seed
                    )

                    # Save
                    sampling_rate = 48000
                    audio_np = output[0, 0].cpu().numpy()
                    audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)
                    audio_int16 = (audio_np * 32767).astype(np.int16)

                    output_path = output_dir / f"{prompt_data['id']}_seed{seed:02d}.wav"
                    wavfile.write(output_path, sampling_rate, audio_int16)

                    # Metadata
                    metadata = {
                        "model": "stable-audio-open",
                        "prompt_id": prompt_data['id'],
                        "prompt": prompt_data['text'],
                        "seed": seed,
                        "genre": prompt_data.get('genre'),
                        "bpm": prompt_data.get('bpm'),
                        "device": device,
                        "architecture": "diffusion"
                    }

                    metadata_path = output_dir / f"{prompt_data['id']}_seed{seed:02d}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    results.append({"success": True, "file": str(output_path)})

                except Exception as e:
                    print(f"\nError on {prompt_data['id']}_seed{seed:02d}: {e}")
                    results.append({"success": False, "error": str(e)})

                pbar.update(1)

    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r.get('success'))

    print(f"\n{'='*60}")
    print(f"STABLE AUDIO COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{total_samples}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Avg: {elapsed/successful:.1f}s per sample" if successful > 0 else "")
    print(f"Output: {output_dir}")

    return results

if __name__ == "__main__":
    print("="*60)
    print("STABLE AUDIO GENERATION FOR MODEL COMPARISON")
    print("="*60)
    print("\nModel: Stable Audio Open 1.0 (diffusion-based)")
    print("vs MusicGen-small (autoregressive transformer)")
    print("\nTarget: 300 audio samples")
    print("Estimated time: 60-90 minutes on GPU")
    print()

    results = generate_stableaudio_full(num_prompts=100, num_seeds=3)

    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Success: {sum(1 for r in results if r.get('success'))}/300")
    print("\nNext steps:")
    print("1. Run: python compute_full_metrics.py")
    print("2. Run: python generate_full_analysis.py")
