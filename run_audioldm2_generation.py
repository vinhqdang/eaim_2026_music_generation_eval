"""
Generate 300 audio samples using AudioLDM2 for comparison with MusicGen.
Uses latent diffusion model (different from MusicGen's autoregressive approach).
"""
import torch
from audioldm2 import text_to_audio, build_model
import scipy.io.wavfile as wavfile
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm

def generate_audioldm2_full(num_prompts=100, num_seeds=3):
    """Generate full AudioLDM2 dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("="*60)
    print(f"AUDIOLDM2 GENERATION: {num_prompts} prompts × {num_seeds} seeds")
    print("="*60)

    # Load model
    print("\nLoading AudioLDM2-music...")
    audioldm2 = build_model(model_name="audioldm2-music")
    audioldm2 = audioldm2.to(device)
    print("✓ Model loaded")

    # Load prompts
    with open("data/prompts/prompts_text.json") as f:
        prompts_data = json.load(f)[:num_prompts]

    output_dir = Path("runs/artifacts/wav/audioldm2_full")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = num_prompts * num_seeds
    print(f"\nGenerating {total_samples} samples...")

    results = []
    start_time = time.time()

    with tqdm(total=total_samples, desc="AudioLDM2 Generation") as pbar:
        for prompt_data in prompts_data:
            for seed in range(num_seeds):
                try:
                    # Set seed
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)

                    # Generate
                    waveform = text_to_audio(
                        audioldm2,
                        text=prompt_data['text'],
                        seed=seed,
                        duration=10.0,
                        guidance_scale=3.5,
                        n_candidate_gen_per_text=1,
                        device=device
                    )

                    # Save
                    audio_np = waveform[0]  # First candidate
                    audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)
                    audio_int16 = (audio_np * 32767).astype(np.int16)

                    output_path = output_dir / f"{prompt_data['id']}_seed{seed:02d}.wav"
                    wavfile.write(output_path, 16000, audio_int16)

                    # Metadata
                    metadata = {
                        "model": "audioldm2-music",
                        "prompt_id": prompt_data['id'],
                        "prompt": prompt_data['text'],
                        "seed": seed,
                        "genre": prompt_data.get('genre'),
                        "bpm": prompt_data.get('bpm'),
                        "device": device,
                        "architecture": "latent_diffusion"
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
    print(f"AUDIOLDM2 COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{total_samples}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Avg: {elapsed/successful:.1f}s per sample" if successful > 0 else "")
    print(f"Output: {output_dir}")

    return results

if __name__ == "__main__":
    print("="*60)
    print("AUDIOLDM2 GENERATION FOR MODEL COMPARISON")
    print("="*60)
    print("\nModel: AudioLDM2-music (latent diffusion)")
    print("vs MusicGen-small (autoregressive transformer)")
    print("\nTarget: 300 audio samples")
    print("Estimated time: 60-90 minutes on GPU")
    print()

    results = generate_audioldm2_full(num_prompts=100, num_seeds=3)

    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Success: {sum(1 for r in results if r.get('success'))}/300")
    print("\nNext steps:")
    print("1. Run: python compute_full_metrics.py")
    print("2. Run: python generate_full_analysis.py")
