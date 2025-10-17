"""
Generate 300 audio samples using MusicGen-large for comparison.
Largest MusicGen model (~3.3GB) - best quality.
"""
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm

def generate_musicgen_large(num_prompts=100, num_seeds=3):
    """Generate full MusicGen-large dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("="*60)
    print(f"MUSICGEN-LARGE GENERATION: {num_prompts} prompts × {num_seeds} seeds")
    print("="*60)

    # Load model
    print("\nLoading MusicGen-large...")
    model_name = "facebook/musicgen-large"
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name, use_safetensors=True)
    model.to(device)
    print("✓ Model loaded")

    # Load prompts
    with open("data/prompts/prompts_text.json") as f:
        prompts_data = json.load(f)[:num_prompts]

    output_dir = Path("runs/artifacts/wav/musicgen_large")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = num_prompts * num_seeds
    print(f"\nGenerating {total_samples} samples...")

    results = []
    start_time = time.time()

    with tqdm(total=total_samples, desc="MusicGen-Large Generation") as pbar:
        for prompt_data in prompts_data:
            for seed in range(num_seeds):
                try:
                    # Set seed
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                    # Process
                    inputs = processor(
                        text=[prompt_data['text']],
                        padding=True,
                        return_tensors="pt",
                    ).to(device)

                    # Generate
                    with torch.no_grad():
                        audio_values = model.generate(
                            **inputs,
                            max_new_tokens=500,
                            do_sample=True,
                            temperature=1.0,
                            guidance_scale=3.0,
                        )

                    # Save
                    sampling_rate = model.config.audio_encoder.sampling_rate
                    audio_np = audio_values[0].cpu().numpy()
                    audio_np = np.squeeze(audio_np)
                    audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)
                    audio_int16 = (audio_np * 32767).astype(np.int16)

                    output_path = output_dir / f"{prompt_data['id']}_seed{seed:02d}.wav"
                    wavfile.write(output_path, sampling_rate, audio_int16)

                    # Metadata
                    metadata = {
                        "model": "musicgen-large",
                        "prompt_id": prompt_data['id'],
                        "prompt": prompt_data['text'],
                        "seed": seed,
                        "genre": prompt_data.get('genre'),
                        "bpm": prompt_data.get('bpm'),
                        "device": device,
                    }

                    metadata_path = output_dir / f"{prompt_data['id']}_seed{seed:02d}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    results.append({"success": True, "file": str(output_path)})

                except Exception as e:
                    print(f"\nError: {e}")
                    results.append({"success": False, "error": str(e)})

                pbar.update(1)

    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r.get('success'))

    print(f"\n{'='*60}")
    print(f"MUSICGEN-LARGE COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{total_samples}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Avg: {elapsed/successful:.1f}s per sample")
    print(f"Output: {output_dir}")

    return results

if __name__ == "__main__":
    print("="*60)
    print("MUSICGEN-LARGE GENERATION")
    print("="*60)
    print("\nTarget: 300 audio samples")
    print("Estimated time: 150-180 minutes on GPU")
    print()

    results = generate_musicgen_large(num_prompts=100, num_seeds=3)

    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Success: {sum(1 for r in results if r.get('success'))}/300")
