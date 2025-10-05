"""
Generate sufficient REAL samples for paper submission.
Target: 20 samples across different prompts to show real experimental results.
"""
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile
import numpy as np
from pathlib import Path
import json
import time

def generate_paper_samples(num_samples=20):
    """Generate real samples for paper."""
    print("="*60)
    print(f"Generating {num_samples} REAL MusicGen Samples for Paper")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load model
    print("\nLoading MusicGen-small model...")
    model_name = "facebook/musicgen-small"
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    print(f"✓ Model loaded")

    # Load prompts
    prompts_file = Path("data/prompts/prompts_text.json")
    with open(prompts_file) as f:
        prompts_data = json.load(f)

    # Use first num_samples prompts
    selected_prompts = prompts_data[:num_samples]

    output_dir = Path("runs/artifacts/wav/musicgen_real")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_start = time.time()

    for i, prompt_data in enumerate(selected_prompts):
        prompt_text = prompt_data['text']
        print(f"\n[{i+1}/{num_samples}] {prompt_text[:60]}...")

        start_time = time.time()

        try:
            # Process
            inputs = processor(
                text=[prompt_text],
                padding=True,
                return_tensors="pt",
            ).to(device)

            # Generate
            with torch.no_grad():
                audio_values = model.generate(
                    **inputs,
                    max_new_tokens=500,  # ~10s
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

            output_path = output_dir / f"prompt_{prompt_data['id']}_real.wav"
            wavfile.write(output_path, sampling_rate, audio_int16)

            generation_time = time.time() - start_time

            # Metadata
            metadata = {
                "model": "musicgen-small",
                "prompt_id": prompt_data['id'],
                "prompt": prompt_text,
                "genre": prompt_data.get('genre', 'unknown'),
                "bpm": prompt_data.get('bpm', None),
                "generation_time_sec": generation_time,
                "device": device,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "REAL_MODEL_OUTPUT": True
            }

            metadata_path = output_dir / f"prompt_{prompt_data['id']}_real_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            results.append({
                "prompt_id": prompt_data['id'],
                "success": True,
                "time": generation_time
            })

            print(f"  ✓ {generation_time:.1f}s")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "prompt_id": prompt_data['id'],
                "success": False
            })

    total_time = time.time() - total_start
    successful = sum(1 for r in results if r['success'])

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{num_samples}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Avg time per clip: {total_time/successful:.1f}s")
    print(f"\nOutput: {output_dir}")
    print("\n✓ THESE ARE REAL MODEL OUTPUTS - NOT SYNTHETIC!")

    return results

if __name__ == "__main__":
    generate_paper_samples(20)
