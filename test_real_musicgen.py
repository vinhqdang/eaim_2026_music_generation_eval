"""
Test real MusicGen model generation.
This will download the model (~3.3GB) and generate actual music.
"""
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile
import numpy as np
from pathlib import Path
import json
import time

def test_musicgen_real():
    """Test MusicGen with real model."""
    print("="*60)
    print("Testing REAL MusicGen Model")
    print("="*60)

    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU - this will be SLOW")
        print("Consider using a GPU for reasonable performance")

    # Load model
    print("\nLoading MusicGen-small model...")
    print("(Using 'small' model for faster testing, ~300MB)")
    print("For paper, use 'facebook/musicgen-large'")

    try:
        model_name = "facebook/musicgen-small"
        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        print(f"✓ Model loaded successfully on {device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

    # Load prompts
    prompts_file = Path("data/prompts/prompts_text.json")
    with open(prompts_file) as f:
        prompts_data = json.load(f)

    # Test with first 3 prompts
    test_prompts = prompts_data[:3]
    output_dir = Path("runs/artifacts/wav/real_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, prompt_data in enumerate(test_prompts):
        prompt_text = prompt_data['text']
        print(f"\n{'='*60}")
        print(f"Generating {i+1}/3: {prompt_text[:60]}...")

        start_time = time.time()

        try:
            # Process input
            inputs = processor(
                text=[prompt_text],
                padding=True,
                return_tensors="pt",
            ).to(device)

            # Generate (10 seconds for speed)
            # 50 tokens/sec * 10 sec = 500 tokens
            with torch.no_grad():
                audio_values = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=1.0,
                    guidance_scale=3.0,
                )

            # Convert to numpy
            sampling_rate = model.config.audio_encoder.sampling_rate
            audio_np = audio_values[0].cpu().numpy()

            # Normalize and save
            audio_np = np.squeeze(audio_np)
            audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)
            audio_int16 = (audio_np * 32767).astype(np.int16)

            output_path = output_dir / f"test_{i:03d}.wav"
            wavfile.write(output_path, sampling_rate, audio_int16)

            generation_time = time.time() - start_time
            file_size = output_path.stat().st_size / 1024  # KB

            print(f"✓ Generated: {output_path.name}")
            print(f"  Time: {generation_time:.1f}s")
            print(f"  Size: {file_size:.1f} KB")
            print(f"  Sample rate: {sampling_rate} Hz")

            # Save metadata
            metadata = {
                "model": "musicgen-small",
                "prompt": prompt_text,
                "duration_target": 10,
                "generation_time": generation_time,
                "device": device,
                "sampling_rate": sampling_rate,
                "max_new_tokens": 500,
                "temperature": 1.0,
                "guidance_scale": 3.0,
                "file_size_kb": file_size,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            metadata_path = output_dir / f"test_{i:03d}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            results.append({
                "prompt_id": i,
                "success": True,
                "generation_time": generation_time,
                "output_path": str(output_path)
            })

        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "prompt_id": i,
                "success": False,
                "error": str(e)
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r['success'])
    print(f"Successful: {successful}/3")

    if successful > 0:
        avg_time = np.mean([r['generation_time'] for r in results if r['success']])
        print(f"Avg generation time: {avg_time:.1f}s per 10s clip")
        print(f"\nEstimated time for full dataset:")
        print(f"  100 prompts × 3 seeds = 300 clips")
        print(f"  Time: ~{(300 * avg_time / 3600):.1f} hours")

    print(f"\nOutput directory: {output_dir}")
    print("\n✓ REAL MODEL TEST COMPLETE")
    print("These are ACTUAL MusicGen outputs, not synthetic!")

    return results

if __name__ == "__main__":
    results = test_musicgen_real()
