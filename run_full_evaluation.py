"""
Full evaluation pipeline for EAIM 2026 paper.
Generates complete dataset: 100 prompts × 2 audio models × 3 seeds = 600 audio samples
"""
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm

class FullEvaluationPipeline:
    """Run complete evaluation with all models."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        if self.device == "cpu":
            print("WARNING: Running on CPU will be very slow!")

    def generate_musicgen_full(self, num_prompts=100, num_seeds=3):
        """Generate full MusicGen dataset."""
        print("="*60)
        print(f"FULL MUSICGEN GENERATION: {num_prompts} prompts × {num_seeds} seeds")
        print("="*60)

        # Load model
        print("\nLoading MusicGen-small...")
        model_name = "facebook/musicgen-small"
        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        model.to(self.device)
        print("✓ Model loaded")

        # Load prompts
        with open("data/prompts/prompts_text.json") as f:
            prompts_data = json.load(f)[:num_prompts]

        output_dir = Path("runs/artifacts/wav/musicgen_full")
        output_dir.mkdir(parents=True, exist_ok=True)

        total_samples = num_prompts * num_seeds
        print(f"\nGenerating {total_samples} samples...")

        results = []
        start_time = time.time()

        with tqdm(total=total_samples, desc="MusicGen Generation") as pbar:
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
                        ).to(self.device)

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
                            "model": "musicgen-small",
                            "prompt_id": prompt_data['id'],
                            "prompt": prompt_data['text'],
                            "seed": seed,
                            "genre": prompt_data.get('genre'),
                            "bpm": prompt_data.get('bpm'),
                            "device": self.device,
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
        print(f"MUSICGEN COMPLETE")
        print(f"{'='*60}")
        print(f"Successful: {successful}/{total_samples}")
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Avg: {elapsed/successful:.1f}s per sample")
        print(f"Output: {output_dir}")

        return results

    def generate_stableaudio_full(self, num_prompts=100, num_seeds=3):
        """Generate full Stable Audio dataset."""
        print("\n" + "="*60)
        print(f"STABLE AUDIO GENERATION: {num_prompts} prompts × {num_seeds} seeds")
        print("="*60)

        try:
            from stable_audio_tools import get_pretrained_model
            from stable_audio_tools.inference.generation import generate_diffusion_cond

            # Load model
            print("\nLoading Stable Audio Open...")
            model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
            model.to(self.device)
            print("✓ Model loaded")

            # Load prompts
            with open("data/prompts/prompts_text.json") as f:
                prompts_data = json.load(f)[:num_prompts]

            output_dir = Path("runs/artifacts/wav/stableaudio_full")
            output_dir.mkdir(parents=True, exist_ok=True)

            total_samples = num_prompts * num_seeds
            results = []
            start_time = time.time()

            with tqdm(total=total_samples, desc="Stable Audio Generation") as pbar:
                for prompt_data in prompts_data:
                    for seed in range(num_seeds):
                        try:
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
                                sample_size=262144,
                                sigma_min=0.3,
                                sigma_max=500,
                                sampler_type="dpmpp-3m-sde",
                                device=self.device,
                                seed=seed
                            )

                            # Save
                            audio = output[0, 0].cpu().numpy()
                            audio = audio / np.max(np.abs(audio))
                            audio_int16 = (audio * 32767).astype(np.int16)

                            output_path = output_dir / f"{prompt_data['id']}_seed{seed:02d}.wav"
                            wavfile.write(output_path, 44100, audio_int16)

                            results.append({"success": True})

                        except Exception as e:
                            print(f"\nError: {e}")
                            results.append({"success": False})

                        pbar.update(1)

            elapsed = time.time() - start_time
            successful = sum(1 for r in results if r.get('success'))

            print(f"\n{'='*60}")
            print(f"STABLE AUDIO COMPLETE")
            print(f"{'='*60}")
            print(f"Successful: {successful}/{total_samples}")
            print(f"Time: {elapsed/60:.1f} minutes")

            return results

        except ImportError:
            print("Stable Audio tools not installed. Skipping.")
            print("Install with: pip install stable-audio-tools")
            return []

def main():
    """Run full evaluation."""
    print("="*60)
    print("FULL EVALUATION PIPELINE - EAIM 2026")
    print("="*60)
    print("\nTarget: 600 audio samples (300 per model)")
    print("Estimated time: 1.5-2 hours on GPU")
    print()

    pipeline = FullEvaluationPipeline()

    # Phase 1: MusicGen
    print("\n" + "="*60)
    print("PHASE 1: MusicGen Generation")
    print("="*60)
    musicgen_results = pipeline.generate_musicgen_full(num_prompts=100, num_seeds=3)

    # Phase 2: Stable Audio (if available)
    print("\n" + "="*60)
    print("PHASE 2: Stable Audio Generation")
    print("="*60)
    stableaudio_results = pipeline.generate_stableaudio_full(num_prompts=100, num_seeds=3)

    # Summary
    print("\n" + "="*60)
    print("FULL EVALUATION COMPLETE")
    print("="*60)
    print(f"MusicGen: {sum(1 for r in musicgen_results if r.get('success'))}/300")
    print(f"Stable Audio: {sum(1 for r in stableaudio_results if r.get('success'))}/300")
    print("\nNext steps:")
    print("1. Run: python compute_full_metrics.py")
    print("2. Run: python generate_full_analysis.py")

if __name__ == "__main__":
    main()
