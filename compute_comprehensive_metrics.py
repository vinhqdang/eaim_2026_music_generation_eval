"""
Compute comprehensive audio metrics on existing samples.
Simpler version that processes each metric separately with progress tracking.
"""
import json
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch

# Import metric calculators
from metrics.audio.tempo import TempoConsistencyCalculator
from metrics.audio.key_stability import KeyStabilityCalculator

def main():
    print("="*60)
    print("COMPREHENSIVE AUDIO METRICS COMPUTATION")
    print("="*60)

    # Find all WAV files
    wav_files = []
    for model_dir in ['musicgen_full', 'musicgen_medium']:
        wav_dir = Path(f"runs/artifacts/wav/{model_dir}")
        if wav_dir.exists():
            files = list(wav_dir.glob("*.wav"))
            print(f"Found {len(files)} files in {model_dir}")
            wav_files.extend(files)

    print(f"\nTotal files: {len(wav_files)}")

    if len(wav_files) == 0:
        print("ERROR: No WAV files found")
        return

    # Initialize metric calculators
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print("\nInitializing metric calculators...")

    tempo_calc = TempoConsistencyCalculator()
    print("✓ Tempo calculator")

    key_calc = KeyStabilityCalculator()
    print("✓ Key stability calculator")

    # Results storage
    results = []

    print(f"\nProcessing {len(wav_files)} files...")
    for wav_file in tqdm(wav_files, desc="Computing metrics"):
        try:
            # Load metadata
            meta_file = wav_file.with_suffix('').with_name(wav_file.stem + '_metadata.json')
            model = "unknown"
            target_bpm = None
            prompt_text = None

            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                    model = meta.get('model', 'unknown')
                    target_bpm = meta.get('bpm')
                    prompt_text = meta.get('prompt')

            # Compute tempo metrics
            tempo_metrics = tempo_calc.compute(str(wav_file))

            # Compute key stability metrics
            key_metrics = key_calc.compute(str(wav_file))

            # Combine all metrics
            result = {
                'file': wav_file.name,
                'model': model,
                'prompt': prompt_text,
                'target_bpm': target_bpm,

                # Tempo metrics
                'tempo_bpm': tempo_metrics.get('tempo_bpm'),
                'tempo_stability': tempo_metrics.get('tempo_stability'),
                'beat_strength': tempo_metrics.get('beat_strength'),

                # Key metrics
                'key': key_metrics.get('key'),
                'key_stability': key_metrics.get('stability'),
                'tonal_strength': key_metrics.get('tonal_strength'),

                'success': True
            }

            # Add tempo error if target available
            if target_bpm and result['tempo_bpm']:
                result['tempo_error'] = abs(result['tempo_bpm'] - target_bpm)

            results.append(result)

        except Exception as e:
            tqdm.write(f"ERROR: {wav_file.name}: {e}")
            results.append({
                'file': wav_file.name,
                'error': str(e),
                'success': False
            })

    # Convert to DataFrame
    df = pd.DataFrame(results)
    success_df = df[df['success'] == True].copy()

    print(f"\n{'='*60}")
    print(f"Results: {len(success_df)}/{len(df)} successful")
    print(f"{'='*60}")

    if len(success_df) > 0:
        # Save results
        out_dir = Path("runs/results_COMPREHENSIVE")
        out_dir.mkdir(parents=True, exist_ok=True)

        success_df.to_csv(out_dir / "metrics_comprehensive.csv", index=False)
        success_df.to_parquet(out_dir / "metrics_comprehensive.parquet", index=False)

        print(f"\nSaved to: {out_dir}")

        # Show summary by model
        for model in success_df['model'].unique():
            model_df = success_df[success_df['model'] == model]
            print(f"\n{model.upper()} (n={len(model_df)})")
            print(f"  Tempo: {model_df['tempo_bpm'].mean():.1f} ± {model_df['tempo_bpm'].std():.1f} BPM")
            print(f"  Tempo stability: {model_df['tempo_stability'].mean():.3f}")
            print(f"  Key stability: {model_df['key_stability'].mean():.3f}")
            print(f"  Tonal strength: {model_df['tonal_strength'].mean():.3f}")

            if 'tempo_error' in model_df.columns:
                errors = model_df['tempo_error'].dropna()
                if len(errors) > 0:
                    print(f"  Tempo error: {errors.median():.1f} BPM median")
                    print(f"  Within 5 BPM: {(errors < 5).sum()}/{len(errors)} ({100*(errors < 5).sum()/len(errors):.1f}%)")

        print(f"\n✓ Comprehensive metrics computed successfully")
    else:
        print("\n✗ No successful metrics")

if __name__ == "__main__":
    main()
