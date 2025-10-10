"""
Simple metrics computation that actually works.
"""
import librosa
import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

def compute_metrics_simple(audio_path):
    """Compute metrics on one file."""
    try:
        y, sr = librosa.load(str(audio_path), sr=22050)

        # Basic metrics
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y)[0])
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        duration = librosa.get_duration(y=y, sr=sr)

        # Load metadata
        meta_file = audio_path.with_suffix('').with_name(audio_path.stem + '_metadata.json')
        model = "unknown"
        target_bpm = None
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                model = meta.get('model', 'unknown')
                target_bpm = meta.get('bpm')

        tempo_error = abs(float(tempo) - target_bpm) if target_bpm else None

        return {
            'file': audio_path.name,
            'model': model,
            'tempo_bpm': float(tempo),
            'rms_energy': float(rms),
            'spectral_centroid': float(spectral_centroid),
            'duration_sec': float(duration),
            'target_bpm': target_bpm,
            'tempo_error': tempo_error,
            'success': True
        }
    except Exception as e:
        return {
            'file': audio_path.name if hasattr(audio_path, 'name') else str(audio_path),
            'error': str(e),
            'success': False
        }

# Process both directories
print("="*60)
print("COMPUTING METRICS - SIMPLE VERSION")
print("="*60)

results = []

for model_dir in ['musicgen_full', 'musicgen_medium']:
    wav_dir = Path(f"runs/artifacts/wav/{model_dir}")
    wav_files = list(wav_dir.glob("*.wav"))
    print(f"\nProcessing {model_dir}: {len(wav_files)} files")

    for wav_file in tqdm(wav_files, desc=model_dir):
        result = compute_metrics_simple(wav_file)
        results.append(result)

# Create dataframe
df = pd.DataFrame(results)
success_df = df[df['success'] == True].copy()

print(f"\n{'='*60}")
print(f"Successful: {len(success_df)}/{len(df)}")
print(f"{'='*60}")

if len(success_df) > 0:
    # Save
    out_dir = Path("runs/results_FULL")
    out_dir.mkdir(parents=True, exist_ok=True)
    success_df.to_csv(out_dir / "metrics_simple.csv", index=False)
    success_df.to_parquet(out_dir / "metrics_simple.parquet", index=False)

    # Show stats by model
    for model in success_df['model'].unique():
        model_df = success_df[success_df['model'] == model]
        print(f"\n{model.upper()} (n={len(model_df)})")
        print(f"  Tempo: {model_df['tempo_bpm'].mean():.1f} ± {model_df['tempo_bpm'].std():.1f} BPM")
        print(f"  Energy: {model_df['rms_energy'].mean():.3f} ± {model_df['rms_energy'].std():.3f}")
        print(f"  Spectral: {model_df['spectral_centroid'].mean():.0f} ± {model_df['spectral_centroid'].std():.0f} Hz")

        if 'tempo_error' in model_df.columns:
            errors = model_df['tempo_error'].dropna()
            if len(errors) > 0:
                print(f"  Tempo error: {errors.median():.1f} BPM median")
                print(f"  Within 5 BPM: {(errors < 5).sum()}/{len(errors)} ({100*(errors < 5).sum()/len(errors):.1f}%)")

    print(f"\n✓ Saved to {out_dir}")
else:
    print("\n✗ No successful metrics")
