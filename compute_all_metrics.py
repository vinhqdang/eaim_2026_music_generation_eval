"""
Compute comprehensive metrics on all available samples (806 total).
Includes MusicGen-small, medium, and large (partial).
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
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
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
            'rms_energy_mean': float(rms),
            'spectral_centroid_mean': float(spectral_centroid),
            'spectral_bandwidth_mean': float(spectral_bandwidth),
            'chroma_mean': float(chroma),
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

print("="*60)
print("COMPUTING METRICS ON ALL 806 SAMPLES")
print("="*60)

results = []

# Process all three model directories
for model_dir in ['musicgen_full', 'musicgen_medium', 'musicgen_large']:
    wav_dir = Path(f"runs/artifacts/wav/{model_dir}")
    if wav_dir.exists():
        files = list(wav_dir.glob("*.wav"))
        print(f"\nProcessing {model_dir}: {len(files)} files")

        for wav_file in tqdm(files, desc=model_dir):
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
    out_dir = Path("runs/results_FINAL")
    out_dir.mkdir(parents=True, exist_ok=True)
    success_df.to_csv(out_dir / "metrics_all_806.csv", index=False)
    success_df.to_parquet(out_dir / "metrics_all_806.parquet", index=False)

    # Show stats by model
    print("\n" + "="*60)
    print("SUMMARY BY MODEL")
    print("="*60)

    for model in sorted(success_df['model'].unique()):
        model_df = success_df[success_df['model'] == model]
        print(f"\n{model.upper()} (n={len(model_df)})")
        print(f"  Tempo: {model_df['tempo_bpm'].mean():.1f} ± {model_df['tempo_bpm'].std():.1f} BPM")
        print(f"  Energy: {model_df['rms_energy_mean'].mean():.3f} ± {model_df['rms_energy_mean'].std():.3f}")
        print(f"  Spectral: {model_df['spectral_centroid_mean'].mean():.0f} ± {model_df['spectral_centroid_mean'].std():.0f} Hz")
        print(f"  Chroma: {model_df['chroma_mean'].mean():.3f} ± {model_df['chroma_mean'].std():.3f}")

        if 'tempo_error' in model_df.columns:
            errors = model_df['tempo_error'].dropna()
            if len(errors) > 0:
                print(f"  Tempo error: {errors.median():.1f} BPM median")
                print(f"  Within 5 BPM: {(errors < 5).sum()}/{len(errors)} ({100*(errors < 5).sum()/len(errors):.1f}%)")

    print(f"\n✓ Saved to {out_dir}")
else:
    print("\n✗ No successful metrics")
