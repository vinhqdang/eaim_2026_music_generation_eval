"""
Compute REAL metrics on REAL MusicGen outputs for paper.
"""
import librosa
import numpy as np
from pathlib import Path
import json
import pandas as pd

def compute_comprehensive_metrics(audio_path):
    """Compute comprehensive metrics on real audio."""
    y, sr = librosa.load(audio_path, sr=22050)

    metrics = {}

    # 1. Tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    metrics['tempo_bpm'] = float(tempo)
    metrics['num_beats'] = len(beats)

    # 2. Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    metrics['spectral_centroid_mean_hz'] = float(np.mean(spectral_centroids))
    metrics['spectral_centroid_std_hz'] = float(np.std(spectral_centroids))

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    metrics['spectral_rolloff_mean_hz'] = float(np.mean(spectral_rolloff))

    # 3. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    metrics['zero_crossing_rate_mean'] = float(np.mean(zcr))

    # 4. RMS energy
    rms = librosa.feature.rms(y=y)[0]
    metrics['rms_energy_mean'] = float(np.mean(rms))
    metrics['rms_energy_std'] = float(np.std(rms))

    # 5. Chroma features (tonality)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    metrics['chroma_mean'] = float(np.mean(chroma))
    metrics['chroma_std'] = float(np.std(chroma))

    # 6. MFCC (timbre)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    metrics['mfcc_mean'] = float(np.mean(mfccs))
    metrics['mfcc_std'] = float(np.std(mfccs))

    # 7. Duration
    metrics['duration_sec'] = float(librosa.get_duration(y=y, sr=sr))

    return metrics

def main():
    """Compute metrics on all real outputs."""
    print("="*60)
    print("Computing REAL Metrics on 20 MusicGen Outputs")
    print("="*60)

    wav_dir = Path("runs/artifacts/wav/musicgen_real")
    wav_files = sorted(wav_dir.glob("*_real.wav"))

    print(f"\nFound {len(wav_files)} audio files\n")

    results = []

    for i, wav_file in enumerate(wav_files, 1):
        print(f"[{i}/{len(wav_files)}] {wav_file.name}")

        # Load metadata
        metadata_file = wav_file.with_name(wav_file.stem + "_metadata.json")
        with open(metadata_file) as f:
            metadata = json.load(f)

        try:
            # Compute metrics
            metrics = compute_comprehensive_metrics(wav_file)

            print(f"  Tempo: {metrics['tempo_bpm']:.1f} BPM (target: {metadata.get('bpm', 'N/A')})")
            print(f"  Duration: {metrics['duration_sec']:.2f}s")
            print(f"  Energy: {metrics['rms_energy_mean']:.3f}")

            # Combine
            result = {
                "file": wav_file.name,
                "prompt_id": metadata['prompt_id'],
                "genre": metadata.get('genre', 'unknown'),
                "target_bpm": metadata.get('bpm', None),
                **metrics
            }
            results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save
    output_dir = Path("runs/results_REAL")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / "musicgen_metrics_real.csv"
    parquet_file = output_dir / "musicgen_metrics_real.parquet"

    df.to_csv(csv_file, index=False)
    df.to_parquet(parquet_file, index=False)

    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS (REAL EXPERIMENTAL DATA)")
    print(f"{'='*60}\n")

    # Key metrics
    key_metrics = ['tempo_bpm', 'duration_sec', 'spectral_centroid_mean_hz',
                   'rms_energy_mean', 'chroma_mean']

    print(df[key_metrics].describe().round(2))

    # Tempo accuracy
    df_with_target = df[df['target_bpm'].notna()]
    if len(df_with_target) > 0:
        df_with_target['tempo_error'] = abs(df_with_target['tempo_bpm'] - df_with_target['target_bpm'])
        print(f"\n{'='*60}")
        print("TEMPO ACCURACY")
        print(f"{'='*60}")
        print(f"Mean error: {df_with_target['tempo_error'].mean():.1f} BPM")
        print(f"Median error: {df_with_target['tempo_error'].median():.1f} BPM")
        print(f"Std error: {df_with_target['tempo_error'].std():.1f} BPM")

    print(f"\n{'='*60}")
    print(f"✓ Results saved to:")
    print(f"  - {csv_file}")
    print(f"  - {parquet_file}")
    print(f"\n✓ THESE ARE REAL METRICS FROM REAL MODEL OUTPUTS!")
    print(f"{'='*60}")

    return df

if __name__ == "__main__":
    df = main()
