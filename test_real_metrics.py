"""
Test real metrics on actual MusicGen outputs.
"""
import librosa
import numpy as np
from pathlib import Path
import json
import pandas as pd

def compute_basic_metrics(audio_path):
    """Compute basic metrics on real audio."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)

    # 1. Tempo
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # 2. Spectral centroid (brightness)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # 3. Zero crossing rate (noisiness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # 4. RMS energy
    rms = librosa.feature.rms(y=y)[0]

    # 5. Duration
    duration = librosa.get_duration(y=y, sr=sr)

    return {
        "tempo_bpm": float(tempo),
        "spectral_centroid_mean": float(np.mean(spectral_centroids)),
        "spectral_centroid_std": float(np.std(spectral_centroids)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "rms_energy_mean": float(np.mean(rms)),
        "rms_energy_std": float(np.std(rms)),
        "duration_sec": float(duration),
    }

def main():
    """Test metrics on real outputs."""
    print("="*60)
    print("Testing REAL Metrics on REAL MusicGen Outputs")
    print("="*60)

    wav_dir = Path("runs/artifacts/wav/real_test")
    wav_files = sorted(wav_dir.glob("test_*.wav"))

    print(f"\nFound {len(wav_files)} audio files")

    results = []

    for wav_file in wav_files:
        print(f"\nProcessing: {wav_file.name}")

        # Load metadata
        metadata_file = wav_file.with_name(wav_file.stem + "_metadata.json")
        with open(metadata_file) as f:
            metadata = json.load(f)

        try:
            # Compute metrics
            metrics = compute_basic_metrics(wav_file)

            print(f"  ✓ Tempo: {metrics['tempo_bpm']:.1f} BPM")
            print(f"  ✓ Duration: {metrics['duration_sec']:.2f}s")
            print(f"  ✓ Spectral Centroid: {metrics['spectral_centroid_mean']:.0f} Hz")
            print(f"  ✓ RMS Energy: {metrics['rms_energy_mean']:.3f}")

            # Combine with metadata
            result = {
                "file": wav_file.name,
                "prompt": metadata['prompt'][:50] + "...",
                **metrics
            }
            results.append(result)

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    output_file = Path("runs/results_REAL/metrics_test.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS (REAL DATA)")
    print(f"{'='*60}")
    print(df[['tempo_bpm', 'spectral_centroid_mean', 'rms_energy_mean']].describe())

    print(f"\n✓ Results saved to: {output_file}")
    print("\nThese are REAL metrics from REAL MusicGen outputs!")

if __name__ == "__main__":
    main()
