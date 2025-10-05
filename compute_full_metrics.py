"""
Compute comprehensive metrics on all generated audio samples.
Processes 600 audio files (300 MusicGen + 300 Stable Audio).
"""
import librosa
import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def compute_all_metrics(audio_path):
    """Compute all metrics on a single audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=22050)

        metrics = {
            "file": audio_path.name,
            "file_path": str(audio_path)
        }

        # Load metadata if available
        metadata_file = audio_path.with_name(audio_path.stem + "_metadata.json")
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                metrics.update({
                    "model": metadata.get("model", "unknown"),
                    "prompt_id": metadata.get("prompt_id", "unknown"),
                    "seed": metadata.get("seed", 0),
                    "genre": metadata.get("genre", "unknown"),
                    "target_bpm": metadata.get("bpm", None),
                })

        # 1. Tempo and beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        metrics['tempo_bpm'] = float(tempo)
        metrics['num_beats'] = len(beats)

        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        metrics['spectral_centroid_std'] = float(np.std(spectral_centroids))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        metrics['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        metrics['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))

        # 3. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        metrics['zero_crossing_rate'] = float(np.mean(zcr))

        # 4. RMS energy
        rms = librosa.feature.rms(y=y)[0]
        metrics['rms_energy_mean'] = float(np.mean(rms))
        metrics['rms_energy_std'] = float(np.std(rms))

        # 5. Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        metrics['chroma_mean'] = float(np.mean(chroma))
        metrics['chroma_std'] = float(np.std(chroma))

        # 6. Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        metrics['spectral_contrast_mean'] = float(np.mean(contrast))

        # 7. Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        metrics['mel_spec_mean'] = float(np.mean(mel_spec))
        metrics['mel_spec_std'] = float(np.std(mel_spec))

        # 8. MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        metrics['mfcc_mean'] = float(np.mean(mfccs))
        metrics['mfcc_std'] = float(np.std(mfccs))

        # 9. Duration
        metrics['duration_sec'] = float(librosa.get_duration(y=y, sr=sr))

        # 10. Tempo error if target available
        if metrics.get('target_bpm'):
            metrics['tempo_error'] = abs(metrics['tempo_bpm'] - metrics['target_bpm'])

        metrics['success'] = True

    except Exception as e:
        metrics = {
            "file": audio_path.name if hasattr(audio_path, 'name') else str(audio_path),
            "success": False,
            "error": str(e)
        }

    return metrics

def process_directory(audio_dir, num_workers=8):
    """Process all audio files in a directory."""
    audio_files = list(Path(audio_dir).glob("*.wav"))

    if not audio_files:
        print(f"No WAV files found in {audio_dir}")
        return pd.DataFrame()

    print(f"Processing {len(audio_files)} files with {num_workers} workers...")

    # Parallel processing
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(compute_all_metrics, audio_files),
            total=len(audio_files),
            desc=f"Processing {Path(audio_dir).name}"
        ))

    df = pd.DataFrame(results)
    successful = df['success'].sum()
    print(f"  ✓ Successful: {successful}/{len(audio_files)}")

    return df

def main():
    """Compute metrics on all generated audio."""
    print("="*60)
    print("COMPUTING COMPREHENSIVE METRICS ON ALL AUDIO")
    print("="*60)

    # Process MusicGen
    print("\n1. MusicGen samples...")
    musicgen_df = process_directory("runs/artifacts/wav/musicgen_full", num_workers=8)

    # Process Stable Audio
    print("\n2. Stable Audio samples...")
    stableaudio_df = process_directory("runs/artifacts/wav/stableaudio_full", num_workers=8)

    # Combine
    all_df = pd.concat([musicgen_df, stableaudio_df], ignore_index=True)

    # Filter successful
    success_df = all_df[all_df['success'] == True].copy()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(all_df)}")
    print(f"Successful: {len(success_df)}")
    print(f"Failed: {len(all_df) - len(success_df)}")

    if len(success_df) > 0:
        # Save results
        output_dir = Path("runs/results_FULL")
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_file = output_dir / "all_audio_metrics.csv"
        parquet_file = output_dir / "all_audio_metrics.parquet"

        success_df.to_csv(csv_file, index=False)
        success_df.to_parquet(parquet_file, index=False)

        print(f"\n✓ Saved to:")
        print(f"  - {csv_file}")
        print(f"  - {parquet_file}")

        # Statistics by model
        print(f"\n{'='*60}")
        print("STATISTICS BY MODEL")
        print(f"{'='*60}\n")

        for model in success_df['model'].unique():
            model_df = success_df[success_df['model'] == model]
            print(f"\n{model.upper()} (n={len(model_df)})")
            print("-" * 40)

            stats = model_df[['tempo_bpm', 'rms_energy_mean', 'spectral_centroid_mean']].describe()
            print(stats.round(2))

            # Tempo accuracy
            if 'tempo_error' in model_df.columns:
                errors = model_df['tempo_error'].dropna()
                if len(errors) > 0:
                    print(f"\nTempo Accuracy:")
                    print(f"  Mean error: {errors.mean():.1f} BPM")
                    print(f"  Median error: {errors.median():.1f} BPM")
                    print(f"  Within 5 BPM: {(errors < 5).sum()}/{len(errors)} ({100*(errors < 5).sum()/len(errors):.1f}%)")

    print(f"\n{'='*60}")
    print("✓ METRICS COMPUTATION COMPLETE")
    print(f"{'='*60}")

    return success_df

if __name__ == "__main__":
    df = main()
