"""
Key and tonality stability detection.

This module analyzes the harmonic content and key stability of audio using
librosa and essentia for key detection and tonality analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import librosa
from scipy import stats


class KeyStabilityCalculator:
    """
    Calculate key and tonality stability metrics for audio.

    This calculator measures:
    - Key detection and confidence
    - Key stability over time
    - Harmonic consistency
    - Tonal strength

    Attributes:
        sample_rate: Target sample rate for audio processing
        hop_length: Hop length for frame-based analysis
        n_fft: FFT size for spectral analysis
    """

    # Key names for major and minor scales
    MAJOR_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    MINOR_KEYS = ['Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm']

    # Krumhansl-Schmuckler key profiles
    MAJOR_PROFILE = np.array([
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
    ])
    MINOR_PROFILE = np.array([
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
    ])

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
    ):
        """
        Initialize key stability calculator.

        Args:
            sample_rate: Target sample rate for audio
            hop_length: Hop length for analysis
            n_fft: FFT size for spectral analysis
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft

    def _load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio waveform, sample rate)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If audio loading fails
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio {audio_path}: {e}")

    def compute_chromagram(
        self,
        audio: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
        Compute chromagram (pitch class distribution) from audio.

        Args:
            audio: Audio waveform
            sr: Sample rate

        Returns:
            Chromagram (12 x n_frames)
        """
        # Compute CQT-based chromagram
        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=sr,
            hop_length=self.hop_length,
            n_chroma=12
        )

        return chroma

    def estimate_key_from_chroma(
        self,
        chroma: np.ndarray
    ) -> Tuple[str, str, float]:
        """
        Estimate key from chromagram using Krumhansl-Schmuckler algorithm.

        Args:
            chroma: Chromagram (12 x n_frames)

        Returns:
            Tuple of (key_name, mode, confidence)
        """
        # Average chroma across time
        avg_chroma = np.mean(chroma, axis=1)

        # Normalize
        avg_chroma = avg_chroma / (np.sum(avg_chroma) + 1e-8)

        # Calculate correlation with all major and minor key profiles
        max_correlation = -1.0
        best_key = None
        best_mode = None

        # Try all major keys
        for i in range(12):
            # Rotate profile to match key
            rotated_profile = np.roll(self.MAJOR_PROFILE, i)
            rotated_profile = rotated_profile / np.sum(rotated_profile)

            # Calculate correlation
            correlation = np.corrcoef(avg_chroma, rotated_profile)[0, 1]

            if correlation > max_correlation:
                max_correlation = correlation
                best_key = self.MAJOR_KEYS[i]
                best_mode = "major"

        # Try all minor keys
        for i in range(12):
            # Rotate profile to match key
            rotated_profile = np.roll(self.MINOR_PROFILE, i)
            rotated_profile = rotated_profile / np.sum(rotated_profile)

            # Calculate correlation
            correlation = np.corrcoef(avg_chroma, rotated_profile)[0, 1]

            if correlation > max_correlation:
                max_correlation = correlation
                best_key = self.MINOR_KEYS[i]
                best_mode = "minor"

        # Confidence is the correlation coefficient
        confidence = float(max(0.0, max_correlation))

        return best_key, best_mode, confidence

    def estimate_local_keys(
        self,
        chroma: np.ndarray,
        window_size: int = 8
    ) -> List[Tuple[str, str, float]]:
        """
        Estimate key for local windows of the audio.

        Args:
            chroma: Chromagram (12 x n_frames)
            window_size: Size of window in frames

        Returns:
            List of (key_name, mode, confidence) for each window
        """
        n_frames = chroma.shape[1]
        local_keys = []

        for i in range(0, n_frames, window_size):
            end_idx = min(i + window_size, n_frames)
            window_chroma = chroma[:, i:end_idx]

            key, mode, confidence = self.estimate_key_from_chroma(window_chroma)
            local_keys.append((key, mode, confidence))

        return local_keys

    def calculate_key_stability(
        self,
        local_keys: List[Tuple[str, str, float]]
    ) -> Dict[str, float]:
        """
        Calculate key stability metrics from local key estimates.

        Args:
            local_keys: List of (key_name, mode, confidence) tuples

        Returns:
            Dictionary with stability metrics
        """
        if not local_keys:
            return {
                "key_stability": 0.0,
                "mode_stability": 0.0,
                "avg_confidence": 0.0,
            }

        keys = [k[0] for k in local_keys]
        modes = [k[1] for k in local_keys]
        confidences = [k[2] for k in local_keys]

        # Key stability: proportion of most common key
        key_counts = {}
        for key in keys:
            key_counts[key] = key_counts.get(key, 0) + 1
        key_stability = max(key_counts.values()) / len(keys) if keys else 0.0

        # Mode stability: proportion of most common mode
        mode_counts = {}
        for mode in modes:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        mode_stability = max(mode_counts.values()) / len(modes) if modes else 0.0

        # Average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return {
            "key_stability": float(key_stability),
            "mode_stability": float(mode_stability),
            "avg_confidence": float(avg_confidence),
            "n_key_changes": len(set(keys)) - 1,
            "n_mode_changes": len(set(modes)) - 1,
        }

    def calculate_tonal_strength(
        self,
        chroma: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate tonal strength and harmonic consistency.

        Args:
            chroma: Chromagram (12 x n_frames)

        Returns:
            Dictionary with tonal strength metrics
        """
        # Tonal centroid (center of mass in tonal space)
        # Higher variance indicates less tonal consistency
        tonal_variance = np.var(chroma, axis=1).mean()

        # Harmonic change detection (how much chroma changes over time)
        chroma_diff = np.diff(chroma, axis=1)
        harmonic_flux = np.mean(np.sum(np.abs(chroma_diff), axis=0))

        # Spectral flatness of chroma (how uniform the pitch distribution is)
        # Lower values indicate stronger tonality
        chroma_flatness = []
        for frame in chroma.T:
            geometric_mean = np.exp(np.mean(np.log(frame + 1e-8)))
            arithmetic_mean = np.mean(frame)
            flatness = geometric_mean / (arithmetic_mean + 1e-8)
            chroma_flatness.append(flatness)

        avg_flatness = np.mean(chroma_flatness)

        # Tonal strength: inverse of flatness (0-1, higher is more tonal)
        tonal_strength = 1.0 - avg_flatness

        return {
            "tonal_strength": float(tonal_strength),
            "tonal_variance": float(tonal_variance),
            "harmonic_flux": float(harmonic_flux),
            "chroma_flatness": float(avg_flatness),
        }

    def compute(
        self,
        audio_path: Union[str, Path],
        window_size: int = 8,
    ) -> Dict[str, Union[str, float, int]]:
        """
        Compute comprehensive key stability and tonality metrics.

        Args:
            audio_path: Path to audio file
            window_size: Window size in frames for local key estimation

        Returns:
            Dictionary containing:
                - global_key: Estimated global key
                - global_mode: Estimated global mode (major/minor)
                - key_confidence: Confidence of key estimate
                - key_stability: Key stability score (0-1)
                - tonal_strength: Tonal strength score (0-1)
                - And other key/tonality metrics

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If processing fails
        """
        # Load audio
        audio, sr = self._load_audio(audio_path)

        # Compute chromagram
        chroma = self.compute_chromagram(audio, sr)

        # Estimate global key
        global_key, global_mode, key_confidence = self.estimate_key_from_chroma(chroma)

        # Estimate local keys
        local_keys = self.estimate_local_keys(chroma, window_size)

        # Calculate key stability
        stability_metrics = self.calculate_key_stability(local_keys)

        # Calculate tonal strength
        tonal_metrics = self.calculate_tonal_strength(chroma)

        # Build result dictionary
        result = {
            "global_key": global_key,
            "global_mode": global_mode,
            "key_confidence": float(key_confidence),
            "n_local_windows": len(local_keys),
            **stability_metrics,
            **tonal_metrics,
        }

        return result

    def compute_batch(
        self,
        audio_paths: List[Union[str, Path]]
    ) -> Dict[str, Union[float, List]]:
        """
        Compute key stability metrics for multiple audio files.

        Args:
            audio_paths: List of audio file paths

        Returns:
            Dictionary containing aggregated metrics:
                - avg_key_stability: Average key stability
                - avg_tonal_strength: Average tonal strength
                - key_distribution: Distribution of detected keys
                - individual_results: List of individual results

        Raises:
            ValueError: If audio_paths is empty
        """
        if not audio_paths:
            raise ValueError("audio_paths must be non-empty")

        individual_results = []
        key_stabilities = []
        tonal_strengths = []
        detected_keys = []

        for audio_path in audio_paths:
            try:
                result = self.compute(audio_path)
                individual_results.append(result)
                key_stabilities.append(result["key_stability"])
                tonal_strengths.append(result["tonal_strength"])
                detected_keys.append(f"{result['global_key']} {result['global_mode']}")
            except Exception as e:
                print(f"Warning: Failed to process {audio_path}: {e}")
                continue

        if not individual_results:
            raise RuntimeError("Failed to process any audio files")

        # Key distribution
        key_distribution = {}
        for key in detected_keys:
            key_distribution[key] = key_distribution.get(key, 0) + 1

        return {
            "avg_key_stability": float(np.mean(key_stabilities)),
            "avg_tonal_strength": float(np.mean(tonal_strengths)),
            "std_key_stability": float(np.std(key_stabilities)),
            "std_tonal_strength": float(np.std(tonal_strengths)),
            "key_distribution": key_distribution,
            "n_files": len(individual_results),
            "individual_results": individual_results,
        }


if __name__ == "__main__":
    # Example usage
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python key_stability.py <audio_file> [audio_file2 ...]")
        sys.exit(1)

    audio_files = [Path(p) for p in sys.argv[1:]]

    calculator = KeyStabilityCalculator()

    if len(audio_files) == 1:
        # Single file analysis
        print(f"\nAnalyzing key stability for: {audio_files[0]}")
        result = calculator.compute(audio_files[0])

        print("\nResults:")
        print(f"  Key: {result['global_key']} {result['global_mode']}")
        print(f"  Confidence: {result['key_confidence']:.4f}")
        print(f"  Key Stability: {result['key_stability']:.4f}")
        print(f"  Tonal Strength: {result['tonal_strength']:.4f}")
        print(f"  Key Changes: {result['n_key_changes']}")
        print(f"  Mode Changes: {result['n_mode_changes']}")
    else:
        # Batch analysis
        print(f"\nAnalyzing key stability for {len(audio_files)} files...")
        result = calculator.compute_batch(audio_files)

        print("\nAggregated Results:")
        print(f"  Avg Key Stability: {result['avg_key_stability']:.4f}")
        print(f"  Avg Tonal Strength: {result['avg_tonal_strength']:.4f}")

        print("\nKey Distribution:")
        for key, count in sorted(result['key_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {key}: {count} files")

        print("\nIndividual Results:")
        for i, res in enumerate(result["individual_results"]):
            print(f"\n  File {i+1}: {audio_files[i].name}")
            print(f"    Key: {res['global_key']} {res['global_mode']}")
            print(f"    Stability: {res['key_stability']:.4f}")
            print(f"    Tonal Strength: {res['tonal_strength']:.4f}")
