"""
Music structure detection via novelty curves.

This module detects structural segments in music (intro, verse, chorus, etc.)
using spectral and harmonic novelty analysis with librosa.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import librosa
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


class StructureDetectionCalculator:
    """
    Detect musical structure using novelty curves and segmentation.

    This calculator measures:
    - Structural segment boundaries
    - Repetition and self-similarity
    - Segment homogeneity
    - Structural complexity

    Attributes:
        sample_rate: Target sample rate for audio processing
        hop_length: Hop length for frame-based analysis
        n_fft: FFT size for spectral analysis
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
    ):
        """
        Initialize structure detection calculator.

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

    def compute_spectral_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, np.ndarray]:
        """
        Compute multiple spectral features for structure analysis.

        Args:
            audio: Audio waveform
            sr: Sample rate

        Returns:
            Dictionary of spectral features
        """
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=128
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=20,
            hop_length=self.hop_length
        )

        # Chroma
        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=sr,
            hop_length=self.hop_length
        )

        # Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(
            y=audio,
            sr=sr,
            hop_length=self.hop_length
        )

        return {
            "mel_spectrogram": mel_spec_db,
            "mfcc": mfcc,
            "chroma": chroma,
            "tonnetz": tonnetz,
        }

    def compute_self_similarity_matrix(
        self,
        features: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute self-similarity matrix from features.

        Args:
            features: Feature matrix (n_features x n_frames)
            metric: Distance metric ('cosine', 'euclidean', etc.)

        Returns:
            Self-similarity matrix (n_frames x n_frames)
        """
        # Normalize features
        features = features / (np.linalg.norm(features, axis=0, keepdims=True) + 1e-8)

        if metric == "cosine":
            # Cosine similarity (1 - cosine distance)
            similarity = features.T @ features
        else:
            # Use librosa's sequence module
            similarity = librosa.segment.recurrence_matrix(
                features,
                metric=metric,
                mode='affinity'
            )

        return similarity

    def compute_novelty_curve(
        self,
        features: np.ndarray,
        kernel_size: int = 9
    ) -> np.ndarray:
        """
        Compute novelty curve from features using checkerboard kernel.

        Args:
            features: Feature matrix (n_features x n_frames)
            kernel_size: Size of checkerboard kernel

        Returns:
            Novelty curve (n_frames,)
        """
        # Compute self-similarity matrix
        similarity = self.compute_self_similarity_matrix(features)

        # Use librosa's timelag convolution for novelty
        novelty = librosa.segment.timelag_filter(
            gaussian_filter1d,
            size=kernel_size
        )

        # Compute novelty from similarity matrix
        novelty_curve = np.sum(np.diff(similarity, axis=1), axis=0)
        novelty_curve = np.abs(novelty_curve)

        # Smooth the novelty curve
        novelty_curve = gaussian_filter1d(novelty_curve, sigma=2)

        return novelty_curve

    def detect_boundaries(
        self,
        novelty_curve: np.ndarray,
        sr: int,
        pre_avg: int = 3,
        post_avg: int = 3,
        pre_max: int = 3,
        post_max: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect segment boundaries from novelty curve using peak picking.

        Args:
            novelty_curve: Novelty curve
            sr: Sample rate
            pre_avg: Pre-average window size
            post_avg: Post-average window size
            pre_max: Pre-maximum window size
            post_max: Post-maximum window size

        Returns:
            Tuple of (boundary_frames, boundary_times)
        """
        # Use librosa's peak picking
        boundary_frames = librosa.util.peak_pick(
            novelty_curve,
            pre_max=pre_max,
            post_max=post_max,
            pre_avg=pre_avg,
            post_avg=post_avg,
            delta=0.1,
            wait=10
        )

        # Convert to time
        boundary_times = librosa.frames_to_time(
            boundary_frames,
            sr=sr,
            hop_length=self.hop_length
        )

        return boundary_frames, boundary_times

    def segment_audio(
        self,
        features: np.ndarray,
        n_segments: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment audio using spectral clustering on self-similarity matrix.

        Args:
            features: Feature matrix (n_features x n_frames)
            n_segments: Number of segments (None for automatic)

        Returns:
            Tuple of (segment_labels, segment_boundaries)
        """
        # Compute self-similarity matrix
        similarity = self.compute_self_similarity_matrix(features)

        # Use librosa's agglomerative clustering
        if n_segments is None:
            # Automatic segmentation
            boundary_frames = librosa.segment.agglomerative(
                similarity,
                k=None
            )
        else:
            # Fixed number of segments
            boundary_frames = librosa.segment.agglomerative(
                similarity,
                k=n_segments
            )

        # Create segment labels
        n_frames = features.shape[1]
        segment_labels = np.zeros(n_frames, dtype=int)

        for i, (start, end) in enumerate(zip(boundary_frames[:-1], boundary_frames[1:])):
            segment_labels[start:end] = i

        return segment_labels, boundary_frames

    def calculate_segment_homogeneity(
        self,
        features: np.ndarray,
        segment_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate homogeneity metrics for segments.

        Args:
            features: Feature matrix (n_features x n_frames)
            segment_labels: Segment label for each frame

        Returns:
            Dictionary with homogeneity metrics
        """
        n_segments = len(np.unique(segment_labels))

        segment_variances = []
        segment_sizes = []

        for label in np.unique(segment_labels):
            segment_features = features[:, segment_labels == label]
            if segment_features.shape[1] > 1:
                variance = np.mean(np.var(segment_features, axis=1))
                segment_variances.append(variance)
                segment_sizes.append(segment_features.shape[1])

        avg_homogeneity = 1.0 / (1.0 + np.mean(segment_variances)) if segment_variances else 0.0

        return {
            "n_segments": n_segments,
            "avg_segment_homogeneity": float(avg_homogeneity),
            "avg_segment_variance": float(np.mean(segment_variances)) if segment_variances else 0.0,
            "avg_segment_size": float(np.mean(segment_sizes)) if segment_sizes else 0.0,
        }

    def calculate_repetition_score(
        self,
        similarity: np.ndarray
    ) -> float:
        """
        Calculate repetition score from self-similarity matrix.

        High values on off-diagonal indicate repetitive structure.

        Args:
            similarity: Self-similarity matrix

        Returns:
            Repetition score (0-1)
        """
        # Exclude main diagonal
        mask = ~np.eye(similarity.shape[0], dtype=bool)
        off_diagonal_mean = np.mean(similarity[mask])

        # Apply sigmoid to map to 0-1 range
        repetition_score = 1.0 / (1.0 + np.exp(-5 * (off_diagonal_mean - 0.5)))

        return float(repetition_score)

    def calculate_structural_complexity(
        self,
        novelty_curve: np.ndarray,
        segment_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate structural complexity metrics.

        Args:
            novelty_curve: Novelty curve
            segment_labels: Segment labels

        Returns:
            Dictionary with complexity metrics
        """
        # Number of unique segments
        n_segments = len(np.unique(segment_labels))

        # Novelty strength (average novelty)
        avg_novelty = np.mean(novelty_curve)

        # Novelty variance (how variable are the changes)
        novelty_variance = np.var(novelty_curve)

        # Structural complexity: combination of segment count and novelty
        # Normalize by length
        complexity = n_segments * avg_novelty / len(novelty_curve)

        return {
            "structural_complexity": float(complexity),
            "avg_novelty": float(avg_novelty),
            "novelty_variance": float(novelty_variance),
        }

    def compute(
        self,
        audio_path: Union[str, Path],
        n_segments: Optional[int] = None,
    ) -> Dict[str, Union[float, int, np.ndarray]]:
        """
        Compute comprehensive structure detection metrics.

        Args:
            audio_path: Path to audio file
            n_segments: Number of segments for segmentation (None for automatic)

        Returns:
            Dictionary containing:
                - n_segments: Number of detected segments
                - segment_boundaries: Boundary times in seconds
                - repetition_score: Repetition score (0-1)
                - structural_complexity: Structural complexity score
                - avg_segment_homogeneity: Average segment homogeneity (0-1)
                - And other structure metrics

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If processing fails
        """
        # Load audio
        audio, sr = self._load_audio(audio_path)
        duration = len(audio) / sr

        # Compute features
        features_dict = self.compute_spectral_features(audio, sr)

        # Use MFCC for structure analysis (good general-purpose features)
        features = features_dict["mfcc"]

        # Compute novelty curve
        novelty_curve = self.compute_novelty_curve(features)

        # Detect boundaries
        boundary_frames, boundary_times = self.detect_boundaries(novelty_curve, sr)

        # Segment audio
        segment_labels, segment_boundaries = self.segment_audio(features, n_segments)

        # Compute self-similarity for repetition analysis
        similarity = self.compute_self_similarity_matrix(features)

        # Calculate metrics
        homogeneity_metrics = self.calculate_segment_homogeneity(features, segment_labels)
        repetition_score = self.calculate_repetition_score(similarity)
        complexity_metrics = self.calculate_structural_complexity(novelty_curve, segment_labels)

        # Convert segment boundaries to time
        segment_boundary_times = librosa.frames_to_time(
            segment_boundaries,
            sr=sr,
            hop_length=self.hop_length
        )

        result = {
            "duration": float(duration),
            "n_boundaries": len(boundary_times),
            "boundary_times": boundary_times.tolist(),
            "segment_boundary_times": segment_boundary_times.tolist(),
            "repetition_score": repetition_score,
            **homogeneity_metrics,
            **complexity_metrics,
        }

        return result

    def compute_batch(
        self,
        audio_paths: List[Union[str, Path]]
    ) -> Dict[str, Union[float, List]]:
        """
        Compute structure detection metrics for multiple audio files.

        Args:
            audio_paths: List of audio file paths

        Returns:
            Dictionary containing aggregated metrics:
                - avg_n_segments: Average number of segments
                - avg_repetition_score: Average repetition score
                - avg_structural_complexity: Average structural complexity
                - individual_results: List of individual results

        Raises:
            ValueError: If audio_paths is empty
        """
        if not audio_paths:
            raise ValueError("audio_paths must be non-empty")

        individual_results = []
        n_segments_list = []
        repetition_scores = []
        complexity_scores = []

        for audio_path in audio_paths:
            try:
                result = self.compute(audio_path)
                individual_results.append(result)
                n_segments_list.append(result["n_segments"])
                repetition_scores.append(result["repetition_score"])
                complexity_scores.append(result["structural_complexity"])
            except Exception as e:
                print(f"Warning: Failed to process {audio_path}: {e}")
                continue

        if not individual_results:
            raise RuntimeError("Failed to process any audio files")

        return {
            "avg_n_segments": float(np.mean(n_segments_list)),
            "avg_repetition_score": float(np.mean(repetition_scores)),
            "avg_structural_complexity": float(np.mean(complexity_scores)),
            "std_n_segments": float(np.std(n_segments_list)),
            "std_repetition_score": float(np.std(repetition_scores)),
            "std_structural_complexity": float(np.std(complexity_scores)),
            "n_files": len(individual_results),
            "individual_results": individual_results,
        }


if __name__ == "__main__":
    # Example usage
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python structure.py <audio_file> [audio_file2 ...]")
        sys.exit(1)

    audio_files = [Path(p) for p in sys.argv[1:]]

    calculator = StructureDetectionCalculator()

    if len(audio_files) == 1:
        # Single file analysis
        print(f"\nAnalyzing structure for: {audio_files[0]}")
        result = calculator.compute(audio_files[0])

        print("\nResults:")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Number of Segments: {result['n_segments']}")
        print(f"  Number of Boundaries: {result['n_boundaries']}")
        print(f"  Repetition Score: {result['repetition_score']:.4f}")
        print(f"  Structural Complexity: {result['structural_complexity']:.4f}")
        print(f"  Avg Segment Homogeneity: {result['avg_segment_homogeneity']:.4f}")

        print(f"\n  Segment Boundaries (seconds):")
        for i, time in enumerate(result['segment_boundary_times']):
            print(f"    Segment {i}: {time:.2f}s")
    else:
        # Batch analysis
        print(f"\nAnalyzing structure for {len(audio_files)} files...")
        result = calculator.compute_batch(audio_files)

        print("\nAggregated Results:")
        print(f"  Avg Number of Segments: {result['avg_n_segments']:.2f}")
        print(f"  Avg Repetition Score: {result['avg_repetition_score']:.4f}")
        print(f"  Avg Structural Complexity: {result['avg_structural_complexity']:.4f}")

        print("\nIndividual Results:")
        for i, res in enumerate(result["individual_results"]):
            print(f"\n  File {i+1}: {audio_files[i].name}")
            print(f"    Segments: {res['n_segments']}")
            print(f"    Repetition: {res['repetition_score']:.4f}")
            print(f"    Complexity: {res['structural_complexity']:.4f}")
