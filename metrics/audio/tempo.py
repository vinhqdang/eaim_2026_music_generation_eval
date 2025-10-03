"""
Tempo and beat consistency analysis.

This module evaluates the rhythmic consistency of generated music by analyzing
tempo stability, beat strength, and temporal consistency using librosa and mir_eval.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import librosa
import mir_eval


class TempoConsistencyCalculator:
    """
    Calculate tempo and beat consistency metrics for audio.

    This calculator measures:
    - Tempo estimation and stability
    - Beat tracking accuracy
    - Tempo consistency across time
    - Beat strength and regularity

    Attributes:
        sample_rate: Target sample rate for audio processing
        hop_length: Hop length for frame-based analysis
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
    ):
        """
        Initialize tempo consistency calculator.

        Args:
            sample_rate: Target sample rate for audio
            hop_length: Hop length for STFT analysis
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length

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

    def estimate_tempo(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate global tempo and local tempo variations.

        Args:
            audio: Audio waveform
            sr: Sample rate

        Returns:
            Tuple of (global_tempo, local_tempos)
        """
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=sr,
            hop_length=self.hop_length
        )

        # Estimate global tempo
        tempo, _ = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length
        )

        # Estimate local tempo using tempogram
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length
        )

        # Get dominant tempo at each time frame
        local_tempos = librosa.feature.tempo(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
            aggregate=None
        )

        return float(tempo), local_tempos

    def detect_beats(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect beat positions in audio.

        Args:
            audio: Audio waveform
            sr: Sample rate

        Returns:
            Tuple of (beat_times, beat_frames)
        """
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio,
            sr=sr,
            hop_length=self.hop_length
        )

        beat_times = librosa.frames_to_time(
            beat_frames,
            sr=sr,
            hop_length=self.hop_length
        )

        return beat_times, beat_frames

    def calculate_tempo_stability(
        self,
        local_tempos: np.ndarray,
        global_tempo: float
    ) -> Dict[str, float]:
        """
        Calculate tempo stability metrics.

        Args:
            local_tempos: Array of local tempo estimates
            global_tempo: Global tempo estimate

        Returns:
            Dictionary with stability metrics
        """
        # Remove NaN values
        local_tempos = local_tempos[~np.isnan(local_tempos)]

        if len(local_tempos) == 0:
            return {
                "tempo_std": float("nan"),
                "tempo_cv": float("nan"),
                "tempo_range": float("nan"),
                "tempo_stability": 0.0,
            }

        tempo_std = float(np.std(local_tempos))
        tempo_mean = float(np.mean(local_tempos))
        tempo_cv = tempo_std / tempo_mean if tempo_mean > 0 else float("nan")
        tempo_range = float(np.ptp(local_tempos))

        # Stability score: inverse of coefficient of variation (0-1, higher is better)
        tempo_stability = 1.0 / (1.0 + tempo_cv) if not np.isnan(tempo_cv) else 0.0

        return {
            "tempo_std": tempo_std,
            "tempo_cv": tempo_cv,
            "tempo_range": tempo_range,
            "tempo_stability": float(tempo_stability),
        }

    def calculate_beat_strength(
        self,
        audio: np.ndarray,
        sr: int,
        beat_times: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate beat strength and regularity metrics.

        Args:
            audio: Audio waveform
            sr: Sample rate
            beat_times: Beat time positions

        Returns:
            Dictionary with beat strength metrics
        """
        if len(beat_times) < 2:
            return {
                "avg_beat_strength": 0.0,
                "beat_strength_std": 0.0,
                "beat_regularity": 0.0,
            }

        # Compute onset strength
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=sr,
            hop_length=self.hop_length
        )

        # Get onset strength at beat positions
        beat_frames = librosa.time_to_frames(
            beat_times,
            sr=sr,
            hop_length=self.hop_length
        )

        # Ensure beat_frames are within bounds
        beat_frames = beat_frames[beat_frames < len(onset_env)]

        if len(beat_frames) < 2:
            return {
                "avg_beat_strength": 0.0,
                "beat_strength_std": 0.0,
                "beat_regularity": 0.0,
            }

        beat_strengths = onset_env[beat_frames]

        # Calculate inter-beat intervals (IBI)
        ibis = np.diff(beat_times)

        # Beat regularity: inverse of IBI coefficient of variation
        ibi_mean = np.mean(ibis)
        ibi_std = np.std(ibis)
        ibi_cv = ibi_std / ibi_mean if ibi_mean > 0 else float("nan")
        beat_regularity = 1.0 / (1.0 + ibi_cv) if not np.isnan(ibi_cv) else 0.0

        return {
            "avg_beat_strength": float(np.mean(beat_strengths)),
            "beat_strength_std": float(np.std(beat_strengths)),
            "beat_regularity": float(beat_regularity),
            "avg_ibi": float(ibi_mean),
            "ibi_std": float(ibi_std),
        }

    def compare_beats(
        self,
        reference_beats: np.ndarray,
        estimated_beats: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare estimated beats against reference beats using mir_eval.

        Args:
            reference_beats: Reference beat times
            estimated_beats: Estimated beat times

        Returns:
            Dictionary with beat tracking metrics
        """
        try:
            # mir_eval beat evaluation
            scores = mir_eval.beat.evaluate(reference_beats, estimated_beats)

            return {
                "f_measure": float(scores["F-measure"]),
                "cemgil": float(scores["Cemgil"]),
                "goto": float(scores["Goto"]),
                "p_score": float(scores["P-score"]),
                "continuity_pll": float(scores["Continuity PLL"]),
                "continuity_cmlc": float(scores["Continuity CMLc"]),
                "continuity_cmlt": float(scores["Continuity CMLt"]),
                "aml": float(scores["AMLt"]),
            }
        except Exception as e:
            print(f"Warning: Beat comparison failed: {e}")
            return {}

    def compute(
        self,
        audio_path: Union[str, Path],
        reference_beats: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[float, int, np.ndarray]]:
        """
        Compute comprehensive tempo and beat consistency metrics.

        Args:
            audio_path: Path to audio file
            reference_beats: Optional reference beat times for evaluation

        Returns:
            Dictionary containing:
                - global_tempo: Estimated global tempo (BPM)
                - tempo_stability: Tempo stability score (0-1)
                - beat_regularity: Beat regularity score (0-1)
                - avg_beat_strength: Average beat strength
                - n_beats: Number of detected beats
                - duration: Audio duration in seconds
                - And other tempo/beat metrics

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If processing fails
        """
        # Load audio
        audio, sr = self._load_audio(audio_path)
        duration = len(audio) / sr

        # Estimate tempo
        global_tempo, local_tempos = self.estimate_tempo(audio, sr)

        # Detect beats
        beat_times, beat_frames = self.detect_beats(audio, sr)

        # Calculate tempo stability
        tempo_metrics = self.calculate_tempo_stability(local_tempos, global_tempo)

        # Calculate beat strength
        beat_metrics = self.calculate_beat_strength(audio, sr, beat_times)

        # Build result dictionary
        result = {
            "global_tempo": global_tempo,
            "n_beats": len(beat_times),
            "duration": float(duration),
            "beats_per_second": len(beat_times) / duration if duration > 0 else 0.0,
            **tempo_metrics,
            **beat_metrics,
        }

        # Compare with reference beats if provided
        if reference_beats is not None and len(reference_beats) > 0:
            comparison = self.compare_beats(reference_beats, beat_times)
            result.update(comparison)

        return result

    def compute_batch(
        self,
        audio_paths: List[Union[str, Path]]
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Compute tempo consistency metrics for multiple audio files.

        Args:
            audio_paths: List of audio file paths

        Returns:
            Dictionary containing aggregated metrics:
                - avg_tempo_stability: Average tempo stability
                - avg_beat_regularity: Average beat regularity
                - tempo_consistency_across_files: Consistency of tempo across files
                - individual_results: List of individual results

        Raises:
            ValueError: If audio_paths is empty
        """
        if not audio_paths:
            raise ValueError("audio_paths must be non-empty")

        individual_results = []
        tempos = []
        tempo_stabilities = []
        beat_regularities = []

        for audio_path in audio_paths:
            try:
                result = self.compute(audio_path)
                individual_results.append(result)
                tempos.append(result["global_tempo"])
                tempo_stabilities.append(result["tempo_stability"])
                beat_regularities.append(result["beat_regularity"])
            except Exception as e:
                print(f"Warning: Failed to process {audio_path}: {e}")
                continue

        if not individual_results:
            raise RuntimeError("Failed to process any audio files")

        # Calculate cross-file consistency
        tempos = np.array(tempos)
        tempo_mean = np.mean(tempos)
        tempo_std = np.std(tempos)
        tempo_cv = tempo_std / tempo_mean if tempo_mean > 0 else float("nan")
        tempo_consistency = 1.0 / (1.0 + tempo_cv) if not np.isnan(tempo_cv) else 0.0

        return {
            "avg_tempo_stability": float(np.mean(tempo_stabilities)),
            "avg_beat_regularity": float(np.mean(beat_regularities)),
            "avg_global_tempo": float(tempo_mean),
            "tempo_std_across_files": float(tempo_std),
            "tempo_consistency_across_files": float(tempo_consistency),
            "n_files": len(individual_results),
            "individual_results": individual_results,
        }


if __name__ == "__main__":
    # Example usage
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python tempo.py <audio_file> [audio_file2 ...]")
        sys.exit(1)

    audio_files = [Path(p) for p in sys.argv[1:]]

    calculator = TempoConsistencyCalculator()

    if len(audio_files) == 1:
        # Single file analysis
        print(f"\nAnalyzing tempo for: {audio_files[0]}")
        result = calculator.compute(audio_files[0])

        print("\nResults:")
        for key, value in result.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    else:
        # Batch analysis
        print(f"\nAnalyzing tempo for {len(audio_files)} files...")
        result = calculator.compute_batch(audio_files)

        print("\nAggregated Results:")
        for key, value in result.items():
            if key != "individual_results" and isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        print("\nIndividual Results:")
        for i, res in enumerate(result["individual_results"]):
            print(f"\n  File {i+1}: {audio_files[i].name}")
            print(f"    Tempo: {res['global_tempo']:.2f} BPM")
            print(f"    Stability: {res['tempo_stability']:.4f}")
            print(f"    Regularity: {res['beat_regularity']:.4f}")
