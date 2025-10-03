"""
Rhythm regularity and syncopation metrics for MIDI files.

This module provides metrics for analyzing rhythmic patterns,
including regularity, syncopation, groove, and microtiming.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
import muspy
import miditoolkit
import pretty_midi
from scipy import stats
from scipy.signal import find_peaks


class RhythmMetricCalculator:
    """
    Calculate rhythm regularity and syncopation metrics from MIDI files.

    This class analyzes rhythmic patterns including onset distributions,
    syncopation strength, groove consistency, and microtiming variations.
    """

    def __init__(
        self,
        resolution: int = 24,
        beat_divisions: List[int] = None,
        min_ioi: float = 0.05
    ):
        """
        Initialize the rhythm metric calculator.

        Args:
            resolution: Ticks per beat for analysis
            beat_divisions: List of beat subdivisions to analyze [4, 8, 16]
            min_ioi: Minimum inter-onset interval in seconds
        """
        self.resolution = resolution
        self.beat_divisions = beat_divisions or [4, 8, 16]
        self.min_ioi = min_ioi

    def compute(
        self,
        midi_path: Union[str, Path],
        seed_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Compute rhythm regularity and syncopation metrics.

        Args:
            midi_path: Path to the MIDI file to analyze
            seed_path: Optional path to seed MIDI for comparison

        Returns:
            Dictionary containing:
                - rhythm_regularity: Overall rhythmic regularity (0-1)
                - syncopation_strength: Degree of syncopation
                - groove_consistency: Consistency of groove pattern
                - onset_density: Average onsets per beat
                - ioi_entropy: Entropy of inter-onset intervals
                - microtiming_variation: Variation from strict grid
                - beat_strength_clarity: Clarity of beat hierarchy
                - rhythmic_complexity: Overall rhythmic complexity
                - seed_rhythm_similarity: Similarity to seed (if provided)
        """
        try:
            # Load MIDI file
            music = muspy.read_midi(str(midi_path))
            midi_toolkit = miditoolkit.MidiFile(str(midi_path))
            midi_pretty = pretty_midi.PrettyMIDI(str(midi_path))

            # Extract onset times and beat information
            onsets, tempo = self._extract_onsets_and_tempo(
                music, midi_toolkit, midi_pretty
            )

            # Calculate regularity metrics
            regularity_metrics = self._calculate_regularity_metrics(
                onsets, tempo
            )

            # Calculate syncopation metrics
            syncopation_metrics = self._calculate_syncopation_metrics(
                onsets, tempo
            )

            # Calculate groove metrics
            groove_metrics = self._calculate_groove_metrics(
                onsets, tempo
            )

            # Calculate complexity metrics
            complexity_metrics = self._calculate_complexity_metrics(
                onsets
            )

            # Combine all metrics
            metrics = {
                **regularity_metrics,
                **syncopation_metrics,
                **groove_metrics,
                **complexity_metrics
            }

            # If seed is provided, calculate similarity
            if seed_path:
                seed_similarity = self._calculate_seed_similarity(
                    midi_path, seed_path
                )
                metrics['seed_rhythm_similarity'] = seed_similarity

            return metrics

        except Exception as e:
            print(f"Error computing rhythm metrics for {midi_path}: {e}")
            return self._get_default_metrics(seed_path is not None)

    def _extract_onsets_and_tempo(
        self,
        music: muspy.Music,
        midi_toolkit: miditoolkit.MidiFile,
        midi_pretty: pretty_midi.PrettyMIDI
    ) -> Tuple[np.ndarray, float]:
        """
        Extract onset times and estimate tempo.

        Args:
            music: MusPy Music object
            midi_toolkit: MidiToolkit MidiFile object
            midi_pretty: PrettyMIDI object

        Returns:
            Tuple of (onset_times_array, estimated_tempo)
        """
        onsets = []

        # Extract onsets from all tracks
        for track in music.tracks:
            for note in track.notes:
                onset_time = note.time / music.resolution  # Convert to beats
                onsets.append(onset_time)

        # Fallback to miditoolkit if needed
        if not onsets:
            for instrument in midi_toolkit.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        onset_time = note.start / midi_toolkit.ticks_per_beat
                        onsets.append(onset_time)

        onsets = np.array(sorted(onsets))

        # Estimate tempo
        if hasattr(music, 'tempos') and music.tempos:
            tempo = music.tempos[0].tempo
        elif midi_pretty.get_tempo_changes()[1].size > 0:
            tempo = midi_pretty.get_tempo_changes()[1][0]
        else:
            tempo = 120.0  # Default tempo

        return onsets, tempo

    def _calculate_regularity_metrics(
        self,
        onsets: np.ndarray,
        tempo: float
    ) -> Dict[str, float]:
        """
        Calculate rhythmic regularity metrics.

        Args:
            onsets: Array of onset times in beats
            tempo: Tempo in BPM

        Returns:
            Dictionary of regularity metrics
        """
        if len(onsets) < 2:
            return {
                'rhythm_regularity': 0.0,
                'microtiming_variation': 0.0,
                'beat_strength_clarity': 0.0
            }

        # Calculate inter-onset intervals
        iois = np.diff(onsets)

        # Quantize onsets to grid for regularity calculation
        grid_resolutions = [1/div for div in self.beat_divisions]
        regularity_scores = []

        for grid_res in grid_resolutions:
            # Quantize onsets to this grid
            quantized = np.round(onsets / grid_res) * grid_res
            # Calculate deviation from grid
            deviations = np.abs(onsets - quantized)
            # Regularity = 1 - mean deviation
            regularity = 1.0 - np.clip(np.mean(deviations) / grid_res, 0, 1)
            regularity_scores.append(regularity)

        # Overall regularity is max across grid resolutions
        rhythm_regularity = float(np.max(regularity_scores))

        # Microtiming variation (deviation from 16th note grid)
        finest_grid = 1 / 16
        quantized_fine = np.round(onsets / finest_grid) * finest_grid
        microtiming_dev = np.std(onsets - quantized_fine) if len(onsets) > 1 else 0.0
        # Normalize to [0, 1], where 0 = perfect timing
        microtiming_variation = float(np.clip(microtiming_dev / finest_grid, 0, 1))

        # Beat strength clarity using autocorrelation
        if len(onsets) > 10:
            # Create onset strength signal
            duration = onsets[-1] - onsets[0]
            num_bins = int(duration * 16)  # 16th note resolution
            onset_signal = np.zeros(num_bins)

            for onset in onsets:
                bin_idx = int((onset - onsets[0]) * 16)
                if 0 <= bin_idx < num_bins:
                    onset_signal[bin_idx] = 1

            # Compute autocorrelation
            autocorr = np.correlate(onset_signal, onset_signal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find peaks in autocorrelation
            peaks, properties = find_peaks(autocorr, distance=8, prominence=1)

            if len(peaks) > 0:
                # Clarity = prominence of main peaks
                clarity = float(np.mean(properties['prominences'][:3]) / np.max(autocorr))
            else:
                clarity = 0.0
        else:
            clarity = 0.0

        beat_strength_clarity = float(np.clip(clarity, 0, 1))

        return {
            'rhythm_regularity': rhythm_regularity,
            'microtiming_variation': microtiming_variation,
            'beat_strength_clarity': beat_strength_clarity
        }

    def _calculate_syncopation_metrics(
        self,
        onsets: np.ndarray,
        tempo: float
    ) -> Dict[str, float]:
        """
        Calculate syncopation strength metrics.

        Args:
            onsets: Array of onset times in beats
            tempo: Tempo in BPM

        Returns:
            Dictionary of syncopation metrics
        """
        if len(onsets) < 2:
            return {'syncopation_strength': 0.0}

        # Define metrical hierarchy weights
        # Stronger beats have higher weights
        def get_metrical_weight(beat_position: float) -> float:
            """Calculate metrical weight for a beat position."""
            # Check if on downbeat (measure start)
            if beat_position % 4 < 0.05:
                return 4.0
            # Check if on beat
            elif beat_position % 1 < 0.05:
                return 3.0
            # Check if on eighth note
            elif beat_position % 0.5 < 0.05:
                return 2.0
            # Check if on sixteenth note
            elif beat_position % 0.25 < 0.05:
                return 1.0
            # Off-grid (syncopated)
            else:
                return 0.5

        # Calculate syncopation using Longuet-Higgins & Lee model
        syncopation_scores = []

        for i in range(len(onsets) - 1):
            onset_pos = onsets[i] % 4  # Position within measure
            next_pos = onsets[i + 1] % 4

            # Metrical weight of current onset
            weight_current = get_metrical_weight(onset_pos)

            # Check if next note is on stronger beat
            weight_next = get_metrical_weight(next_pos)

            # Syncopation occurs when weak beat is followed by stronger beat
            if weight_next > weight_current:
                syncopation = weight_next - weight_current
                syncopation_scores.append(syncopation)

        if syncopation_scores:
            # Average syncopation strength
            syncopation_strength = float(np.mean(syncopation_scores) / 4.0)
        else:
            syncopation_strength = 0.0

        return {'syncopation_strength': syncopation_strength}

    def _calculate_groove_metrics(
        self,
        onsets: np.ndarray,
        tempo: float
    ) -> Dict[str, float]:
        """
        Calculate groove consistency metrics.

        Args:
            onsets: Array of onset times in beats
            tempo: Tempo in BPM

        Returns:
            Dictionary of groove metrics
        """
        if len(onsets) < 8:
            return {
                'groove_consistency': 0.0,
                'onset_density': 0.0
            }

        # Calculate onset density (onsets per beat)
        duration = onsets[-1] - onsets[0]
        onset_density = len(onsets) / duration if duration > 0 else 0.0

        # Extract groove pattern (repeating rhythmic unit)
        # Use 1-bar (4 beats) as groove unit
        bar_length = 4.0
        num_bars = int(duration / bar_length)

        if num_bars > 1:
            # Quantize onsets within each bar
            groove_patterns = []

            for bar_idx in range(num_bars):
                bar_start = onsets[0] + bar_idx * bar_length
                bar_end = bar_start + bar_length

                # Get onsets in this bar
                bar_onsets = onsets[(onsets >= bar_start) & (onsets < bar_end)]
                # Normalize to bar-relative positions
                bar_relative = (bar_onsets - bar_start) % bar_length

                # Quantize to 16th notes
                quantized = np.round(bar_relative * 4) / 4  # 16th note grid
                groove_patterns.append(set(quantized))

            # Calculate groove consistency (Jaccard similarity between bars)
            if len(groove_patterns) > 1:
                similarities = []
                for i in range(len(groove_patterns) - 1):
                    set1 = groove_patterns[i]
                    set2 = groove_patterns[i + 1]
                    if len(set1) > 0 or len(set2) > 0:
                        jaccard = len(set1 & set2) / len(set1 | set2)
                        similarities.append(jaccard)

                groove_consistency = float(np.mean(similarities)) if similarities else 0.0
            else:
                groove_consistency = 1.0
        else:
            groove_consistency = 1.0

        return {
            'groove_consistency': groove_consistency,
            'onset_density': float(onset_density)
        }

    def _calculate_complexity_metrics(
        self,
        onsets: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate rhythmic complexity metrics.

        Args:
            onsets: Array of onset times in beats

        Returns:
            Dictionary of complexity metrics
        """
        if len(onsets) < 2:
            return {
                'ioi_entropy': 0.0,
                'rhythmic_complexity': 0.0
            }

        # Calculate inter-onset intervals
        iois = np.diff(onsets)

        # Quantize IOIs to 16th notes
        quantized_iois = np.round(iois * 16) / 16

        # Calculate IOI entropy
        ioi_counts = Counter(quantized_iois)
        total = sum(ioi_counts.values())
        ioi_probs = np.array([count / total for count in ioi_counts.values()])

        if len(ioi_probs) > 0:
            ioi_entropy = -np.sum(ioi_probs * np.log2(ioi_probs))
        else:
            ioi_entropy = 0.0

        # Overall rhythmic complexity (combination of metrics)
        # High complexity = high entropy + high onset density + moderate syncopation
        complexity = ioi_entropy / np.log2(len(self.beat_divisions) * 4)  # Normalize

        return {
            'ioi_entropy': float(ioi_entropy),
            'rhythmic_complexity': float(np.clip(complexity, 0, 1))
        }

    def _calculate_seed_similarity(
        self,
        continuation_path: Union[str, Path],
        seed_path: Union[str, Path]
    ) -> float:
        """
        Calculate rhythmic similarity between seed and continuation.

        Args:
            continuation_path: Path to continuation MIDI
            seed_path: Path to seed MIDI

        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Compute metrics for both files
            seed_metrics = self.compute(seed_path, seed_path=None)
            cont_metrics = self.compute(continuation_path, seed_path=None)

            # Compare key rhythmic characteristics
            features = [
                'rhythm_regularity',
                'syncopation_strength',
                'groove_consistency',
                'onset_density',
                'rhythmic_complexity'
            ]

            similarities = []
            for feature in features:
                seed_val = seed_metrics.get(feature, 0.0)
                cont_val = cont_metrics.get(feature, 0.0)

                # Calculate normalized difference
                max_val = max(abs(seed_val), abs(cont_val), 1.0)
                diff = abs(seed_val - cont_val) / max_val
                similarity = 1.0 - diff
                similarities.append(similarity)

            return float(np.mean(similarities))

        except Exception as e:
            print(f"Error calculating seed rhythm similarity: {e}")
            return 0.0

    def _get_default_metrics(self, has_seed: bool = False) -> Dict[str, float]:
        """
        Get default metrics when computation fails.

        Args:
            has_seed: Whether seed comparison metrics should be included

        Returns:
            Dictionary of default metric values
        """
        metrics = {
            'rhythm_regularity': 0.0,
            'syncopation_strength': 0.0,
            'groove_consistency': 0.0,
            'onset_density': 0.0,
            'ioi_entropy': 0.0,
            'microtiming_variation': 0.0,
            'beat_strength_clarity': 0.0,
            'rhythmic_complexity': 0.0
        }

        if has_seed:
            metrics['seed_rhythm_similarity'] = 0.0

        return metrics
