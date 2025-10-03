"""
Voice-leading cost calculation for MIDI files.

This module provides metrics for analyzing voice-leading quality,
including parallel motion detection, voice crossing, and smooth motion costs.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from collections import defaultdict
import muspy
import miditoolkit
import pretty_midi


class VoiceLeadingMetricCalculator:
    """
    Calculate voice-leading quality metrics from MIDI files.

    This class analyzes voice-leading by tracking individual voices,
    computing motion costs, and detecting voice-leading errors.
    """

    def __init__(
        self,
        max_voices: int = 4,
        min_duration: float = 0.1,
        time_resolution: float = 0.05
    ):
        """
        Initialize the voice-leading metric calculator.

        Args:
            max_voices: Maximum number of voices to track
            min_duration: Minimum note duration in seconds to consider
            time_resolution: Time resolution in seconds for voice tracking
        """
        self.max_voices = max_voices
        self.min_duration = min_duration
        self.time_resolution = time_resolution

    def compute(
        self,
        midi_path: Union[str, Path],
        seed_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Compute voice-leading cost metrics.

        Args:
            midi_path: Path to the MIDI file to analyze
            seed_path: Optional path to seed MIDI for comparison

        Returns:
            Dictionary containing:
                - total_voice_leading_cost: Total cost of all voice motions
                - avg_voice_leading_cost_per_transition: Average cost per transition
                - parallel_motion_violations: Count of parallel 5ths/octaves
                - voice_crossing_count: Number of voice crossings
                - large_leap_count: Number of leaps larger than an octave
                - smooth_motion_ratio: Ratio of stepwise motions
                - voice_independence: Measure of voice independence (0-1)
                - avg_semitone_distance: Average distance in semitones
                - seed_voice_leading_similarity: Similarity to seed (if provided)
        """
        try:
            # Load MIDI file
            midi_toolkit = miditoolkit.MidiFile(str(midi_path))
            midi_pretty = pretty_midi.PrettyMIDI(str(midi_path))

            # Extract voice segments
            voice_segments = self._extract_voice_segments(midi_toolkit, midi_pretty)

            # Track voices over time
            voice_timeline = self._build_voice_timeline(voice_segments)

            # Calculate voice-leading costs
            vl_metrics = self._calculate_voice_leading_costs(voice_timeline)

            # Detect voice-leading violations
            violation_metrics = self._detect_violations(voice_timeline)

            # Calculate independence and smoothness
            quality_metrics = self._calculate_quality_metrics(voice_timeline)

            # Combine all metrics
            metrics = {**vl_metrics, **violation_metrics, **quality_metrics}

            # If seed is provided, calculate similarity
            if seed_path:
                seed_similarity = self._calculate_seed_similarity(
                    midi_path, seed_path
                )
                metrics['seed_voice_leading_similarity'] = seed_similarity

            return metrics

        except Exception as e:
            print(f"Error computing voice-leading metrics for {midi_path}: {e}")
            return self._get_default_metrics(seed_path is not None)

    def _extract_voice_segments(
        self,
        midi_toolkit: miditoolkit.MidiFile,
        midi_pretty: pretty_midi.PrettyMIDI
    ) -> List[Tuple[float, float, int, int]]:
        """
        Extract voice segments (start, end, pitch, track) from MIDI.

        Args:
            midi_toolkit: MidiToolkit MidiFile object
            midi_pretty: PrettyMIDI object

        Returns:
            List of (start_time, end_time, pitch, track_id) tuples
        """
        segments = []

        for track_id, instrument in enumerate(midi_toolkit.instruments):
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                start_time = note.start / midi_toolkit.ticks_per_beat
                end_time = note.end / midi_toolkit.ticks_per_beat
                duration = end_time - start_time

                if duration >= self.min_duration:
                    segments.append((start_time, end_time, note.pitch, track_id))

        # Sort by start time
        segments.sort(key=lambda x: x[0])

        return segments

    def _build_voice_timeline(
        self,
        segments: List[Tuple[float, float, int, int]]
    ) -> List[List[Optional[int]]]:
        """
        Build a timeline of voice assignments.

        Args:
            segments: List of (start_time, end_time, pitch, track_id) tuples

        Returns:
            List of time slices, each containing pitches for active voices
        """
        if not segments:
            return []

        # Determine time range
        start_time = segments[0][0]
        end_time = max(seg[1] for seg in segments)

        # Create time slices
        num_slices = int((end_time - start_time) / self.time_resolution) + 1
        timeline = []

        for i in range(num_slices):
            current_time = start_time + i * self.time_resolution
            active_notes = []

            # Find all notes active at this time
            for seg_start, seg_end, pitch, track_id in segments:
                if seg_start <= current_time < seg_end:
                    active_notes.append((pitch, track_id, seg_start))

            # Sort by pitch (highest to lowest) for voice assignment
            active_notes.sort(reverse=True)

            # Assign to voices
            voices = [None] * self.max_voices
            for idx, (pitch, track_id, seg_start) in enumerate(active_notes[:self.max_voices]):
                voices[idx] = pitch

            timeline.append(voices)

        return timeline

    def _calculate_voice_leading_costs(
        self,
        timeline: List[List[Optional[int]]]
    ) -> Dict[str, float]:
        """
        Calculate voice-leading motion costs.

        Args:
            timeline: List of voice states over time

        Returns:
            Dictionary of voice-leading cost metrics
        """
        if len(timeline) < 2:
            return {
                'total_voice_leading_cost': 0.0,
                'avg_voice_leading_cost_per_transition': 0.0,
                'avg_semitone_distance': 0.0
            }

        total_cost = 0.0
        transition_count = 0
        semitone_distances = []

        for i in range(len(timeline) - 1):
            current_voices = timeline[i]
            next_voices = timeline[i + 1]

            # Calculate cost for each voice transition
            for voice_idx in range(self.max_voices):
                current_pitch = current_voices[voice_idx]
                next_pitch = next_voices[voice_idx]

                if current_pitch is not None and next_pitch is not None:
                    # Calculate semitone distance
                    distance = abs(next_pitch - current_pitch)
                    semitone_distances.append(distance)

                    # Cost function: prefer stepwise motion
                    if distance == 0:
                        cost = 0.0  # Repeated note
                    elif distance <= 2:
                        cost = 1.0  # Step (desirable)
                    elif distance <= 4:
                        cost = 2.0  # Small leap
                    elif distance <= 7:
                        cost = 3.0  # Medium leap
                    elif distance <= 12:
                        cost = 5.0  # Large leap (octave)
                    else:
                        cost = 10.0  # Very large leap (discouraged)

                    total_cost += cost
                    transition_count += 1

        avg_cost = total_cost / transition_count if transition_count > 0 else 0.0
        avg_distance = np.mean(semitone_distances) if semitone_distances else 0.0

        return {
            'total_voice_leading_cost': float(total_cost),
            'avg_voice_leading_cost_per_transition': float(avg_cost),
            'avg_semitone_distance': float(avg_distance)
        }

    def _detect_violations(
        self,
        timeline: List[List[Optional[int]]]
    ) -> Dict[str, float]:
        """
        Detect voice-leading violations.

        Args:
            timeline: List of voice states over time

        Returns:
            Dictionary of violation counts
        """
        if len(timeline) < 2:
            return {
                'parallel_motion_violations': 0,
                'voice_crossing_count': 0,
                'large_leap_count': 0
            }

        parallel_violations = 0
        voice_crossings = 0
        large_leaps = 0

        for i in range(len(timeline) - 1):
            current_voices = timeline[i]
            next_voices = timeline[i + 1]

            # Check for parallel fifths and octaves
            for v1 in range(self.max_voices):
                for v2 in range(v1 + 1, self.max_voices):
                    if (current_voices[v1] is not None and
                        current_voices[v2] is not None and
                        next_voices[v1] is not None and
                        next_voices[v2] is not None):

                        # Calculate intervals
                        current_interval = abs(current_voices[v1] - current_voices[v2])
                        next_interval = abs(next_voices[v1] - next_voices[v2])

                        # Check for parallel perfect intervals (P5 or P8)
                        if current_interval in [7, 12] and current_interval == next_interval:
                            # Verify both voices moved in same direction
                            motion1 = next_voices[v1] - current_voices[v1]
                            motion2 = next_voices[v2] - current_voices[v2]
                            if motion1 * motion2 > 0:  # Same direction
                                parallel_violations += 1

            # Check for voice crossings
            for v1 in range(self.max_voices):
                for v2 in range(v1 + 1, self.max_voices):
                    if (current_voices[v1] is not None and
                        current_voices[v2] is not None):
                        # v1 should be higher than v2 (voices sorted by pitch)
                        if current_voices[v1] < current_voices[v2]:
                            voice_crossings += 1

            # Check for large leaps (more than an octave)
            for v in range(self.max_voices):
                if current_voices[v] is not None and next_voices[v] is not None:
                    leap = abs(next_voices[v] - current_voices[v])
                    if leap > 12:
                        large_leaps += 1

        return {
            'parallel_motion_violations': int(parallel_violations),
            'voice_crossing_count': int(voice_crossings),
            'large_leap_count': int(large_leaps)
        }

    def _calculate_quality_metrics(
        self,
        timeline: List[List[Optional[int]]]
    ) -> Dict[str, float]:
        """
        Calculate voice-leading quality metrics.

        Args:
            timeline: List of voice states over time

        Returns:
            Dictionary of quality metrics
        """
        if len(timeline) < 2:
            return {
                'smooth_motion_ratio': 0.0,
                'voice_independence': 0.0
            }

        stepwise_motions = 0
        total_motions = 0
        motion_vectors = [[] for _ in range(self.max_voices)]

        for i in range(len(timeline) - 1):
            current_voices = timeline[i]
            next_voices = timeline[i + 1]

            for voice_idx in range(self.max_voices):
                current_pitch = current_voices[voice_idx]
                next_pitch = next_voices[voice_idx]

                if current_pitch is not None and next_pitch is not None:
                    motion = next_pitch - current_pitch
                    motion_vectors[voice_idx].append(motion)

                    # Count stepwise motions (up to 2 semitones)
                    if abs(motion) <= 2:
                        stepwise_motions += 1
                    total_motions += 1

        smooth_ratio = stepwise_motions / total_motions if total_motions > 0 else 0.0

        # Calculate voice independence (correlation between voice motions)
        independence_scores = []
        for v1 in range(self.max_voices):
            for v2 in range(v1 + 1, self.max_voices):
                if len(motion_vectors[v1]) > 1 and len(motion_vectors[v2]) > 1:
                    # Align lengths
                    min_len = min(len(motion_vectors[v1]), len(motion_vectors[v2]))
                    vec1 = np.array(motion_vectors[v1][:min_len])
                    vec2 = np.array(motion_vectors[v2][:min_len])

                    if np.std(vec1) > 0 and np.std(vec2) > 0:
                        # Calculate correlation
                        correlation = np.corrcoef(vec1, vec2)[0, 1]
                        # Independence = 1 - |correlation|
                        independence = 1.0 - abs(correlation)
                        independence_scores.append(independence)

        avg_independence = np.mean(independence_scores) if independence_scores else 0.5

        return {
            'smooth_motion_ratio': float(smooth_ratio),
            'voice_independence': float(avg_independence)
        }

    def _calculate_seed_similarity(
        self,
        continuation_path: Union[str, Path],
        seed_path: Union[str, Path]
    ) -> float:
        """
        Calculate voice-leading similarity between seed and continuation.

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

            # Compare key voice-leading characteristics
            features = [
                'avg_voice_leading_cost_per_transition',
                'smooth_motion_ratio',
                'voice_independence',
                'avg_semitone_distance'
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
            print(f"Error calculating seed voice-leading similarity: {e}")
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
            'total_voice_leading_cost': 0.0,
            'avg_voice_leading_cost_per_transition': 0.0,
            'parallel_motion_violations': 0,
            'voice_crossing_count': 0,
            'large_leap_count': 0,
            'smooth_motion_ratio': 0.0,
            'voice_independence': 0.0,
            'avg_semitone_distance': 0.0
        }

        if has_seed:
            metrics['seed_voice_leading_similarity'] = 0.0

        return metrics
