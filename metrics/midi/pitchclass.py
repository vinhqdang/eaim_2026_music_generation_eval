"""
Pitch-class and chord consistency metrics for MIDI files.

This module provides metrics for analyzing pitch-class distributions,
chord progressions, and harmonic consistency in symbolic music.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from collections import Counter
import muspy
import miditoolkit
import pretty_midi


class PitchClassMetricCalculator:
    """
    Calculate pitch-class and chord consistency metrics from MIDI files.

    This class analyzes harmonic content using pitch-class distributions,
    chord extraction, and consistency measures.
    """

    def __init__(
        self,
        resolution: int = 24,
        chord_threshold: float = 0.1,
        min_duration: float = 0.25
    ):
        """
        Initialize the pitch-class metric calculator.

        Args:
            resolution: Time resolution in ticks per beat for analysis
            chord_threshold: Minimum duration ratio to consider a note part of a chord
            min_duration: Minimum note duration in seconds to consider
        """
        self.resolution = resolution
        self.chord_threshold = chord_threshold
        self.min_duration = min_duration

    def compute(
        self,
        midi_path: Union[str, Path],
        seed_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Compute pitch-class and chord consistency metrics.

        Args:
            midi_path: Path to the MIDI file to analyze
            seed_path: Optional path to seed MIDI for comparison

        Returns:
            Dictionary containing:
                - pitch_class_entropy: Entropy of pitch class distribution
                - pitch_class_histogram_uniformity: Uniformity measure
                - dominant_pitch_classes: Number of dominant pitch classes (>10% occurrence)
                - chord_progression_consistency: Consistency of chord progressions
                - chord_diversity: Number of unique chord types
                - harmonic_rhythm_regularity: Regularity of harmonic changes
                - pitch_class_transition_entropy: Entropy of PC transitions
                - seed_continuation_consistency: Consistency with seed (if provided)
        """
        try:
            # Load MIDI file using multiple libraries for robust parsing
            music = muspy.read_midi(str(midi_path))
            midi_toolkit = miditoolkit.MidiFile(str(midi_path))
            midi_pretty = pretty_midi.PrettyMIDI(str(midi_path))

            # Extract pitch-class information
            pitch_classes = self._extract_pitch_classes(music, midi_toolkit)

            # Calculate basic pitch-class metrics
            pc_metrics = self._calculate_pitch_class_metrics(pitch_classes)

            # Extract and analyze chords
            chords = self._extract_chords(midi_toolkit, midi_pretty)
            chord_metrics = self._calculate_chord_metrics(chords)

            # Calculate transition metrics
            transition_metrics = self._calculate_transition_metrics(pitch_classes)

            # Combine all metrics
            metrics = {**pc_metrics, **chord_metrics, **transition_metrics}

            # If seed is provided, calculate consistency metrics
            if seed_path:
                seed_consistency = self._calculate_seed_consistency(
                    midi_path, seed_path
                )
                metrics['seed_continuation_consistency'] = seed_consistency

            return metrics

        except Exception as e:
            print(f"Error computing pitch-class metrics for {midi_path}: {e}")
            return self._get_default_metrics(seed_path is not None)

    def _extract_pitch_classes(
        self,
        music: muspy.Music,
        midi_toolkit: miditoolkit.MidiFile
    ) -> List[int]:
        """
        Extract pitch classes from MIDI file.

        Args:
            music: MusPy Music object
            midi_toolkit: MidiToolkit MidiFile object

        Returns:
            List of pitch classes (0-11)
        """
        pitch_classes = []

        # Extract from MusPy tracks
        for track in music.tracks:
            for note in track.notes:
                if note.duration >= self.min_duration * music.resolution:
                    pitch_classes.append(note.pitch % 12)

        # Fallback to miditoolkit if MusPy didn't find notes
        if not pitch_classes:
            for instrument in midi_toolkit.instruments:
                for note in instrument.notes:
                    duration = note.end - note.start
                    if duration >= self.min_duration * midi_toolkit.ticks_per_beat:
                        pitch_classes.append(note.pitch % 12)

        return pitch_classes

    def _calculate_pitch_class_metrics(
        self,
        pitch_classes: List[int]
    ) -> Dict[str, float]:
        """
        Calculate pitch-class distribution metrics.

        Args:
            pitch_classes: List of pitch classes

        Returns:
            Dictionary of pitch-class metrics
        """
        if not pitch_classes:
            return {
                'pitch_class_entropy': 0.0,
                'pitch_class_histogram_uniformity': 0.0,
                'dominant_pitch_classes': 0
            }

        # Calculate histogram
        pc_counts = Counter(pitch_classes)
        total_notes = len(pitch_classes)
        pc_probs = np.array([pc_counts.get(i, 0) / total_notes for i in range(12)])

        # Calculate entropy
        pc_probs_nonzero = pc_probs[pc_probs > 0]
        entropy = -np.sum(pc_probs_nonzero * np.log2(pc_probs_nonzero))

        # Calculate uniformity (1 - variance of distribution)
        uniform_dist = np.ones(12) / 12
        uniformity = 1.0 - np.sum((pc_probs - uniform_dist) ** 2)

        # Count dominant pitch classes (>10% occurrence)
        dominant_pcs = np.sum(pc_probs > 0.1)

        return {
            'pitch_class_entropy': float(entropy),
            'pitch_class_histogram_uniformity': float(uniformity),
            'dominant_pitch_classes': int(dominant_pcs)
        }

    def _extract_chords(
        self,
        midi_toolkit: miditoolkit.MidiFile,
        midi_pretty: pretty_midi.PrettyMIDI
    ) -> List[Tuple[float, List[int]]]:
        """
        Extract chord sequences from MIDI file.

        Args:
            midi_toolkit: MidiToolkit MidiFile object
            midi_pretty: PrettyMIDI object

        Returns:
            List of (timestamp, chord_pitches) tuples
        """
        # Collect all notes with timestamps
        all_notes = []
        for instrument in midi_toolkit.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    start_time = note.start / midi_toolkit.ticks_per_beat
                    end_time = note.end / midi_toolkit.ticks_per_beat
                    all_notes.append((start_time, end_time, note.pitch))

        if not all_notes:
            return []

        # Sort by start time
        all_notes.sort(key=lambda x: x[0])

        # Group notes into chords (notes starting within a small time window)
        chords = []
        time_window = 0.1  # 100ms window for chord detection

        i = 0
        while i < len(all_notes):
            current_time = all_notes[i][0]
            chord_pitches = []

            # Collect all notes starting around the same time
            while i < len(all_notes) and all_notes[i][0] - current_time <= time_window:
                chord_pitches.append(all_notes[i][2])
                i += 1

            if len(chord_pitches) >= 2:  # Only consider as chord if 2+ notes
                chords.append((current_time, chord_pitches))

        return chords

    def _calculate_chord_metrics(
        self,
        chords: List[Tuple[float, List[int]]]
    ) -> Dict[str, float]:
        """
        Calculate chord-based metrics.

        Args:
            chords: List of (timestamp, chord_pitches) tuples

        Returns:
            Dictionary of chord metrics
        """
        if not chords:
            return {
                'chord_progression_consistency': 0.0,
                'chord_diversity': 0,
                'harmonic_rhythm_regularity': 0.0
            }

        # Extract chord types (as pitch class sets)
        chord_types = []
        chord_times = []

        for timestamp, pitches in chords:
            # Normalize to pitch classes and sort
            pc_set = tuple(sorted(set(p % 12 for p in pitches)))
            chord_types.append(pc_set)
            chord_times.append(timestamp)

        # Calculate diversity (number of unique chord types)
        unique_chords = len(set(chord_types))

        # Calculate progression consistency (inverse of normalized edit distance)
        if len(chord_types) > 1:
            # Calculate transitions between different chords
            transitions = []
            for i in range(len(chord_types) - 1):
                if chord_types[i] != chord_types[i + 1]:
                    transitions.append((chord_types[i], chord_types[i + 1]))

            if transitions:
                # Consistency = how often we see repeated transitions
                transition_counts = Counter(transitions)
                max_count = max(transition_counts.values())
                consistency = max_count / len(transitions)
            else:
                consistency = 1.0  # All same chord = perfectly consistent
        else:
            consistency = 1.0

        # Calculate harmonic rhythm regularity
        if len(chord_times) > 2:
            time_diffs = np.diff(chord_times)
            # Regularity = 1 - coefficient of variation of time differences
            if np.mean(time_diffs) > 0:
                cv = np.std(time_diffs) / np.mean(time_diffs)
                regularity = 1.0 / (1.0 + cv)  # Normalize to [0, 1]
            else:
                regularity = 0.0
        else:
            regularity = 0.0

        return {
            'chord_progression_consistency': float(consistency),
            'chord_diversity': int(unique_chords),
            'harmonic_rhythm_regularity': float(regularity)
        }

    def _calculate_transition_metrics(
        self,
        pitch_classes: List[int]
    ) -> Dict[str, float]:
        """
        Calculate pitch-class transition metrics.

        Args:
            pitch_classes: List of pitch classes

        Returns:
            Dictionary of transition metrics
        """
        if len(pitch_classes) < 2:
            return {'pitch_class_transition_entropy': 0.0}

        # Build transition matrix
        transition_matrix = np.zeros((12, 12))

        for i in range(len(pitch_classes) - 1):
            from_pc = pitch_classes[i]
            to_pc = pitch_classes[i + 1]
            transition_matrix[from_pc, to_pc] += 1

        # Calculate transition probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transition_matrix / row_sums

        # Calculate average entropy across all source pitch classes
        entropies = []
        for i in range(12):
            probs = transition_probs[i]
            probs_nonzero = probs[probs > 0]
            if len(probs_nonzero) > 0:
                entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
                entropies.append(entropy)

        avg_entropy = np.mean(entropies) if entropies else 0.0

        return {'pitch_class_transition_entropy': float(avg_entropy)}

    def _calculate_seed_consistency(
        self,
        continuation_path: Union[str, Path],
        seed_path: Union[str, Path]
    ) -> float:
        """
        Calculate consistency between seed and continuation.

        Args:
            continuation_path: Path to continuation MIDI
            seed_path: Path to seed MIDI

        Returns:
            Consistency score between 0 and 1
        """
        try:
            # Load both files
            seed_music = muspy.read_midi(str(seed_path))
            cont_music = muspy.read_midi(str(continuation_path))

            # Extract pitch classes from seed
            seed_midi = miditoolkit.MidiFile(str(seed_path))
            seed_pcs = self._extract_pitch_classes(seed_music, seed_midi)

            # Extract pitch classes from continuation
            cont_midi = miditoolkit.MidiFile(str(continuation_path))
            cont_pcs = self._extract_pitch_classes(cont_music, cont_midi)

            if not seed_pcs or not cont_pcs:
                return 0.0

            # Calculate pitch class distributions
            seed_dist = np.array([seed_pcs.count(i) / len(seed_pcs) for i in range(12)])
            cont_dist = np.array([cont_pcs.count(i) / len(cont_pcs) for i in range(12)])

            # Calculate cosine similarity between distributions
            dot_product = np.dot(seed_dist, cont_dist)
            norm_seed = np.linalg.norm(seed_dist)
            norm_cont = np.linalg.norm(cont_dist)

            if norm_seed > 0 and norm_cont > 0:
                consistency = dot_product / (norm_seed * norm_cont)
            else:
                consistency = 0.0

            return float(consistency)

        except Exception as e:
            print(f"Error calculating seed consistency: {e}")
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
            'pitch_class_entropy': 0.0,
            'pitch_class_histogram_uniformity': 0.0,
            'dominant_pitch_classes': 0,
            'chord_progression_consistency': 0.0,
            'chord_diversity': 0,
            'harmonic_rhythm_regularity': 0.0,
            'pitch_class_transition_entropy': 0.0
        }

        if has_seed:
            metrics['seed_continuation_consistency'] = 0.0

        return metrics
