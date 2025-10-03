"""
Motif development metrics using n-gram analysis for MIDI files.

This module provides metrics for analyzing motivic development,
repetition, variation, and thematic coherence in symbolic music.
"""

from typing import Dict, List, Optional, Tuple, Union, Set
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
import muspy
import miditoolkit
import pretty_midi
from itertools import combinations


class MotifMetricCalculator:
    """
    Calculate motif development metrics from MIDI files using n-gram analysis.

    This class analyzes motivic content through pitch and rhythm n-grams,
    repetition patterns, variation techniques, and thematic development.
    """

    def __init__(
        self,
        n_gram_sizes: List[int] = None,
        min_occurrences: int = 2,
        similarity_threshold: float = 0.7,
        time_resolution: float = 0.25
    ):
        """
        Initialize the motif metric calculator.

        Args:
            n_gram_sizes: List of n-gram sizes to analyze [3, 4, 5]
            min_occurrences: Minimum occurrences to consider a motif
            similarity_threshold: Threshold for motif variation detection
            time_resolution: Time resolution in beats for rhythm quantization
        """
        self.n_gram_sizes = n_gram_sizes or [3, 4, 5]
        self.min_occurrences = min_occurrences
        self.similarity_threshold = similarity_threshold
        self.time_resolution = time_resolution

    def compute(
        self,
        midi_path: Union[str, Path],
        seed_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Compute motif development metrics.

        Args:
            midi_path: Path to the MIDI file to analyze
            seed_path: Optional path to seed MIDI for comparison

        Returns:
            Dictionary containing:
                - motif_repetition_rate: Rate of motif repetitions
                - motif_variation_rate: Rate of motif variations
                - unique_motifs_count: Number of unique motifs
                - thematic_coherence: Overall thematic coherence
                - motif_development_score: Score for motivic development
                - pitch_ngram_diversity: Diversity of pitch n-grams
                - rhythm_ngram_diversity: Diversity of rhythm n-grams
                - self_similarity_index: Self-similarity of the piece
                - seed_motif_continuation: Motif continuation from seed (if provided)
        """
        try:
            # Load MIDI file
            music = muspy.read_midi(str(midi_path))
            midi_toolkit = miditoolkit.MidiFile(str(midi_path))
            midi_pretty = pretty_midi.PrettyMIDI(str(midi_path))

            # Extract note sequences
            note_sequence = self._extract_note_sequence(music, midi_toolkit)

            # Extract pitch and rhythm n-grams
            pitch_ngrams = self._extract_pitch_ngrams(note_sequence)
            rhythm_ngrams = self._extract_rhythm_ngrams(note_sequence)

            # Calculate repetition metrics
            repetition_metrics = self._calculate_repetition_metrics(
                pitch_ngrams, rhythm_ngrams
            )

            # Calculate variation metrics
            variation_metrics = self._calculate_variation_metrics(
                pitch_ngrams, rhythm_ngrams
            )

            # Calculate development metrics
            development_metrics = self._calculate_development_metrics(
                note_sequence, pitch_ngrams
            )

            # Calculate self-similarity
            self_similarity = self._calculate_self_similarity(note_sequence)

            # Combine all metrics
            metrics = {
                **repetition_metrics,
                **variation_metrics,
                **development_metrics,
                'self_similarity_index': self_similarity
            }

            # If seed is provided, calculate continuation metrics
            if seed_path:
                seed_continuation = self._calculate_seed_continuation(
                    midi_path, seed_path
                )
                metrics['seed_motif_continuation'] = seed_continuation

            return metrics

        except Exception as e:
            print(f"Error computing motif metrics for {midi_path}: {e}")
            return self._get_default_metrics(seed_path is not None)

    def _extract_note_sequence(
        self,
        music: muspy.Music,
        midi_toolkit: miditoolkit.MidiFile
    ) -> List[Tuple[float, int, float]]:
        """
        Extract note sequence (time, pitch, duration).

        Args:
            music: MusPy Music object
            midi_toolkit: MidiToolkit MidiFile object

        Returns:
            List of (onset_time, pitch, duration) tuples
        """
        notes = []

        # Extract from MusPy
        for track in music.tracks:
            for note in track.notes:
                onset_time = note.time / music.resolution
                duration = note.duration / music.resolution
                notes.append((onset_time, note.pitch, duration))

        # Fallback to miditoolkit
        if not notes:
            for instrument in midi_toolkit.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        onset_time = note.start / midi_toolkit.ticks_per_beat
                        duration = (note.end - note.start) / midi_toolkit.ticks_per_beat
                        notes.append((onset_time, note.pitch, duration))

        # Sort by onset time
        notes.sort(key=lambda x: x[0])

        return notes

    def _extract_pitch_ngrams(
        self,
        note_sequence: List[Tuple[float, int, float]]
    ) -> Dict[int, List[Tuple[int, ...]]]:
        """
        Extract pitch n-grams from note sequence.

        Args:
            note_sequence: List of (onset_time, pitch, duration) tuples

        Returns:
            Dictionary mapping n-gram size to list of pitch n-grams
        """
        if not note_sequence:
            return {n: [] for n in self.n_gram_sizes}

        pitches = [pitch for _, pitch, _ in note_sequence]
        ngrams = {}

        for n in self.n_gram_sizes:
            if len(pitches) >= n:
                # Extract contour (interval sequence)
                ngrams_list = []
                for i in range(len(pitches) - n + 1):
                    # Use interval representation (more robust to transposition)
                    intervals = tuple(
                        pitches[i + j + 1] - pitches[i + j]
                        for j in range(n - 1)
                    )
                    ngrams_list.append(intervals)
                ngrams[n] = ngrams_list
            else:
                ngrams[n] = []

        return ngrams

    def _extract_rhythm_ngrams(
        self,
        note_sequence: List[Tuple[float, int, float]]
    ) -> Dict[int, List[Tuple[float, ...]]]:
        """
        Extract rhythm n-grams from note sequence.

        Args:
            note_sequence: List of (onset_time, pitch, duration) tuples

        Returns:
            Dictionary mapping n-gram size to list of rhythm n-grams
        """
        if len(note_sequence) < 2:
            return {n: [] for n in self.n_gram_sizes}

        # Calculate inter-onset intervals (IOIs)
        onsets = [onset for onset, _, _ in note_sequence]
        iois = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]

        # Quantize IOIs
        quantized_iois = [
            round(ioi / self.time_resolution) * self.time_resolution
            for ioi in iois
        ]

        ngrams = {}

        for n in self.n_gram_sizes:
            if len(quantized_iois) >= n:
                ngrams_list = []
                for i in range(len(quantized_iois) - n + 1):
                    rhythm_pattern = tuple(quantized_iois[i:i + n])
                    ngrams_list.append(rhythm_pattern)
                ngrams[n] = ngrams_list
            else:
                ngrams[n] = []

        return ngrams

    def _calculate_repetition_metrics(
        self,
        pitch_ngrams: Dict[int, List[Tuple]],
        rhythm_ngrams: Dict[int, List[Tuple]]
    ) -> Dict[str, float]:
        """
        Calculate motif repetition metrics.

        Args:
            pitch_ngrams: Dictionary of pitch n-grams
            rhythm_ngrams: Dictionary of rhythm n-grams

        Returns:
            Dictionary of repetition metrics
        """
        # Combine pitch and rhythm n-grams across all sizes
        all_pitch_ngrams = []
        all_rhythm_ngrams = []

        for n in self.n_gram_sizes:
            all_pitch_ngrams.extend(pitch_ngrams.get(n, []))
            all_rhythm_ngrams.extend(rhythm_ngrams.get(n, []))

        # Count occurrences
        pitch_counts = Counter(all_pitch_ngrams)
        rhythm_counts = Counter(all_rhythm_ngrams)

        # Calculate repetition rates
        repeated_pitch = sum(1 for count in pitch_counts.values() if count >= self.min_occurrences)
        repeated_rhythm = sum(1 for count in rhythm_counts.values() if count >= self.min_occurrences)

        total_unique_pitch = len(pitch_counts)
        total_unique_rhythm = len(rhythm_counts)

        pitch_repetition = repeated_pitch / total_unique_pitch if total_unique_pitch > 0 else 0.0
        rhythm_repetition = repeated_rhythm / total_unique_rhythm if total_unique_rhythm > 0 else 0.0

        # Overall motif repetition rate
        motif_repetition_rate = (pitch_repetition + rhythm_repetition) / 2

        # Count unique motifs (high-frequency patterns)
        unique_motifs = len([c for c in pitch_counts.values() if c >= self.min_occurrences])

        # Calculate diversity metrics
        pitch_diversity = len(pitch_counts)
        rhythm_diversity = len(rhythm_counts)

        return {
            'motif_repetition_rate': float(motif_repetition_rate),
            'unique_motifs_count': int(unique_motifs),
            'pitch_ngram_diversity': int(pitch_diversity),
            'rhythm_ngram_diversity': int(rhythm_diversity)
        }

    def _calculate_variation_metrics(
        self,
        pitch_ngrams: Dict[int, List[Tuple]],
        rhythm_ngrams: Dict[int, List[Tuple]]
    ) -> Dict[str, float]:
        """
        Calculate motif variation metrics.

        Args:
            pitch_ngrams: Dictionary of pitch n-grams
            rhythm_ngrams: Dictionary of rhythm n-grams

        Returns:
            Dictionary of variation metrics
        """
        # Find similar but not identical n-grams (variations)
        variation_count = 0
        total_pairs = 0

        for n in self.n_gram_sizes:
            ngrams = pitch_ngrams.get(n, [])
            if len(ngrams) < 2:
                continue

            # Sample pairs for efficiency (max 1000 pairs)
            unique_ngrams = list(set(ngrams))
            if len(unique_ngrams) > 50:
                import random
                unique_ngrams = random.sample(unique_ngrams, 50)

            for ngram1, ngram2 in combinations(unique_ngrams, 2):
                total_pairs += 1

                # Calculate similarity (normalized edit distance)
                similarity = self._calculate_ngram_similarity(ngram1, ngram2)

                # Variation: similar but not identical
                if self.similarity_threshold <= similarity < 1.0:
                    variation_count += 1

        variation_rate = variation_count / total_pairs if total_pairs > 0 else 0.0

        return {'motif_variation_rate': float(variation_rate)}

    def _calculate_development_metrics(
        self,
        note_sequence: List[Tuple[float, int, float]],
        pitch_ngrams: Dict[int, List[Tuple]]
    ) -> Dict[str, float]:
        """
        Calculate motif development metrics.

        Args:
            note_sequence: List of (onset_time, pitch, duration) tuples
            pitch_ngrams: Dictionary of pitch n-grams

        Returns:
            Dictionary of development metrics
        """
        if len(note_sequence) < 10:
            return {
                'thematic_coherence': 0.0,
                'motif_development_score': 0.0
            }

        # Divide piece into sections
        num_sections = 4
        section_length = len(note_sequence) // num_sections

        if section_length < 3:
            return {
                'thematic_coherence': 0.0,
                'motif_development_score': 0.0
            }

        # Extract motifs from each section
        section_motifs = []

        for i in range(num_sections):
            start_idx = i * section_length
            end_idx = start_idx + section_length if i < num_sections - 1 else len(note_sequence)

            section_notes = note_sequence[start_idx:end_idx]
            pitches = [p for _, p, _ in section_notes]

            # Extract n-grams from this section
            if len(pitches) >= 3:
                section_ngrams = set()
                for n in self.n_gram_sizes[:1]:  # Use smallest n-gram size
                    if len(pitches) >= n:
                        for j in range(len(pitches) - n + 1):
                            intervals = tuple(
                                pitches[j + k + 1] - pitches[j + k]
                                for k in range(n - 1)
                            )
                            section_ngrams.add(intervals)
                section_motifs.append(section_ngrams)
            else:
                section_motifs.append(set())

        # Calculate thematic coherence (overlap between sections)
        coherence_scores = []
        for i in range(len(section_motifs) - 1):
            for j in range(i + 1, len(section_motifs)):
                if len(section_motifs[i]) > 0 or len(section_motifs[j]) > 0:
                    intersection = len(section_motifs[i] & section_motifs[j])
                    union = len(section_motifs[i] | section_motifs[j])
                    jaccard = intersection / union if union > 0 else 0.0
                    coherence_scores.append(jaccard)

        thematic_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0

        # Development score: balance between repetition and variation
        # High development = moderate repetition + high variation
        repetition = sum(1 for count in Counter(
            [ng for ngrams in pitch_ngrams.values() for ng in ngrams]
        ).values() if count >= self.min_occurrences)

        total_ngrams = sum(len(ngrams) for ngrams in pitch_ngrams.values())
        rep_ratio = repetition / total_ngrams if total_ngrams > 0 else 0.0

        # Ideal development has ~30-50% repetition with high coherence
        ideal_rep = 0.4
        development_score = (1 - abs(rep_ratio - ideal_rep)) * thematic_coherence

        return {
            'thematic_coherence': thematic_coherence,
            'motif_development_score': float(development_score)
        }

    def _calculate_self_similarity(
        self,
        note_sequence: List[Tuple[float, int, float]]
    ) -> float:
        """
        Calculate self-similarity matrix and average similarity.

        Args:
            note_sequence: List of (onset_time, pitch, duration) tuples

        Returns:
            Average self-similarity score
        """
        if len(note_sequence) < 10:
            return 0.0

        # Create a simple self-similarity matrix using pitch
        window_size = 8  # Number of notes per window
        step_size = 4

        windows = []
        for i in range(0, len(note_sequence) - window_size + 1, step_size):
            window = note_sequence[i:i + window_size]
            pitches = [p for _, p, _ in window]
            windows.append(pitches)

        if len(windows) < 2:
            return 0.0

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                # Use correlation as similarity measure
                if len(windows[i]) == len(windows[j]):
                    corr = np.corrcoef(windows[i], windows[j])[0, 1]
                    if not np.isnan(corr):
                        similarities.append(abs(corr))

        return float(np.mean(similarities)) if similarities else 0.0

    def _calculate_ngram_similarity(
        self,
        ngram1: Tuple,
        ngram2: Tuple
    ) -> float:
        """
        Calculate similarity between two n-grams.

        Args:
            ngram1: First n-gram
            ngram2: Second n-gram

        Returns:
            Similarity score between 0 and 1
        """
        if len(ngram1) != len(ngram2):
            return 0.0

        # Calculate normalized edit distance
        matches = sum(1 for a, b in zip(ngram1, ngram2) if a == b)
        similarity = matches / len(ngram1)

        return similarity

    def _calculate_seed_continuation(
        self,
        continuation_path: Union[str, Path],
        seed_path: Union[str, Path]
    ) -> float:
        """
        Calculate motif continuation from seed.

        Args:
            continuation_path: Path to continuation MIDI
            seed_path: Path to seed MIDI

        Returns:
            Continuation score between 0 and 1
        """
        try:
            # Load seed and continuation
            seed_music = muspy.read_midi(str(seed_path))
            seed_midi = miditoolkit.MidiFile(str(seed_path))
            seed_notes = self._extract_note_sequence(seed_music, seed_midi)

            cont_music = muspy.read_midi(str(continuation_path))
            cont_midi = miditoolkit.MidiFile(str(continuation_path))
            cont_notes = self._extract_note_sequence(cont_music, cont_midi)

            if not seed_notes or not cont_notes:
                return 0.0

            # Extract motifs from seed
            seed_pitch_ngrams = self._extract_pitch_ngrams(seed_notes)
            seed_motifs = set()
            for n in self.n_gram_sizes[:1]:  # Use smallest size
                seed_motifs.update(seed_pitch_ngrams.get(n, []))

            # Extract motifs from continuation
            cont_pitch_ngrams = self._extract_pitch_ngrams(cont_notes)
            cont_motifs = set()
            for n in self.n_gram_sizes[:1]:
                cont_motifs.update(cont_pitch_ngrams.get(n, []))

            # Calculate overlap
            if len(seed_motifs) == 0:
                return 0.0

            intersection = len(seed_motifs & cont_motifs)
            continuation_score = intersection / len(seed_motifs)

            return float(continuation_score)

        except Exception as e:
            print(f"Error calculating seed motif continuation: {e}")
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
            'motif_repetition_rate': 0.0,
            'motif_variation_rate': 0.0,
            'unique_motifs_count': 0,
            'thematic_coherence': 0.0,
            'motif_development_score': 0.0,
            'pitch_ngram_diversity': 0,
            'rhythm_ngram_diversity': 0,
            'self_similarity_index': 0.0
        }

        if has_seed:
            metrics['seed_motif_continuation'] = 0.0

        return metrics
