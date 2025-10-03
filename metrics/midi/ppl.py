"""
Perplexity calculation under referee language model for MIDI files.

This module provides metrics for evaluating MIDI sequences using
perplexity scores from pretrained music language models.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import muspy
import miditoolkit
import pretty_midi


class PerplexityMetricCalculator:
    """
    Calculate perplexity metrics from MIDI files using language models.

    This class evaluates MIDI sequences by computing perplexity scores
    under various referee language models, including n-gram models and
    neural sequence models.
    """

    def __init__(
        self,
        model_type: str = "ngram",
        n_gram_size: int = 4,
        vocab_size: int = 128,
        context_length: int = 512,
        device: str = None
    ):
        """
        Initialize the perplexity metric calculator.

        Args:
            model_type: Type of language model ("ngram", "lstm", "transformer")
            n_gram_size: Size of n-grams for n-gram model
            vocab_size: Size of vocabulary (typically 128 for MIDI pitches)
            context_length: Maximum context length for neural models
            device: Device for neural models ("cuda" or "cpu")
        """
        self.model_type = model_type
        self.n_gram_size = n_gram_size
        self.vocab_size = vocab_size
        self.context_length = context_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model components
        self.ngram_model = None
        self.neural_model = None

    def compute(
        self,
        midi_path: Union[str, Path],
        seed_path: Optional[Union[str, Path]] = None,
        reference_corpus: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute perplexity metrics.

        Args:
            midi_path: Path to the MIDI file to analyze
            seed_path: Optional path to seed MIDI for conditional perplexity
            reference_corpus: Optional list of reference MIDI paths for model training

        Returns:
            Dictionary containing:
                - perplexity: Overall perplexity score
                - log_likelihood: Average log-likelihood per token
                - pitch_perplexity: Perplexity of pitch sequence
                - rhythm_perplexity: Perplexity of rhythm sequence
                - conditional_perplexity: Perplexity conditioned on seed (if provided)
                - cross_entropy: Cross-entropy loss
                - bits_per_token: Bits per token (normalized perplexity)
        """
        try:
            # Load MIDI file
            midi_toolkit = miditoolkit.MidiFile(str(midi_path))
            midi_pretty = pretty_midi.PrettyMIDI(str(midi_path))

            # Convert MIDI to token sequence
            tokens = self._midi_to_tokens(midi_toolkit, midi_pretty)

            if len(tokens) < 2:
                return self._get_default_metrics(seed_path is not None)

            # Build or use reference model
            if reference_corpus and not self.ngram_model:
                self._train_ngram_model(reference_corpus)
            elif not self.ngram_model:
                # Build model from current file (self-perplexity)
                self._train_ngram_model([str(midi_path)])

            # Calculate perplexity metrics
            perplexity_metrics = self._calculate_perplexity(tokens)

            # Calculate separate pitch and rhythm perplexity
            pitch_tokens = self._extract_pitch_tokens(midi_toolkit)
            rhythm_tokens = self._extract_rhythm_tokens(midi_toolkit)

            pitch_ppl = self._calculate_sequence_perplexity(pitch_tokens)
            rhythm_ppl = self._calculate_sequence_perplexity(rhythm_tokens)

            metrics = {
                **perplexity_metrics,
                'pitch_perplexity': pitch_ppl,
                'rhythm_perplexity': rhythm_ppl
            }

            # If seed is provided, calculate conditional perplexity
            if seed_path:
                conditional_ppl = self._calculate_conditional_perplexity(
                    midi_path, seed_path
                )
                metrics['conditional_perplexity'] = conditional_ppl

            return metrics

        except Exception as e:
            print(f"Error computing perplexity metrics for {midi_path}: {e}")
            return self._get_default_metrics(seed_path is not None)

    def _midi_to_tokens(
        self,
        midi_toolkit: miditoolkit.MidiFile,
        midi_pretty: pretty_midi.PrettyMIDI
    ) -> List[int]:
        """
        Convert MIDI file to token sequence.

        Args:
            midi_toolkit: MidiToolkit MidiFile object
            midi_pretty: PrettyMIDI object

        Returns:
            List of integer tokens representing the MIDI sequence
        """
        tokens = []

        # Collect all note events
        events = []
        for instrument in midi_toolkit.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    # Store as (time, event_type, value)
                    events.append((note.start, 'note_on', note.pitch))
                    events.append((note.end, 'note_off', note.pitch))

        # Sort by time
        events.sort(key=lambda x: x[0])

        # Convert to tokens
        # Token format: pitch (0-127) for note events
        for time, event_type, pitch in events:
            if event_type == 'note_on':
                tokens.append(pitch)

        return tokens

    def _extract_pitch_tokens(
        self,
        midi_toolkit: miditoolkit.MidiFile
    ) -> List[int]:
        """
        Extract pitch sequence as tokens.

        Args:
            midi_toolkit: MidiToolkit MidiFile object

        Returns:
            List of pitch tokens
        """
        pitches = []

        for instrument in midi_toolkit.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    pitches.append((note.start, note.pitch))

        # Sort by time
        pitches.sort(key=lambda x: x[0])

        return [p for _, p in pitches]

    def _extract_rhythm_tokens(
        self,
        midi_toolkit: miditoolkit.MidiFile
    ) -> List[int]:
        """
        Extract rhythm sequence as tokens (quantized IOIs).

        Args:
            midi_toolkit: MidiToolkit MidiFile object

        Returns:
            List of rhythm tokens
        """
        onsets = []

        for instrument in midi_toolkit.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    onsets.append(note.start)

        onsets = sorted(set(onsets))

        if len(onsets) < 2:
            return []

        # Calculate IOIs and quantize to vocabulary
        iois = []
        for i in range(len(onsets) - 1):
            ioi = onsets[i + 1] - onsets[i]
            # Quantize to ticks (map to 0-127 range)
            quantized_ioi = min(int(ioi / midi_toolkit.ticks_per_beat * 16), 127)
            iois.append(quantized_ioi)

        return iois

    def _train_ngram_model(
        self,
        corpus_paths: List[str]
    ) -> None:
        """
        Train n-gram language model from corpus.

        Args:
            corpus_paths: List of MIDI file paths for training
        """
        # Count n-grams
        ngram_counts = defaultdict(lambda: defaultdict(int))
        context_counts = defaultdict(int)

        for path in corpus_paths:
            try:
                midi = miditoolkit.MidiFile(path)
                tokens = self._extract_pitch_tokens(midi)

                # Count n-grams
                for i in range(len(tokens) - self.n_gram_size):
                    context = tuple(tokens[i:i + self.n_gram_size - 1])
                    next_token = tokens[i + self.n_gram_size - 1]

                    ngram_counts[context][next_token] += 1
                    context_counts[context] += 1

            except Exception as e:
                print(f"Error processing {path} for n-gram model: {e}")
                continue

        # Convert to probabilities with Laplace smoothing
        self.ngram_model = {}

        for context, next_token_counts in ngram_counts.items():
            total_count = context_counts[context]
            probs = {}

            for token, count in next_token_counts.items():
                # Laplace smoothing
                probs[token] = (count + 1) / (total_count + self.vocab_size)

            self.ngram_model[context] = probs

        # Store default probability for unseen contexts
        self.default_prob = 1.0 / self.vocab_size

    def _calculate_perplexity(
        self,
        tokens: List[int]
    ) -> Dict[str, float]:
        """
        Calculate perplexity metrics for token sequence.

        Args:
            tokens: List of integer tokens

        Returns:
            Dictionary of perplexity metrics
        """
        if len(tokens) < self.n_gram_size:
            return {
                'perplexity': float('inf'),
                'log_likelihood': float('-inf'),
                'cross_entropy': float('inf'),
                'bits_per_token': float('inf')
            }

        log_probs = []

        for i in range(self.n_gram_size - 1, len(tokens)):
            context = tuple(tokens[i - self.n_gram_size + 1:i])
            next_token = tokens[i]

            # Get probability from n-gram model
            if self.ngram_model and context in self.ngram_model:
                prob = self.ngram_model[context].get(next_token, self.default_prob)
            else:
                prob = self.default_prob

            log_probs.append(np.log2(prob))

        # Calculate metrics
        avg_log_prob = np.mean(log_probs)
        cross_entropy = -avg_log_prob
        perplexity = 2 ** cross_entropy

        return {
            'perplexity': float(perplexity),
            'log_likelihood': float(avg_log_prob * np.log(2)),  # Convert to natural log
            'cross_entropy': float(cross_entropy),
            'bits_per_token': float(cross_entropy)
        }

    def _calculate_sequence_perplexity(
        self,
        tokens: List[int]
    ) -> float:
        """
        Calculate perplexity for a token sequence.

        Args:
            tokens: List of integer tokens

        Returns:
            Perplexity score
        """
        if len(tokens) < 2:
            return float('inf')

        # Use simple unigram model for individual sequences
        token_counts = defaultdict(int)
        for token in tokens:
            token_counts[token] += 1

        total = len(tokens)
        log_probs = []

        for token in tokens:
            prob = token_counts[token] / total
            log_probs.append(np.log2(prob))

        cross_entropy = -np.mean(log_probs)
        perplexity = 2 ** cross_entropy

        return float(perplexity)

    def _calculate_conditional_perplexity(
        self,
        continuation_path: Union[str, Path],
        seed_path: Union[str, Path]
    ) -> float:
        """
        Calculate conditional perplexity of continuation given seed.

        Args:
            continuation_path: Path to continuation MIDI
            seed_path: Path to seed MIDI

        Returns:
            Conditional perplexity score
        """
        try:
            # Load seed and continuation
            seed_midi = miditoolkit.MidiFile(str(seed_path))
            seed_tokens = self._extract_pitch_tokens(seed_midi)

            cont_midi = miditoolkit.MidiFile(str(continuation_path))
            cont_tokens = self._extract_pitch_tokens(cont_midi)

            if not seed_tokens or not cont_tokens:
                return float('inf')

            # Build model from seed
            temp_model = defaultdict(lambda: defaultdict(int))
            context_counts = defaultdict(int)

            for i in range(len(seed_tokens) - self.n_gram_size + 1):
                context = tuple(seed_tokens[i:i + self.n_gram_size - 1])
                next_token = seed_tokens[i + self.n_gram_size - 1]

                temp_model[context][next_token] += 1
                context_counts[context] += 1

            # Calculate perplexity of continuation
            log_probs = []

            for i in range(self.n_gram_size - 1, len(cont_tokens)):
                context = tuple(cont_tokens[i - self.n_gram_size + 1:i])
                next_token = cont_tokens[i]

                if context in temp_model:
                    total = context_counts[context]
                    count = temp_model[context].get(next_token, 0)
                    prob = (count + 1) / (total + self.vocab_size)
                else:
                    prob = 1.0 / self.vocab_size

                log_probs.append(np.log2(prob))

            if log_probs:
                cross_entropy = -np.mean(log_probs)
                perplexity = 2 ** cross_entropy
            else:
                perplexity = float('inf')

            return float(perplexity)

        except Exception as e:
            print(f"Error calculating conditional perplexity: {e}")
            return float('inf')

    def _get_default_metrics(self, has_seed: bool = False) -> Dict[str, float]:
        """
        Get default metrics when computation fails.

        Args:
            has_seed: Whether seed comparison metrics should be included

        Returns:
            Dictionary of default metric values
        """
        metrics = {
            'perplexity': float('inf'),
            'log_likelihood': float('-inf'),
            'pitch_perplexity': float('inf'),
            'rhythm_perplexity': float('inf'),
            'cross_entropy': float('inf'),
            'bits_per_token': float('inf')
        }

        if has_seed:
            metrics['conditional_perplexity'] = float('inf')

        return metrics


class NeuralLanguageModel(torch.nn.Module):
    """
    Simple neural language model for MIDI sequences.

    This is a placeholder for more sophisticated models like LSTM or Transformer.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2
    ):
        """
        Initialize neural language model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
        """
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of token indices [batch, seq_len]
            hidden: Optional hidden state tuple

        Returns:
            Tuple of (output logits, hidden state)
        """
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)

        return logits, hidden

    def calculate_perplexity(
        self,
        tokens: List[int],
        device: torch.device
    ) -> float:
        """
        Calculate perplexity for a token sequence.

        Args:
            tokens: List of integer tokens
            device: Device for computation

        Returns:
            Perplexity score
        """
        self.eval()

        with torch.no_grad():
            # Convert to tensor
            x = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
            y = torch.tensor(tokens[1:], dtype=torch.long).to(device)

            # Forward pass
            logits, _ = self.forward(x)
            logits = logits.squeeze(0)

            # Calculate cross-entropy loss
            loss = F.cross_entropy(logits, y)
            perplexity = torch.exp(loss).item()

        return float(perplexity)
