"""
REMI (Revamped MIDI-derived events) Transformer wrapper for pop music generation.
Based on the Pop Music Transformer architecture using REMI representation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import pretty_midi
from typing import List, Tuple, Optional, Dict


class REMIEvent:
    """REMI event representation for symbolic music."""

    def __init__(self, name: str, value: Optional[int] = None):
        """
        Initialize REMI event.

        Args:
            name: Event type (Bar, Position, Tempo, Chord, Pitch, Duration, Velocity)
            value: Event value (specific to event type)
        """
        self.name = name
        self.value = value

    def __repr__(self):
        if self.value is not None:
            return f"{self.name}_{self.value}"
        return self.name

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

    def __hash__(self):
        return hash((self.name, self.value))


class REMIVocabulary:
    """REMI vocabulary for tokenization."""

    def __init__(self):
        """Initialize REMI vocabulary."""
        self.event_to_idx = {}
        self.idx_to_event = {}
        self._build_vocabulary()

    def _build_vocabulary(self):
        """Build REMI vocabulary."""
        idx = 0

        # Special tokens
        for token in ["<PAD>", "<SOS>", "<EOS>"]:
            self.event_to_idx[token] = idx
            self.idx_to_event[idx] = token
            idx += 1

        # Bar events
        self.event_to_idx["Bar"] = idx
        self.idx_to_event[idx] = "Bar"
        idx += 1

        # Position events (16th note resolution, 16 positions per bar in 4/4)
        for pos in range(16):
            event = f"Position_{pos}"
            self.event_to_idx[event] = idx
            self.idx_to_event[idx] = event
            idx += 1

        # Tempo events (30-210 BPM, step of 4)
        for tempo in range(30, 214, 4):
            event = f"Tempo_{tempo}"
            self.event_to_idx[event] = idx
            self.idx_to_event[idx] = event
            idx += 1

        # Chord events (major, minor, etc.)
        for root in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
            for quality in ["maj", "min", "dim", "aug", "7"]:
                event = f"Chord_{root}:{quality}"
                self.event_to_idx[event] = idx
                self.idx_to_event[idx] = event
                idx += 1

        # Pitch events (MIDI note numbers 21-108, piano range)
        for pitch in range(21, 109):
            event = f"Pitch_{pitch}"
            self.event_to_idx[event] = idx
            self.idx_to_event[idx] = event
            idx += 1

        # Duration events (in 16th note units, 1-64)
        for duration in range(1, 65):
            event = f"Duration_{duration}"
            self.event_to_idx[event] = idx
            self.idx_to_event[idx] = event
            idx += 1

        # Velocity events (MIDI velocity, 1-127, step of 4)
        for velocity in range(1, 128, 4):
            event = f"Velocity_{velocity}"
            self.event_to_idx[event] = idx
            self.idx_to_event[idx] = event
            idx += 1

        self.vocab_size = idx

    def encode(self, event: str) -> int:
        """Encode event to index."""
        return self.event_to_idx.get(event, self.event_to_idx["<PAD>"])

    def decode(self, idx: int) -> str:
        """Decode index to event."""
        return self.idx_to_event.get(idx, "<PAD>")


class REMITransformer(nn.Module):
    """REMI Transformer model for music generation."""

    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Initialize REMI Transformer.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(8192, d_model)  # Max sequence length

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """Forward pass."""
        seq_len = src.size(1)
        positions = torch.arange(seq_len, device=src.device).unsqueeze(0)

        embedded = self.embedding(src) * np.sqrt(self.d_model)
        embedded = embedded + self.pos_encoder(positions)

        output = self.transformer(embedded, src_mask)
        logits = self.fc_out(output)

        return logits


class REMITransformerWrapper:
    """Wrapper for REMI Transformer model."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize REMI Transformer wrapper.

        Args:
            model_path: Path to pretrained checkpoint
            device: Device to run model on (cuda/cpu)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading REMI Transformer model on {self.device}...")

        # Initialize vocabulary
        self.vocab = REMIVocabulary()

        # Initialize model
        self.model = REMITransformer(
            vocab_size=self.vocab.vocab_size,
            d_model=512,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )

        if model_path and Path(model_path).exists():
            self._load_checkpoint(model_path)
        else:
            print("No checkpoint provided. Using randomly initialized model.")

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully")

    def _load_checkpoint(self, model_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {model_path}")

    def generate(self, primer_midi: Optional[str] = None, num_bars: int = 16,
                 temperature: float = 1.0, top_k: int = 5, top_p: float = 0.9,
                 seed: Optional[int] = None) -> pretty_midi.PrettyMIDI:
        """
        Generate music using REMI Transformer.

        Args:
            primer_midi: Path to primer MIDI file (optional)
            num_bars: Number of bars to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            seed: Random seed for reproducibility

        Returns:
            Generated MIDI as PrettyMIDI object
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        # Initialize with SOS token
        generated_tokens = [self.vocab.encode("<SOS>")]

        # Add primer if provided
        if primer_midi:
            primer_tokens = self._midi_to_remi(primer_midi)
            generated_tokens.extend(primer_tokens)

        # Generate tokens
        max_length = num_bars * 64  # Approximate tokens per bar
        current_bars = 0

        with torch.no_grad():
            while len(generated_tokens) < max_length and current_bars < num_bars:
                # Prepare input
                input_seq = torch.tensor([generated_tokens], device=self.device)

                # Forward pass
                logits = self.model(input_seq)
                next_token_logits = logits[0, -1, :]

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                # Check for Bar event
                if self.vocab.decode(next_token) == "Bar":
                    current_bars += 1

                generated_tokens.append(next_token)

                # Check for EOS
                if next_token == self.vocab.encode("<EOS>"):
                    break

        # Convert tokens to MIDI
        generated_midi = self._remi_to_midi(generated_tokens)

        return generated_midi

    def _midi_to_remi(self, midi_path: str) -> List[int]:
        """Convert MIDI file to REMI token sequence."""
        # Placeholder implementation
        # In production, implement full MIDI to REMI conversion
        return []

    def _remi_to_midi(self, tokens: List[int]) -> pretty_midi.PrettyMIDI:
        """
        Convert REMI token sequence to MIDI.

        Args:
            tokens: List of token indices

        Returns:
            PrettyMIDI object
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        current_time = 0.0
        current_tempo = 120
        current_velocity = 80
        sixteenth_note_duration = 60.0 / current_tempo / 4  # Duration of 16th note

        pending_note = None  # (pitch, start_time)

        for token_idx in tokens:
            event = self.vocab.decode(token_idx)

            if event.startswith("Bar"):
                current_time = ((current_time // (4.0 * sixteenth_note_duration)) + 1) * (4.0 * sixteenth_note_duration)

            elif event.startswith("Position_"):
                position = int(event.split("_")[1])
                bar_start = (current_time // (4.0 * sixteenth_note_duration)) * (4.0 * sixteenth_note_duration)
                current_time = bar_start + position * sixteenth_note_duration

            elif event.startswith("Tempo_"):
                current_tempo = int(event.split("_")[1])
                sixteenth_note_duration = 60.0 / current_tempo / 4

            elif event.startswith("Velocity_"):
                current_velocity = int(event.split("_")[1])

            elif event.startswith("Pitch_"):
                pitch = int(event.split("_")[1])
                pending_note = (pitch, current_time)

            elif event.startswith("Duration_"):
                if pending_note:
                    pitch, start_time = pending_note
                    duration_units = int(event.split("_")[1])
                    duration = duration_units * sixteenth_note_duration
                    end_time = start_time + duration

                    note = pretty_midi.Note(
                        velocity=current_velocity,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                    instrument.notes.append(note)
                    pending_note = None

        midi.instruments.append(instrument)
        return midi

    def generate_and_save(self, output_path: str, primer_midi: Optional[str] = None,
                         num_bars: int = 16, **kwargs) -> Dict:
        """
        Generate music and save to MIDI file.

        Args:
            output_path: Output MIDI file path
            primer_midi: Path to primer MIDI (optional)
            num_bars: Number of bars to generate
            **kwargs: Additional generation parameters

        Returns:
            Metadata dictionary
        """
        generated_midi = self.generate(
            primer_midi=primer_midi,
            num_bars=num_bars,
            **kwargs
        )

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save MIDI
        generated_midi.write(str(output_path))

        metadata = {
            "model": "remi-transformer",
            "primer": str(primer_midi) if primer_midi else None,
            "num_bars": num_bars,
            "output_path": str(output_path),
            **kwargs
        }

        return metadata


def main():
    """Test the wrapper."""
    wrapper = REMITransformerWrapper()

    print("Generating pop music with REMI Transformer...")

    output_dir = Path(__file__).parent.parent.parent / "runs" / "artifacts" / "midi"
    output_path = output_dir / "test_remi_transformer.mid"

    metadata = wrapper.generate_and_save(
        output_path=output_path,
        num_bars=16,
        temperature=1.0,
        seed=42
    )

    print(f"Generated MIDI saved to: {metadata['output_path']}")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()
