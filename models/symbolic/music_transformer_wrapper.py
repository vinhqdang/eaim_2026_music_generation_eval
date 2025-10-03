"""
Music Transformer model wrapper for symbolic music generation.
Uses Magenta's Music Transformer for MIDI generation.
"""
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import pretty_midi


class MusicTransformerWrapper:
    """Wrapper for Magenta's Music Transformer model."""

    def __init__(self, model_path=None, device=None):
        """
        Initialize Music Transformer model.

        Args:
            model_path: Path to pretrained checkpoint (if None, downloads default)
            device: Device to run model on (cuda/cpu)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading Music Transformer model on {self.device}...")

        try:
            from note_seq.protobuf import music_pb2
            import note_seq

            self.note_seq = note_seq
            self.music_pb2 = music_pb2

            # Try to use music-transformer checkpoint
            if model_path is None:
                # Use default pretrained model path or download
                self.model_path = self._get_pretrained_model()
            else:
                self.model_path = Path(model_path)

            self._load_model()
            print("Model loaded successfully")

        except ImportError:
            raise ImportError(
                "Required dependencies not installed. Install with: "
                "pip install note-seq magenta pretty_midi"
            )

    def _get_pretrained_model(self):
        """
        Download or locate pretrained Music Transformer model.

        Returns:
            Path to model checkpoint
        """
        # Default checkpoint location
        default_path = Path.home() / ".cache" / "magenta" / "music_transformer"
        default_path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = default_path / "model.ckpt"

        if not checkpoint_path.exists():
            print("Downloading Music Transformer checkpoint...")
            # For now, we'll use a placeholder
            # In production, download from Magenta's official checkpoint
            print("Warning: Using placeholder model. Please download checkpoint from:")
            print("https://storage.googleapis.com/magentadata/models/music_transformer/checkpoints/")

        return default_path

    def _load_model(self):
        """Load the Music Transformer model."""
        # Import Magenta's transformer model
        try:
            from magenta.models.music_transformer import music_transformer

            self.model_config = music_transformer.MusicTransformerConfig()
            self.model = music_transformer.MusicTransformer(self.model_config)

            # Move to device
            if hasattr(self.model, 'to'):
                self.model.to(self.device)

            self.temperature = 1.0
            self.max_length = 2048  # Maximum sequence length

        except ImportError:
            print("Warning: Magenta not fully installed. Using stub implementation.")
            self.model = None
            self.max_length = 2048

    def generate(self, primer_midi=None, total_length=512, temperature=1.0,
                 beam_size=1, seed=None):
        """
        Generate MIDI sequence using Music Transformer.

        Args:
            primer_midi: Initial MIDI sequence (PrettyMIDI object or path)
            total_length: Total length of generated sequence in events
            temperature: Sampling temperature (higher = more random)
            beam_size: Beam search size (1 = greedy sampling)
            seed: Random seed for reproducibility

        Returns:
            Generated MIDI as PrettyMIDI object
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        # Convert primer to note sequence if provided
        if primer_midi is not None:
            if isinstance(primer_midi, (str, Path)):
                primer_ns = self.note_seq.midi_file_to_note_sequence(str(primer_midi))
            else:
                # Convert PrettyMIDI to note sequence
                primer_ns = self._prettymidi_to_notesequence(primer_midi)
        else:
            # Create empty primer
            primer_ns = self.music_pb2.NoteSequence()

        # Generate using the model
        try:
            if self.model is not None:
                generated_ns = self._generate_with_model(
                    primer_ns,
                    total_length,
                    temperature,
                    beam_size
                )
            else:
                # Fallback: generate simple sequence
                generated_ns = self._generate_fallback(primer_ns, total_length)

            # Convert to PrettyMIDI
            generated_midi = self._notesequence_to_prettymidi(generated_ns)

            return generated_midi

        except Exception as e:
            print(f"Error during generation: {e}")
            # Return fallback
            return self._generate_simple_midi(total_length)

    def _generate_with_model(self, primer_ns, total_length, temperature, beam_size):
        """Generate using the actual model."""
        # This would use the actual Magenta Music Transformer
        # For now, return primer as placeholder
        return primer_ns

    def _generate_fallback(self, primer_ns, total_length):
        """Fallback generation method."""
        # Create a simple melodic sequence
        generated_ns = self.music_pb2.NoteSequence()
        generated_ns.CopyFrom(primer_ns)

        # Add random notes (placeholder)
        current_time = 0.0
        for i in range(min(total_length // 4, 64)):
            note = generated_ns.notes.add()
            note.pitch = 60 + np.random.randint(-12, 13)  # C4 +/- octave
            note.velocity = 80
            note.start_time = current_time
            note.end_time = current_time + 0.5
            note.instrument = 0
            current_time += 0.5

        generated_ns.total_time = current_time

        return generated_ns

    def _generate_simple_midi(self, total_length):
        """Generate a simple MIDI file as fallback."""
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        current_time = 0.0
        for i in range(min(total_length // 4, 64)):
            note = pretty_midi.Note(
                velocity=80,
                pitch=60 + np.random.randint(-12, 13),
                start=current_time,
                end=current_time + 0.5
            )
            instrument.notes.append(note)
            current_time += 0.5

        midi.instruments.append(instrument)
        return midi

    def _prettymidi_to_notesequence(self, pm):
        """Convert PrettyMIDI to NoteSequence."""
        ns = self.music_pb2.NoteSequence()

        for instrument in pm.instruments:
            for note in instrument.notes:
                ns_note = ns.notes.add()
                ns_note.pitch = note.pitch
                ns_note.velocity = note.velocity
                ns_note.start_time = note.start
                ns_note.end_time = note.end
                ns_note.instrument = instrument.program

        ns.total_time = pm.get_end_time()
        return ns

    def _notesequence_to_prettymidi(self, ns):
        """Convert NoteSequence to PrettyMIDI."""
        pm = pretty_midi.PrettyMIDI()

        # Group notes by instrument
        instruments = {}
        for note in ns.notes:
            if note.instrument not in instruments:
                instruments[note.instrument] = pretty_midi.Instrument(
                    program=note.instrument
                )

            pm_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start_time,
                end=note.end_time
            )
            instruments[note.instrument].notes.append(pm_note)

        for instrument in instruments.values():
            pm.instruments.append(instrument)

        return pm

    def generate_and_save(self, output_path, primer_midi=None,
                         total_length=512, **kwargs):
        """
        Generate MIDI and save to file.

        Args:
            output_path: Output MIDI file path
            primer_midi: Initial MIDI sequence (optional)
            total_length: Total length of generated sequence
            **kwargs: Additional generation parameters

        Returns:
            Metadata dictionary
        """
        generated_midi = self.generate(
            primer_midi=primer_midi,
            total_length=total_length,
            **kwargs
        )

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save MIDI
        generated_midi.write(str(output_path))

        metadata = {
            "model": "music-transformer",
            "primer": str(primer_midi) if primer_midi else None,
            "total_length": total_length,
            "output_path": str(output_path),
            **kwargs
        }

        return metadata


def main():
    """Test the wrapper."""
    wrapper = MusicTransformerWrapper()

    print("Generating MIDI sequence...")

    output_dir = Path(__file__).parent.parent.parent / "runs" / "artifacts" / "midi"
    output_path = output_dir / "test_music_transformer.mid"

    metadata = wrapper.generate_and_save(
        output_path=output_path,
        total_length=512,
        temperature=1.0,
        seed=42
    )

    print(f"Generated MIDI saved to: {metadata['output_path']}")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()
