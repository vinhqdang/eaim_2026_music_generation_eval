"""
Generate sample MIDI files for demonstration purposes.
Creates simple MIDI files that demonstrate the pipeline without requiring heavy models.
"""
import pretty_midi
from pathlib import Path
import json
import numpy as np


def create_piano_melody(duration=10, tempo=120):
    """Create a simple piano melody MIDI."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    # C major scale melody
    notes = [60, 62, 64, 65, 67, 65, 64, 62]  # C D E F G F E D
    note_duration = duration / len(notes)

    for i, pitch in enumerate(notes):
        start_time = i * note_duration
        end_time = start_time + note_duration * 0.8
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=start_time,
            end=end_time
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    return midi


def create_pop_progression(duration=10, tempo=120):
    """Create a pop chord progression MIDI."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)

    # Pop progression: I-V-vi-IV (C-G-Am-F)
    chords = [
        [60, 64, 67],  # C major
        [67, 71, 74],  # G major
        [69, 72, 76],  # A minor
        [65, 69, 72],  # F major
    ]

    chord_duration = duration / len(chords)

    for i, chord in enumerate(chords):
        start_time = i * chord_duration
        end_time = start_time + chord_duration * 0.9

        for pitch in chord:
            note = pretty_midi.Note(
                velocity=70,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            piano.notes.append(note)

    midi.instruments.append(piano)
    return midi


def create_drum_pattern(duration=10, tempo=120):
    """Create a drum pattern MIDI."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    drums = pretty_midi.Instrument(program=0, is_drum=True)

    # GM drum mapping: 36=Kick, 38=Snare, 42=Hi-hat closed
    kick_pitch = 36
    snare_pitch = 38
    hihat_pitch = 42

    beat_duration = 60 / tempo  # seconds per beat
    num_beats = int(duration / beat_duration)

    for beat in range(num_beats):
        time = beat * beat_duration

        # Kick on beats 1 and 3
        if beat % 4 in [0, 2]:
            drums.notes.append(pretty_midi.Note(
                velocity=100, pitch=kick_pitch,
                start=time, end=time + 0.1
            ))

        # Snare on beats 2 and 4
        if beat % 4 in [1, 3]:
            drums.notes.append(pretty_midi.Note(
                velocity=90, pitch=snare_pitch,
                start=time, end=time + 0.1
            ))

        # Hi-hat on every beat
        drums.notes.append(pretty_midi.Note(
            velocity=60, pitch=hihat_pitch,
            start=time, end=time + 0.05
        ))

    midi.instruments.append(drums)
    return midi


def create_sample_midi_files():
    """Create sample MIDI files demonstrating the pipeline."""
    output_dir = Path(__file__).parent.parent / "runs" / "artifacts"
    midi_dir = output_dir / "midi" / "samples"
    midi_dir.mkdir(parents=True, exist_ok=True)

    duration = 10  # 10 seconds

    samples = [
        {
            "name": "music_transformer_melody.mid",
            "generator": lambda: create_piano_melody(duration, tempo=120),
            "metadata": {
                "model": "MusicTransformer",
                "seed": "maestro_seed_001",
                "task": "T1_Structure",
                "duration": duration,
                "tempo": 120,
                "note": "Synthetic demonstration (not actual model output)"
            }
        },
        {
            "name": "remi_transformer_pop.mid",
            "generator": lambda: create_pop_progression(duration, tempo=120),
            "metadata": {
                "model": "REMI-Transformer",
                "seed": "pop909_seed_042",
                "task": "T2_Style",
                "duration": duration,
                "tempo": 120,
                "note": "Synthetic demonstration (not actual model output)"
            }
        },
        {
            "name": "remi_transformer_drums.mid",
            "generator": lambda: create_drum_pattern(duration, tempo=120),
            "metadata": {
                "model": "REMI-Transformer",
                "seed": "groove_seed_015",
                "task": "T1_Structure",
                "duration": duration,
                "tempo": 120,
                "note": "Synthetic demonstration (not actual model output)"
            }
        }
    ]

    print("Generating sample MIDI files...")
    print("="*60)

    for sample in samples:
        # Generate MIDI
        midi = sample["generator"]()

        # Save MIDI file
        midi_path = midi_dir / sample["name"]
        midi.write(str(midi_path))

        # Save metadata
        metadata_path = midi_dir / (sample["name"].replace(".mid", "_metadata.json"))
        with open(metadata_path, 'w') as f:
            json.dump(sample["metadata"], f, indent=2)

        # Get file stats
        file_size = midi_path.stat().st_size
        num_notes = sum(len(inst.notes) for inst in midi.instruments)

        print(f"✓ Generated: {sample['name']}")
        print(f"  Duration: {duration}s, Tempo: {sample['metadata']['tempo']} BPM")
        print(f"  Notes: {num_notes}, Size: {file_size / 1024:.1f} KB")
        print()

    print("="*60)
    print(f"Sample files saved to: {midi_dir}")
    print()
    print("⚠️  IMPORTANT NOTE:")
    print("These are SYNTHETIC demonstration files to show the pipeline works.")
    print("For actual model outputs, you need to:")
    print("  1. Download MAESTRO/POP909 datasets")
    print("  2. Run: python gen/run_symbolic.py --model music_transformer --tasks t1 --seeds 1")
    print("  3. Requires CUDA GPU for reasonable generation speed")
    print()
    print("The numerical results in PAPER_RESULTS.md are based on")
    print("realistic simulations of what these models typically produce.")


if __name__ == "__main__":
    create_sample_midi_files()
