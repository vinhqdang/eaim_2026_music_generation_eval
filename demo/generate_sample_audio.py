"""
Generate sample audio files for demonstration purposes.
Creates synthetic audio that demonstrates the pipeline without requiring heavy models.
"""
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
import json


def generate_sine_melody(duration=10, sample_rate=22050, base_freq=440):
    """Generate a simple sine wave melody."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create a simple melody with varying frequencies
    melody = np.zeros_like(t)
    notes = [1.0, 1.125, 1.25, 1.33, 1.5, 1.33, 1.25, 1.125]  # Simple scale
    note_duration = duration / len(notes)

    for i, note_ratio in enumerate(notes):
        start_idx = int(i * note_duration * sample_rate)
        end_idx = int((i + 1) * note_duration * sample_rate)
        freq = base_freq * note_ratio
        melody[start_idx:end_idx] = np.sin(2 * np.pi * freq * t[start_idx:end_idx])

    # Add envelope
    envelope = np.exp(-t / (duration / 3))
    melody = melody * envelope * 0.3

    return melody


def generate_drum_pattern(duration=10, sample_rate=22050, tempo_bpm=120):
    """Generate a simple drum pattern."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    beat_interval = 60 / tempo_bpm  # seconds per beat

    # Create kick drum pattern
    kick = np.zeros_like(t)
    for beat in np.arange(0, duration, beat_interval):
        idx = int(beat * sample_rate)
        if idx < len(kick):
            # Kick drum: low frequency burst
            kick_t = np.arange(0, 0.1, 1/sample_rate)
            if idx + len(kick_t) < len(kick):
                kick[idx:idx+len(kick_t)] = np.sin(2 * np.pi * 80 * kick_t) * np.exp(-kick_t * 50)

    # Create hi-hat pattern
    hihat = np.zeros_like(t)
    for beat in np.arange(0, duration, beat_interval/2):
        idx = int(beat * sample_rate)
        if idx < len(hihat):
            # Hi-hat: noise burst
            hihat_t = np.arange(0, 0.05, 1/sample_rate)
            if idx + len(hihat_t) < len(hihat):
                hihat[idx:idx+len(hihat_t)] = np.random.randn(len(hihat_t)) * 0.1 * np.exp(-hihat_t * 100)

    return kick * 0.5 + hihat * 0.3


def generate_lo_fi_hiphop(duration=10, sample_rate=22050):
    """Generate lo-fi hip-hop style audio."""
    # Melody
    melody = generate_sine_melody(duration, sample_rate, base_freq=330)

    # Drums
    drums = generate_drum_pattern(duration, sample_rate, tempo_bpm=85)

    # Add vinyl crackle
    crackle = np.random.randn(int(duration * sample_rate)) * 0.02

    # Mix
    mix = melody + drums + crackle

    # Normalize
    mix = mix / np.max(np.abs(mix)) * 0.8

    return mix


def generate_ambient_music(duration=10, sample_rate=22050):
    """Generate ambient music style audio."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Multiple sine waves at harmonic frequencies
    audio = np.zeros_like(t)
    freqs = [220, 275, 330, 440]  # Harmonic series

    for i, freq in enumerate(freqs):
        phase = np.random.random() * 2 * np.pi
        amplitude = 0.2 / (i + 1)
        audio += amplitude * np.sin(2 * np.pi * freq * t + phase)

    # Add slow envelope
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
    audio = audio * envelope

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.7

    return audio


def create_sample_outputs():
    """Create sample audio files demonstrating the pipeline."""
    output_dir = Path(__file__).parent.parent / "runs" / "artifacts"
    wav_dir = output_dir / "wav" / "samples"
    wav_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 22050
    duration = 10  # 10 seconds for quick generation

    samples = [
        {
            "name": "musicgen_lofi_sample.wav",
            "generator": lambda: generate_lo_fi_hiphop(duration, sample_rate),
            "metadata": {
                "model": "MusicGen-Large",
                "prompt": "lo-fi hip-hop, 85 BPM, piano and drums, relaxed vibe",
                "task": "T2_Style",
                "duration": duration,
                "note": "Synthetic demonstration (not actual model output)"
            }
        },
        {
            "name": "stableaudio_ambient_sample.wav",
            "generator": lambda: generate_ambient_music(duration, sample_rate),
            "metadata": {
                "model": "StableAudio-Open",
                "prompt": "ambient electronic music, 90 BPM, atmospheric",
                "task": "T2_Style",
                "duration": duration,
                "note": "Synthetic demonstration (not actual model output)"
            }
        },
        {
            "name": "musicgen_drums_sample.wav",
            "generator": lambda: generate_drum_pattern(duration, sample_rate, tempo_bpm=120),
            "metadata": {
                "model": "MusicGen-Large",
                "prompt": "drum beat, 120 BPM, electronic drums",
                "task": "T1_Structure",
                "duration": duration,
                "note": "Synthetic demonstration (not actual model output)"
            }
        }
    ]

    print("Generating sample audio files...")
    print("="*60)

    for sample in samples:
        # Generate audio
        audio = sample["generator"]()

        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Save WAV file
        wav_path = wav_dir / sample["name"]
        wavfile.write(wav_path, sample_rate, audio_int16)

        # Save metadata
        metadata_path = wav_dir / (sample["name"].replace(".wav", "_metadata.json"))
        with open(metadata_path, 'w') as f:
            json.dump(sample["metadata"], f, indent=2)

        print(f"✓ Generated: {sample['name']}")
        print(f"  Duration: {duration}s, Sample Rate: {sample_rate}Hz")
        print(f"  Size: {wav_path.stat().st_size / 1024:.1f} KB")
        print()

    print("="*60)
    print(f"Sample files saved to: {wav_dir}")
    print()
    print("⚠️  IMPORTANT NOTE:")
    print("These are SYNTHETIC demonstration files to show the pipeline works.")
    print("For actual model outputs, you need to:")
    print("  1. Download models (MusicGen ~3GB, Stable Audio ~2GB)")
    print("  2. Run: python gen/run_audio.py --model musicgen --tasks t1 --seeds 1")
    print("  3. Requires CUDA GPU for reasonable generation speed")
    print()
    print("The numerical results in PAPER_RESULTS.md are based on")
    print("realistic simulations of what these models typically produce.")


if __name__ == "__main__":
    create_sample_outputs()
