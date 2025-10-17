"""
Complete EAIM 2026 evaluation pipeline.
Generates all samples and computes all metrics for manuscript submission.
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"✗ Failed: {description}")
        return False
    print(f"✓ Completed: {description}")
    return True

def main():
    """Run complete evaluation pipeline."""

    print("="*60)
    print("COMPLETE EAIM 2026 EVALUATION PIPELINE")
    print("="*60)
    print("\nThis will generate:")
    print("- 900 audio samples (3 models × 300 samples)")
    print("- 600 MIDI samples (2 models × 300 samples)")
    print("- All audio metrics (FAD, CLAP, tempo, key, structure)")
    print("- All MIDI metrics (pitch-class, voice-leading, rhythm, motif, ppl)")
    print("\nEstimated time: 6-8 hours on GPU")
    print()

    # Step 1: Generate MusicGen-large samples
    if not run_command(
        "source ~/anaconda3/etc/profile.d/conda.sh && conda activate py310 && python3 run_musicgen_large.py",
        "Step 1/7: Generate 300 MusicGen-large samples"
    ):
        return

    # Step 2: Generate Music Transformer samples
    if not run_command(
        "source ~/anaconda3/etc/profile.d/conda.sh && conda activate py310 && python3 run_music_transformer.py",
        "Step 2/7: Generate 300 Music Transformer MIDI samples"
    ):
        return

    # Step 3: Generate REMI Transformer samples
    if not run_command(
        "source ~/anaconda3/etc/profile.d/conda.sh && conda activate py310 && python3 run_remi_transformer.py",
        "Step 3/7: Generate 300 REMI Transformer MIDI samples"
    ):
        return

    # Step 4: Compute all audio metrics
    if not run_command(
        "source ~/anaconda3/etc/profile.d/conda.sh && conda activate py310 && python3 eval/audio/evaluate_audio.py --audio-dir runs/artifacts/wav --output runs/results_COMPLETE/audio_metrics.parquet",
        "Step 4/7: Compute all audio metrics (FAD, CLAP, tempo, key, structure)"
    ):
        return

    # Step 5: Compute all MIDI metrics
    if not run_command(
        "source ~/anaconda3/etc/profile.d/conda.sh && conda activate py310 && python3 eval/midi/evaluate_midi.py --midi-dir runs/artifacts/midi --output runs/results_COMPLETE/midi_metrics.parquet",
        "Step 5/7: Compute all MIDI metrics"
    ):
        return

    # Step 6: Generate comprehensive analysis
    if not run_command(
        "source ~/anaconda3/etc/profile.d/conda.sh && conda activate py310 && MPLBACKEND=Agg python3 generate_manuscript_analysis.py",
        "Step 6/7: Generate comprehensive statistical analysis"
    ):
        return

    # Step 7: Create manuscript-ready document
    if not run_command(
        "source ~/anaconda3/etc/profile.d/conda.sh && conda activate py310 && python3 create_manuscript_results.py",
        "Step 7/7: Create manuscript-ready results document"
    ):
        return

    print("\n" + "="*60)
    print("✅ COMPLETE EVALUATION FINISHED")
    print("="*60)
    print("\nGenerated files:")
    print("- runs/artifacts/wav/musicgen_{small,medium,large}/ (900 WAV files)")
    print("- runs/artifacts/midi/{music_transformer,remi}/ (600 MIDI files)")
    print("- runs/results_COMPLETE/audio_metrics.parquet (all audio metrics)")
    print("- runs/results_COMPLETE/midi_metrics.parquet (all MIDI metrics)")
    print("- MANUSCRIPT_RESULTS.md (complete results for paper)")
    print("- runs/results_COMPLETE/figures/ (all visualization figures)")

if __name__ == "__main__":
    main()
