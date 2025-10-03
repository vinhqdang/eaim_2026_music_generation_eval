# Music Generation Evaluation Framework

A comprehensive evaluation framework for behavioral assessment of music generation models without human evaluation. Evaluates 4 models across 3 behavioral tasks using 13 automated metrics.

**For EAIM 2026 Workshop** ([https://amaai-lab.github.io/EAIM2026/](https://amaai-lab.github.io/EAIM2026/))

See **[resultv1.md](resultv1.md)** for complete implementation details and documentation.

## Quick Start

```bash
# 1. Setup environment
conda create -n py310 python=3.10
conda activate py310
pip install -r requirements.txt

# 2. Prepare data
python prep/build_prompts.py
python prep/select_midi_seeds.py

# 3. Generate music (audio + symbolic)
python gen/run_audio.py --model musicgen --tasks t1,t2,t3 --seeds 3
python gen/run_symbolic.py --model music_transformer --tasks t1,t2,t3 --seeds 3

# 4. Evaluate
python eval/audio/evaluate_audio.py --wav_dir runs/artifacts/wav --output runs/logs/audio_metrics.parquet
python eval/midi/evaluate_midi.py --midi_dir runs/artifacts/midi --output runs/logs/midi_metrics.parquet

# 5. Analyze results
python analysis/aggregate.py
```

## Features

This framework provides five comprehensive audio metric modules:

1. **FAD (Fréchet Audio Distance)** - Measures distribution distance between reference and generated audio
2. **CLAP Score** - Evaluates text-audio alignment using contrastive learning
3. **Tempo Consistency** - Analyzes beat tracking and tempo stability
4. **Key Stability** - Detects musical key and measures tonal consistency
5. **Structure Detection** - Identifies musical structure via novelty curves

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- Conda package manager

### Setup

1. Create and activate conda environment:
```bash
conda create -n py310 python=3.10
conda activate py310
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The framework will automatically use CUDA if available.

## Usage

### Music Generation

#### Audio Generation (Text-to-Audio)

Generate audio samples using MusicGen or Stable Audio models:

```bash
# Activate conda environment
conda activate py310

# Generate with MusicGen for all tasks
python gen/run_audio.py --model musicgen --tasks t1,t2,t3 --seeds 3

# Generate with Stable Audio for specific task
python gen/run_audio.py --model stableaudio --tasks t1 --seeds 5

# Resume interrupted generation
python gen/run_audio.py --model musicgen --tasks t1,t2,t3 --resume
```

**Key features:**
- Loads prompts from `data/prompts/prompts_text.json`
- Supports MusicGen and Stable Audio models
- Generates 100 prompts × seeds per task
- Saves WAV files and metadata to `runs/artifacts/`
- Progress tracking with tqdm
- Resume functionality for interrupted runs
- Automatic CUDA detection

**Command-line options:**
- `--model`: Model to use (musicgen, stableaudio)
- `--tasks`: Comma-separated tasks (t1,t2,t3)
- `--seeds`: Number of random seeds per prompt (default: 3)
- `--prompts`: Path to prompts JSON file
- `--output-dir`: Output directory (default: runs/artifacts)
- `--log-dir`: Log directory (default: runs/logs)
- `--device`: Device to use (cuda/cpu, auto-detects if not specified)
- `--resume`: Resume from previous run

#### Symbolic Generation (MIDI)

Generate symbolic music using Music Transformer or REMI Transformer:

```bash
# Generate with Music Transformer for all tasks
python gen/run_symbolic.py --model music_transformer --tasks t1,t2,t3 --seeds 3

# Generate with REMI Transformer for specific task
python gen/run_symbolic.py --model remi_transformer --tasks t1 --seeds 5

# Resume interrupted generation
python gen/run_symbolic.py --model music_transformer --tasks t1,t2,t3 --resume
```

**Key features:**
- Loads MIDI seeds from `data/midi_seeds.json`
- Supports Music Transformer and REMI Transformer models
- Generates 100 seeds × seeds per task
- Saves MIDI files and metadata to `runs/artifacts/`
- Progress tracking with tqdm
- Resume functionality for interrupted runs
- Automatic CUDA detection

**Command-line options:**
- `--model`: Model to use (music_transformer, remi_transformer)
- `--tasks`: Comma-separated tasks (t1,t2,t3)
- `--seeds`: Number of random seeds per MIDI seed (default: 3)
- `--midi-seeds`: Path to MIDI seeds JSON file
- `--output-dir`: Output directory (default: runs/artifacts)
- `--log-dir`: Log directory (default: runs/logs)
- `--device`: Device to use (cuda/cpu, auto-detects if not specified)
- `--resume`: Resume from previous run

### Audio Metrics

### 1. FAD (Fréchet Audio Distance)

Calculate FAD between reference and generated audio using CLAP or VGGish embeddings.

```python
from metrics.audio import FADCalculator

# Initialize with CLAP embeddings
calculator = FADCalculator(embedding_type="clap")

# Compute FAD
result = calculator.compute(
    reference_paths=["ref1.wav", "ref2.wav", "ref3.wav"],
    generated_paths=["gen1.wav", "gen2.wav", "gen3.wav"]
)

print(f"FAD Score: {result['fad_score']:.4f}")
```

**Command-line usage:**
```bash
python metrics/audio/fad.py /path/to/reference_dir /path/to/generated_dir
```

**Key metrics:**
- `fad_score`: Lower is better (measures distribution distance)
- `embedding_type`: Type of embedding used
- `n_reference`, `n_generated`: Number of samples

### 2. CLAP Score

Measure text-audio alignment using CLAP embeddings.

```python
from metrics.audio import CLAPScoreCalculator

calculator = CLAPScoreCalculator()

# Compute CLAP Score for matching pairs
result = calculator.compute(
    audio_paths=["song1.wav", "song2.wav"],
    text_descriptions=["piano melody", "guitar riff"],
    return_individual_scores=True
)

print(f"CLAP Score: {result['clap_score']:.4f}")
```

**Command-line usage:**
```bash
python metrics/audio/clapscore.py "song1.wav,song2.wav" "piano melody|guitar riff"
```

**Key metrics:**
- `clap_score`: Average similarity (higher is better, 0-1 range)
- `std`: Standard deviation of scores
- `individual_scores`: Per-sample scores

**Advanced usage:**
```python
# Cross-similarity matrix
result = calculator.compute_cross_similarity(
    audio_paths=audio_files,
    text_descriptions=texts
)

# Rank texts for audio
rankings = calculator.rank_texts_for_audio(
    audio_path="song.wav",
    candidate_texts=["piano", "guitar", "drums"],
    top_k=3
)
```

### 3. Tempo Consistency

Analyze beat tracking and tempo stability.

```python
from metrics.audio import TempoConsistencyCalculator

calculator = TempoConsistencyCalculator()

# Single file analysis
result = calculator.compute("music.wav")

print(f"Tempo: {result['global_tempo']:.2f} BPM")
print(f"Stability: {result['tempo_stability']:.4f}")
print(f"Beat Regularity: {result['beat_regularity']:.4f}")
```

**Command-line usage:**
```bash
python metrics/audio/tempo.py music1.wav music2.wav music3.wav
```

**Key metrics:**
- `global_tempo`: Estimated tempo in BPM
- `tempo_stability`: Consistency of tempo over time (0-1, higher is better)
- `beat_regularity`: Regularity of beat intervals (0-1, higher is better)
- `avg_beat_strength`: Average strength of detected beats

**Batch processing:**
```python
# Analyze multiple files
result = calculator.compute_batch(audio_paths)
print(f"Avg Tempo Stability: {result['avg_tempo_stability']:.4f}")
```

### 4. Key Stability

Detect musical key and measure tonal consistency.

```python
from metrics.audio import KeyStabilityCalculator

calculator = KeyStabilityCalculator()

result = calculator.compute("music.wav")

print(f"Key: {result['global_key']} {result['global_mode']}")
print(f"Confidence: {result['key_confidence']:.4f}")
print(f"Key Stability: {result['key_stability']:.4f}")
print(f"Tonal Strength: {result['tonal_strength']:.4f}")
```

**Command-line usage:**
```bash
python metrics/audio/key_stability.py music1.wav music2.wav
```

**Key metrics:**
- `global_key`: Detected key (C, D, E, F, G, A, B with # for sharps)
- `global_mode`: Major or minor
- `key_confidence`: Confidence of key detection (0-1)
- `key_stability`: Consistency of key over time (0-1, higher is better)
- `tonal_strength`: Strength of tonal content (0-1, higher is better)
- `n_key_changes`: Number of key changes detected

### 5. Structure Detection

Identify musical structure via novelty curves.

```python
from metrics.audio import StructureDetectionCalculator

calculator = StructureDetectionCalculator()

result = calculator.compute("music.wav")

print(f"Segments: {result['n_segments']}")
print(f"Repetition Score: {result['repetition_score']:.4f}")
print(f"Complexity: {result['structural_complexity']:.4f}")
print(f"Boundaries: {result['segment_boundary_times']}")
```

**Command-line usage:**
```bash
python metrics/audio/structure.py music1.wav music2.wav
```

**Key metrics:**
- `n_segments`: Number of detected structural segments
- `repetition_score`: Measure of repetitive structure (0-1)
- `structural_complexity`: Overall structural complexity
- `avg_segment_homogeneity`: Consistency within segments (0-1)
- `segment_boundary_times`: Timestamps of segment boundaries

## Comprehensive Evaluation Scripts

### Audio Evaluation

Evaluate all audio generations with comprehensive metrics in parallel:

```bash
# Activate conda environment
conda activate py310

# Basic evaluation
python eval/audio/evaluate_audio.py \
  --wav_dir runs/artifacts/wav \
  --output runs/logs/audio_metrics.parquet

# With reference audio for FAD computation
python eval/audio/evaluate_audio.py \
  --wav_dir runs/artifacts/wav \
  --reference_dir data/reference_audio \
  --output runs/logs/audio_metrics.parquet

# With text prompts for CLAP score
python eval/audio/evaluate_audio.py \
  --wav_dir runs/artifacts/wav \
  --metadata data/audio_metadata.csv \
  --output runs/logs/audio_metrics.parquet

# Incremental evaluation (skip already processed files)
python eval/audio/evaluate_audio.py \
  --wav_dir runs/artifacts/wav \
  --output runs/logs/audio_metrics.parquet \
  --incremental

# Force recomputation of all files
python eval/audio/evaluate_audio.py \
  --wav_dir runs/artifacts/wav \
  --output runs/logs/audio_metrics.parquet \
  --force
```

**Features:**
- Computes FAD (VGGish + CLAP), CLAPScore, tempo, key, structure metrics
- Parallel processing with multiprocessing
- Progress bars with tqdm
- Graceful error handling
- Incremental updates (resume/skip processed files)
- Saves results to Parquet format
- Logs computation time per metric

**Command-line options:**
- `--wav_dir`: Directory containing WAV files (default: runs/artifacts/wav)
- `--output`: Output path for metrics parquet (default: runs/logs/audio_metrics.parquet)
- `--reference_dir`: Optional reference audio directory for FAD
- `--metadata`: Optional metadata file (CSV/JSON/parquet) with text_prompt column
- `--pattern`: File pattern for audio files (default: *.wav)
- `--device`: Device for computation (cuda/cpu, auto-detects if not specified)
- `--workers`: Number of worker processes (default: CPU count - 1)
- `--batch_size`: Batch size for processing (default: 8)
- `--incremental`: Enable incremental evaluation
- `--force`: Force recomputation of all files

**Metadata file format:**
CSV/JSON/Parquet file with columns:
- `file_name`: Audio file name
- `text_prompt`: Text description for CLAP score (optional)

**Output format:**
Parquet file with columns:
- `file_path`, `file_name`: File identification
- `tempo_*`: Tempo consistency metrics
- `key_*`: Key stability metrics
- `structure_*`: Structure detection metrics
- `clap_*`: CLAP score metrics (if prompts provided)
- `fad_vggish_score`, `fad_clap_score`: FAD scores (if reference provided)
- `success`, `error`: Processing status
- `computation_time`: Time taken per file

### MIDI Evaluation

Evaluate all symbolic music generations with comprehensive metrics:

```bash
# Activate conda environment
conda activate py310

# Basic evaluation
python eval/midi/evaluate_midi.py \
  --midi_dir runs/artifacts/midi \
  --output runs/logs/midi_metrics.parquet

# With seed MIDI files for continuation metrics
python eval/midi/evaluate_midi.py \
  --midi_dir runs/artifacts/midi \
  --seed_dir data/midi_seeds \
  --output runs/logs/midi_metrics.parquet

# With reference corpus for perplexity
python eval/midi/evaluate_midi.py \
  --midi_dir runs/artifacts/midi \
  --reference_corpus data/reference_midi \
  --output runs/logs/midi_metrics.parquet

# With metadata for seed mappings
python eval/midi/evaluate_midi.py \
  --midi_dir runs/artifacts/midi \
  --metadata data/midi_metadata.csv \
  --output runs/logs/midi_metrics.parquet

# Incremental evaluation
python eval/midi/evaluate_midi.py \
  --midi_dir runs/artifacts/midi \
  --output runs/logs/midi_metrics.parquet \
  --incremental
```

**Features:**
- Computes pitch-class, voice-leading, rhythm, motif, perplexity metrics
- Parallel processing with multiprocessing
- Progress bars with tqdm
- Graceful error handling
- Incremental updates (resume/skip processed files)
- Saves results to Parquet format
- Logs computation time per metric

**Command-line options:**
- `--midi_dir`: Directory containing MIDI files (default: runs/artifacts/midi)
- `--output`: Output path for metrics parquet (default: runs/logs/midi_metrics.parquet)
- `--seed_dir`: Optional seed MIDI directory for continuation metrics
- `--reference_corpus`: Optional reference corpus for perplexity
- `--metadata`: Optional metadata file (CSV/JSON/parquet) with seed_file column
- `--pattern`: File pattern for MIDI files (default: *.mid)
- `--device`: Device for computation (cuda/cpu, auto-detects if not specified)
- `--workers`: Number of worker processes (default: CPU count - 1)
- `--batch_size`: Batch size for processing (default: 16)
- `--incremental`: Enable incremental evaluation
- `--force`: Force recomputation of all files

**Metadata file format:**
CSV/JSON/Parquet file with columns:
- `file_name`: MIDI file name
- `seed_file`: Seed MIDI file name (optional)

**Output format:**
Parquet file with columns:
- `file_path`, `file_name`, `has_seed`: File identification
- `pc_*`: Pitch-class metrics
- `vl_*`: Voice-leading metrics
- `rhythm_*`: Rhythm metrics
- `motif_*`: Motif development metrics
- `ppl_*`: Perplexity metrics
- `success`, `error`: Processing status
- `computation_time`: Time taken per file

### Complete Evaluation Pipeline

Run both audio and MIDI evaluations:

```bash
# Activate conda environment
conda activate py310

# Step 1: Evaluate audio
python eval/audio/evaluate_audio.py \
  --wav_dir runs/artifacts/wav \
  --reference_dir data/reference_audio \
  --metadata data/audio_metadata.csv \
  --output runs/logs/audio_metrics.parquet \
  --incremental

# Step 2: Evaluate MIDI
python eval/midi/evaluate_midi.py \
  --midi_dir runs/artifacts/midi \
  --seed_dir data/midi_seeds \
  --reference_corpus data/reference_midi \
  --metadata data/midi_metadata.csv \
  --output runs/logs/midi_metrics.parquet \
  --incremental

# Step 3: Analyze results (optional)
python analysis/analyze_metrics.py \
  --audio_metrics runs/logs/audio_metrics.parquet \
  --midi_metrics runs/logs/midi_metrics.parquet \
  --output_dir runs/analysis
```

### Loading and Analyzing Results

```python
import pandas as pd
import numpy as np

# Load audio metrics
audio_df = pd.read_parquet('runs/logs/audio_metrics.parquet')

# Show summary statistics
print("Audio Metrics Summary:")
print(audio_df[['tempo_global_tempo', 'key_key_stability', 'structure_repetition_score']].describe())

# Load MIDI metrics
midi_df = pd.read_parquet('runs/logs/midi_metrics.parquet')

# Show summary statistics
print("\nMIDI Metrics Summary:")
print(midi_df[['pc_pitch_class_entropy', 'rhythm_rhythm_regularity', 'motif_motif_repetition_rate']].describe())

# Filter successful evaluations
successful_audio = audio_df[audio_df['success']]
print(f"\nAudio success rate: {len(successful_audio) / len(audio_df):.1%}")

successful_midi = midi_df[midi_df['success']]
print(f"MIDI success rate: {len(successful_midi) / len(midi_df):.1%}")
```

## Batch Processing

All calculators support batch processing for multiple files:

```python
from metrics.audio import (
    TempoConsistencyCalculator,
    KeyStabilityCalculator,
    StructureDetectionCalculator
)

audio_files = ["song1.wav", "song2.wav", "song3.wav"]

# Tempo analysis
tempo_calc = TempoConsistencyCalculator()
tempo_results = tempo_calc.compute_batch(audio_files)

# Key analysis
key_calc = KeyStabilityCalculator()
key_results = key_calc.compute_batch(audio_files)

# Structure analysis
structure_calc = StructureDetectionCalculator()
structure_results = structure_calc.compute_batch(audio_files)
```

## Design Patterns

All metric calculators follow a consistent OOP design:

### Base Interface
```python
class MetricCalculator:
    def __init__(self, **config):
        """Initialize with configuration parameters"""
        pass

    def compute(self, audio_path, **kwargs):
        """
        Compute metric for single audio file
        Returns: Dict[str, Union[float, int, np.ndarray]]
        """
        pass

    def compute_batch(self, audio_paths):
        """
        Compute metric for multiple audio files
        Returns: Dict with aggregated metrics
        """
        pass
```

### Key Features
- **Type hints**: All methods use proper type annotations
- **Error handling**: Graceful handling of edge cases and errors
- **Docstrings**: Comprehensive documentation for all classes and methods
- **Logging**: Informative progress messages during computation
- **Modularity**: Each metric is independent and self-contained

## Technical Details

### Audio Processing
- Default sample rate: 22050 Hz (adjustable)
- Hop length: 512 samples
- FFT size: 2048 samples

### Models Used
- **CLAP**: `laion/larger_clap_music` for text-audio alignment
- **VGGish**: torch.hub model for audio embeddings
- **Librosa**: For audio feature extraction
- **mir_eval**: For beat evaluation metrics

### Dependencies
See `requirements.txt` for full list. Key dependencies:
- PyTorch (CUDA support)
- Transformers (HuggingFace)
- librosa
- mir_eval
- essentia
- laion-clap

## Performance Considerations

### GPU Acceleration
- FAD and CLAP Score use GPU automatically if available
- Set `device='cpu'` to force CPU computation

### Memory Management
- Large batches are processed sequentially to avoid OOM
- Audio files are loaded on-demand

### Processing Time
Approximate processing times per audio file (30 seconds, on GPU):
- FAD: ~0.5-1s per file (embedding extraction)
- CLAP Score: ~0.5s per file
- Tempo: ~1-2s per file
- Key Stability: ~1-2s per file
- Structure: ~2-3s per file

## Examples

### Complete Evaluation Pipeline
```python
from pathlib import Path
from metrics.audio import (
    FADCalculator,
    CLAPScoreCalculator,
    TempoConsistencyCalculator,
    KeyStabilityCalculator,
    StructureDetectionCalculator
)

def evaluate_generated_music(ref_dir, gen_dir, prompts):
    """Complete evaluation of generated music."""

    # Get audio files
    ref_files = sorted(Path(ref_dir).glob("*.wav"))
    gen_files = sorted(Path(gen_dir).glob("*.wav"))

    results = {}

    # 1. FAD
    fad_calc = FADCalculator(embedding_type="clap")
    results["fad"] = fad_calc.compute(ref_files, gen_files)

    # 2. CLAP Score
    clap_calc = CLAPScoreCalculator()
    results["clap"] = clap_calc.compute(gen_files, prompts)

    # 3. Tempo Consistency
    tempo_calc = TempoConsistencyCalculator()
    results["tempo"] = tempo_calc.compute_batch(gen_files)

    # 4. Key Stability
    key_calc = KeyStabilityCalculator()
    results["key"] = key_calc.compute_batch(gen_files)

    # 5. Structure Detection
    struct_calc = StructureDetectionCalculator()
    results["structure"] = struct_calc.compute_batch(gen_files)

    return results

# Run evaluation
results = evaluate_generated_music(
    ref_dir="data/reference",
    gen_dir="data/generated",
    prompts=["piano melody", "guitar riff", "drum beat"]
)

print("Evaluation Results:")
print(f"FAD Score: {results['fad']['fad_score']:.4f}")
print(f"CLAP Score: {results['clap']['clap_score']:.4f}")
print(f"Avg Tempo Stability: {results['tempo']['avg_tempo_stability']:.4f}")
print(f"Avg Key Stability: {results['key']['avg_key_stability']:.4f}")
print(f"Avg Repetition Score: {results['structure']['avg_repetition_score']:.4f}")
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Use CPU instead: `device='cpu'`
- Process files individually

### Audio Loading Errors
- Ensure audio files are valid WAV or MP3 format
- Check sample rate compatibility
- Verify file paths are correct

### Model Download Issues
- Models are downloaded automatically from HuggingFace
- Ensure internet connection for first run
- Check HuggingFace cache: `~/.cache/huggingface/`

## Contributing

When adding new metrics:
1. Follow the OOP design pattern
2. Implement `compute()` and `compute_batch()` methods
3. Add comprehensive docstrings and type hints
4. Include error handling for edge cases
5. Update this README with usage examples
6. Add dependencies to requirements.txt

## License

See LICENSE file for details.

## Citation

If you use this framework in your research, please cite:
```bibtex
@software{music_generation_eval,
  title={Music Generation Evaluation Framework},
  year={2025},
  url={https://github.com/yourusername/eaim_2026_music_generation_eval}
}
```

## References

- FAD: Kilgour et al. "Fréchet Audio Distance: A Reference-free Metric for Evaluating Music Enhancement Algorithms"
- CLAP: Wu et al. "Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation"
- mir_eval: Raffel et al. "mir_eval: A Transparent Implementation of Common MIR Metrics"
- librosa: McFee et al. "librosa: Audio and Music Signal Analysis in Python"
