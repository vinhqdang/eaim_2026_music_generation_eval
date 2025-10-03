"""
Comprehensive MIDI evaluation script.

This script evaluates all symbolic music generations by computing pitch-class,
voice-leading, rhythm, motif, and perplexity metrics. Supports parallel processing,
batch evaluation, graceful failure handling, and incremental updates.

Usage:
    python evaluate_midi.py --midi_dir runs/artifacts/midi --output runs/logs/midi_metrics.parquet
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metrics.midi.pitchclass import PitchClassMetricCalculator
from metrics.midi.voiceleading import VoiceLeadingMetricCalculator
from metrics.midi.rhythm import RhythmMetricCalculator
from metrics.midi.motif import MotifMetricCalculator
from metrics.midi.ppl import PerplexityMetricCalculator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('midi_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)


class MIDIEvaluator:
    """
    Comprehensive MIDI evaluator with parallel processing support.

    This class evaluates MIDI files using multiple metrics including pitch-class
    analysis, voice-leading, rhythm, motif development, and perplexity.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        n_workers: Optional[int] = None,
        batch_size: int = 16
    ):
        """
        Initialize MIDI evaluator.

        Args:
            device: Device for computation ('cuda' or 'cpu')
            n_workers: Number of worker processes (None = CPU count)
            batch_size: Batch size for parallel processing
        """
        self.device = device
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.batch_size = batch_size

        logger.info(f"Initializing MIDIEvaluator with {self.n_workers} workers")

        # Initialize calculators (will be initialized per process)
        self.pitchclass = None
        self.voiceleading = None
        self.rhythm = None
        self.motif = None
        self.perplexity = None

    def _init_calculators(self):
        """Initialize metric calculators (called in each worker process)."""
        try:
            self.pitchclass = PitchClassMetricCalculator()
            logger.info("Initialized pitch-class calculator")
        except Exception as e:
            logger.warning(f"Failed to initialize pitch-class calculator: {e}")
            self.pitchclass = None

        try:
            self.voiceleading = VoiceLeadingMetricCalculator()
            logger.info("Initialized voice-leading calculator")
        except Exception as e:
            logger.warning(f"Failed to initialize voice-leading calculator: {e}")
            self.voiceleading = None

        try:
            self.rhythm = RhythmMetricCalculator()
            logger.info("Initialized rhythm calculator")
        except Exception as e:
            logger.warning(f"Failed to initialize rhythm calculator: {e}")
            self.rhythm = None

        try:
            self.motif = MotifMetricCalculator()
            logger.info("Initialized motif calculator")
        except Exception as e:
            logger.warning(f"Failed to initialize motif calculator: {e}")
            self.motif = None

        try:
            self.perplexity = PerplexityMetricCalculator(device=self.device)
            logger.info("Initialized perplexity calculator")
        except Exception as e:
            logger.warning(f"Failed to initialize perplexity calculator: {e}")
            self.perplexity = None

        logger.info("All calculators initialized")

    def _compute_single_file_metrics(
        self,
        midi_path: Path,
        seed_path: Optional[Path] = None,
        reference_corpus: Optional[List[Path]] = None
    ) -> Dict[str, any]:
        """
        Compute all metrics for a single MIDI file.

        Args:
            midi_path: Path to MIDI file
            seed_path: Optional path to seed MIDI for continuation metrics
            reference_corpus: Optional reference corpus for perplexity

        Returns:
            Dictionary of computed metrics
        """
        start_time = time.time()
        metrics = {
            'file_path': str(midi_path),
            'file_name': midi_path.name,
            'has_seed': seed_path is not None,
            'success': False,
            'error': None
        }

        try:
            # Compute pitch-class metrics
            if self.pitchclass:
                try:
                    pc_metrics = self.pitchclass.compute(midi_path, seed_path)
                    for key, val in pc_metrics.items():
                        metrics[f'pc_{key}'] = val
                except Exception as e:
                    logger.warning(f"Pitch-class calculation failed for {midi_path.name}: {e}")
                    metrics['pc_error'] = str(e)

            # Compute voice-leading metrics
            if self.voiceleading:
                try:
                    vl_metrics = self.voiceleading.compute(midi_path, seed_path)
                    for key, val in vl_metrics.items():
                        metrics[f'vl_{key}'] = val
                except Exception as e:
                    logger.warning(f"Voice-leading calculation failed for {midi_path.name}: {e}")
                    metrics['vl_error'] = str(e)

            # Compute rhythm metrics
            if self.rhythm:
                try:
                    rhythm_metrics = self.rhythm.compute(midi_path, seed_path)
                    for key, val in rhythm_metrics.items():
                        metrics[f'rhythm_{key}'] = val
                except Exception as e:
                    logger.warning(f"Rhythm calculation failed for {midi_path.name}: {e}")
                    metrics['rhythm_error'] = str(e)

            # Compute motif metrics
            if self.motif:
                try:
                    motif_metrics = self.motif.compute(midi_path, seed_path)
                    for key, val in motif_metrics.items():
                        metrics[f'motif_{key}'] = val
                except Exception as e:
                    logger.warning(f"Motif calculation failed for {midi_path.name}: {e}")
                    metrics['motif_error'] = str(e)

            # Compute perplexity metrics
            if self.perplexity:
                try:
                    # Convert reference_corpus paths to strings
                    ref_corpus_str = None
                    if reference_corpus:
                        ref_corpus_str = [str(p) for p in reference_corpus]

                    ppl_metrics = self.perplexity.compute(
                        midi_path,
                        seed_path,
                        ref_corpus_str
                    )
                    for key, val in ppl_metrics.items():
                        metrics[f'ppl_{key}'] = val
                except Exception as e:
                    logger.warning(f"Perplexity calculation failed for {midi_path.name}: {e}")
                    metrics['ppl_error'] = str(e)

            metrics['success'] = True
            metrics['computation_time'] = time.time() - start_time

        except Exception as e:
            logger.error(f"Error processing {midi_path.name}: {e}")
            metrics['error'] = str(e)
            metrics['computation_time'] = time.time() - start_time

        return metrics

    def evaluate_batch(
        self,
        midi_paths: List[Path],
        seed_paths: Optional[List[Path]] = None,
        reference_corpus: Optional[List[Path]] = None
    ) -> pd.DataFrame:
        """
        Evaluate a batch of MIDI files.

        Args:
            midi_paths: List of MIDI file paths
            seed_paths: Optional list of seed MIDI paths (must match midi_paths length)
            reference_corpus: Optional reference corpus for perplexity

        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"Evaluating {len(midi_paths)} MIDI files")

        # Initialize calculators in main process
        if self.pitchclass is None:
            self._init_calculators()

        # Prepare arguments for parallel processing
        if seed_paths:
            if len(seed_paths) != len(midi_paths):
                logger.warning("Seed paths length doesn't match MIDI files, ignoring seeds")
                seed_paths = None

        # Process files in parallel with progress bar
        results = []
        with Pool(processes=self.n_workers, initializer=self._init_worker) as pool:
            if seed_paths:
                tasks = [
                    (path, seed, reference_corpus)
                    for path, seed in zip(midi_paths, seed_paths)
                ]
            else:
                tasks = [
                    (path, None, reference_corpus)
                    for path in midi_paths
                ]

            # Use imap_unordered for better performance with progress bar
            for result in tqdm(
                pool.imap_unordered(self._process_single_file_wrapper, tasks),
                total=len(tasks),
                desc="Processing MIDI files"
            ):
                results.append(result)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Log summary
        success_rate = df['success'].sum() / len(df) if len(df) > 0 else 0
        logger.info(f"Evaluation complete: {success_rate:.1%} success rate")

        return df

    @staticmethod
    def _init_worker():
        """Initialize worker process (if needed for multiprocessing)."""
        pass

    @staticmethod
    def _process_single_file_wrapper(
        args: Tuple[Path, Optional[Path], Optional[List[Path]]]
    ) -> Dict:
        """
        Wrapper for processing single file (for multiprocessing).

        Args:
            args: Tuple of (midi_path, seed_path, reference_corpus)

        Returns:
            Dictionary of metrics
        """
        midi_path, seed_path, reference_corpus = args

        # Create temporary evaluator for this worker
        evaluator = MIDIEvaluator(n_workers=1)
        evaluator._init_calculators()

        return evaluator._compute_single_file_metrics(
            midi_path,
            seed_path,
            reference_corpus
        )

    def load_existing_results(self, output_path: Path) -> Optional[pd.DataFrame]:
        """
        Load existing evaluation results for incremental updates.

        Args:
            output_path: Path to existing results file

        Returns:
            DataFrame of existing results or None if not found
        """
        if output_path.exists():
            try:
                df = pd.read_parquet(output_path)
                logger.info(f"Loaded {len(df)} existing results from {output_path}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")
        return None

    def incremental_evaluate(
        self,
        midi_paths: List[Path],
        output_path: Path,
        seed_paths: Optional[List[Path]] = None,
        reference_corpus: Optional[List[Path]] = None,
        force_recompute: bool = False
    ) -> pd.DataFrame:
        """
        Evaluate MIDI files incrementally, skipping already processed files.

        Args:
            midi_paths: List of MIDI file paths
            output_path: Path to save results
            seed_paths: Optional seed MIDI paths
            reference_corpus: Optional reference corpus for perplexity
            force_recompute: Force recomputation of all files

        Returns:
            DataFrame with all results (existing + new)
        """
        # Load existing results
        existing_df = None if force_recompute else self.load_existing_results(output_path)

        # Determine which files need processing
        if existing_df is not None and not existing_df.empty:
            existing_files = set(existing_df['file_path'].values)
            new_paths = [p for p in midi_paths if str(p) not in existing_files]
            logger.info(f"Found {len(new_paths)} new files to process (skipping {len(existing_files)})")
        else:
            new_paths = midi_paths
            logger.info(f"Processing all {len(new_paths)} files")

        # Process new files if any
        if new_paths:
            # Filter seed paths for new paths
            if seed_paths:
                path_to_seed = {str(p): s for p, s in zip(midi_paths, seed_paths)}
                new_seeds = [path_to_seed[str(p)] for p in new_paths]
            else:
                new_seeds = None

            new_df = self.evaluate_batch(new_paths, new_seeds, reference_corpus)

            # Merge with existing results
            if existing_df is not None and not existing_df.empty:
                df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                df = new_df
        else:
            df = existing_df
            logger.info("No new files to process")

        return df


def load_midi_metadata(metadata_path: Path) -> Optional[pd.DataFrame]:
    """
    Load metadata file containing seed mappings and other info.

    Args:
        metadata_path: Path to metadata CSV/JSON/parquet file

    Returns:
        DataFrame with metadata or None
    """
    if not metadata_path.exists():
        return None

    try:
        if metadata_path.suffix == '.csv':
            return pd.read_csv(metadata_path)
        elif metadata_path.suffix == '.json':
            return pd.read_json(metadata_path)
        elif metadata_path.suffix == '.parquet':
            return pd.read_parquet(metadata_path)
    except Exception as e:
        logger.warning(f"Failed to load metadata from {metadata_path}: {e}")

    return None


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate MIDI generations with comprehensive symbolic music metrics"
    )
    parser.add_argument(
        '--midi_dir',
        type=Path,
        default=Path('runs/artifacts/midi'),
        help='Directory containing MIDI files to evaluate'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('runs/logs/midi_metrics.parquet'),
        help='Output path for metrics (parquet format)'
    )
    parser.add_argument(
        '--seed_dir',
        type=Path,
        default=None,
        help='Directory containing seed MIDI files for continuation evaluation'
    )
    parser.add_argument(
        '--reference_corpus',
        type=Path,
        default=None,
        help='Directory containing reference MIDI corpus for perplexity'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        default=None,
        help='Path to metadata file with seed mappings (CSV/JSON/parquet)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.mid',
        help='File pattern for MIDI files (default: *.mid)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device for computation (cuda/cpu, default: auto)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker processes (default: CPU count - 1)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for processing (default: 16)'
    )
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Enable incremental evaluation (skip already processed files)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recomputation of all files'
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.midi_dir.exists():
        logger.error(f"MIDI directory not found: {args.midi_dir}")
        sys.exit(1)

    # Find MIDI files
    midi_paths = sorted(args.midi_dir.glob(args.pattern))
    if not midi_paths:
        # Try alternate extension
        midi_paths = sorted(args.midi_dir.glob('*.midi'))

    if not midi_paths:
        logger.error(f"No MIDI files found in {args.midi_dir} with pattern {args.pattern}")
        sys.exit(1)

    logger.info(f"Found {len(midi_paths)} MIDI files")

    # Load seed paths if provided
    seed_paths = None
    if args.seed_dir and args.seed_dir.exists():
        seed_paths = sorted(args.seed_dir.glob(args.pattern))
        if not seed_paths:
            seed_paths = sorted(args.seed_dir.glob('*.midi'))

        if seed_paths:
            logger.info(f"Found {len(seed_paths)} seed MIDI files")
        else:
            logger.warning("No seed files found in seed directory")
            seed_paths = None

    # Load metadata if provided
    if args.metadata:
        metadata_df = load_midi_metadata(args.metadata)
        if metadata_df is not None and 'seed_file' in metadata_df.columns:
            # Match seeds to MIDI files
            path_to_seed = dict(zip(metadata_df['file_name'], metadata_df['seed_file']))
            seed_dir = args.seed_dir or args.midi_dir
            seed_paths = []
            for midi_path in midi_paths:
                seed_name = path_to_seed.get(midi_path.name, None)
                if seed_name:
                    seed_path = seed_dir / seed_name
                    seed_paths.append(seed_path if seed_path.exists() else None)
                else:
                    seed_paths.append(None)
            logger.info(f"Loaded seed mappings for {sum(1 for s in seed_paths if s)} files")

    # Load reference corpus if provided
    reference_corpus = None
    if args.reference_corpus and args.reference_corpus.exists():
        reference_corpus = sorted(args.reference_corpus.glob(args.pattern))
        if not reference_corpus:
            reference_corpus = sorted(args.reference_corpus.glob('*.midi'))
        logger.info(f"Found {len(reference_corpus)} reference MIDI files")

    # Create evaluator
    evaluator = MIDIEvaluator(
        device=args.device,
        n_workers=args.workers,
        batch_size=args.batch_size
    )

    # Run evaluation
    start_time = time.time()

    if args.incremental:
        results_df = evaluator.incremental_evaluate(
            midi_paths,
            args.output,
            seed_paths,
            reference_corpus,
            force_recompute=args.force
        )
    else:
        results_df = evaluator.evaluate_batch(
            midi_paths,
            seed_paths,
            reference_corpus
        )

    total_time = time.time() - start_time

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

    # Print summary statistics
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total files processed: {len(results_df)}")
    logger.info(f"Success rate: {results_df['success'].sum() / len(results_df):.1%}")
    logger.info(f"Total computation time: {total_time:.2f}s")
    logger.info(f"Average time per file: {total_time / len(results_df):.2f}s")

    # Print metric statistics by category
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns

    metric_categories = {
        'Pitch-Class': [c for c in numeric_cols if c.startswith('pc_')],
        'Voice-Leading': [c for c in numeric_cols if c.startswith('vl_')],
        'Rhythm': [c for c in numeric_cols if c.startswith('rhythm_')],
        'Motif': [c for c in numeric_cols if c.startswith('motif_')],
        'Perplexity': [c for c in numeric_cols if c.startswith('ppl_')]
    }

    for category, cols in metric_categories.items():
        if cols:
            logger.info(f"\n{category} Metrics:")
            for col in cols[:5]:  # Show first 5 metrics per category
                mean_val = results_df[col].mean()
                std_val = results_df[col].std()
                logger.info(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")
            if len(cols) > 5:
                logger.info(f"  ... and {len(cols) - 5} more metrics")

    logger.info("="*80)


if __name__ == "__main__":
    main()
