"""
Comprehensive audio evaluation script.

This script evaluates all audio generations by computing FAD (VGGish + CLAP),
CLAPScore, tempo, key, and structure metrics. Supports parallel processing,
batch evaluation, graceful failure handling, and incremental updates.

Usage:
    python evaluate_audio.py --wav_dir runs/artifacts/wav --output runs/logs/audio_metrics.parquet
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

from metrics.audio.fad import FADCalculator
from metrics.audio.clapscore import CLAPScoreCalculator
from metrics.audio.tempo import TempoConsistencyCalculator
from metrics.audio.key_stability import KeyStabilityCalculator
from metrics.audio.structure import StructureDetectionCalculator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('audio_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)


class AudioEvaluator:
    """
    Comprehensive audio evaluator with parallel processing support.

    This class evaluates audio files using multiple metrics including FAD,
    CLAP score, tempo consistency, key stability, and structural analysis.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        n_workers: Optional[int] = None,
        batch_size: int = 8
    ):
        """
        Initialize audio evaluator.

        Args:
            device: Device for computation ('cuda' or 'cpu')
            n_workers: Number of worker processes (None = CPU count)
            batch_size: Batch size for parallel processing
        """
        self.device = device
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.batch_size = batch_size

        logger.info(f"Initializing AudioEvaluator with {self.n_workers} workers")

        # Initialize calculators (will be initialized per process)
        self.fad_vggish = None
        self.fad_clap = None
        self.clap_score = None
        self.tempo = None
        self.key_stability = None
        self.structure = None

    def _init_calculators(self):
        """Initialize metric calculators (called in each worker process)."""
        try:
            self.fad_vggish = FADCalculator(embedding_type="vggish", device=self.device)
            logger.info("Initialized VGGish FAD calculator")
        except Exception as e:
            logger.warning(f"Failed to initialize VGGish FAD: {e}")
            self.fad_vggish = None

        try:
            self.fad_clap = FADCalculator(embedding_type="clap", device=self.device)
            logger.info("Initialized CLAP FAD calculator")
        except Exception as e:
            logger.warning(f"Failed to initialize CLAP FAD: {e}")
            self.fad_clap = None

        try:
            self.clap_score = CLAPScoreCalculator(device=self.device)
            logger.info("Initialized CLAP score calculator")
        except Exception as e:
            logger.warning(f"Failed to initialize CLAP score: {e}")
            self.clap_score = None

        self.tempo = TempoConsistencyCalculator()
        self.key_stability = KeyStabilityCalculator()
        self.structure = StructureDetectionCalculator()

        logger.info("All calculators initialized")

    def _compute_single_file_metrics(
        self,
        audio_path: Path,
        text_prompt: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Compute all metrics for a single audio file.

        Args:
            audio_path: Path to audio file
            text_prompt: Optional text prompt for CLAP score

        Returns:
            Dictionary of computed metrics
        """
        start_time = time.time()
        metrics = {
            'file_path': str(audio_path),
            'file_name': audio_path.name,
            'success': False,
            'error': None
        }

        try:
            # Compute tempo metrics
            try:
                tempo_metrics = self.tempo.compute(audio_path)
                for key, val in tempo_metrics.items():
                    metrics[f'tempo_{key}'] = val
            except Exception as e:
                logger.warning(f"Tempo calculation failed for {audio_path.name}: {e}")
                metrics['tempo_error'] = str(e)

            # Compute key stability metrics
            try:
                key_metrics = self.key_stability.compute(audio_path)
                for key, val in key_metrics.items():
                    metrics[f'key_{key}'] = val
            except Exception as e:
                logger.warning(f"Key stability calculation failed for {audio_path.name}: {e}")
                metrics['key_error'] = str(e)

            # Compute structure metrics
            try:
                structure_metrics = self.structure.compute(audio_path)
                for key, val in structure_metrics.items():
                    if key not in ['boundary_times', 'segment_boundary_times']:
                        metrics[f'structure_{key}'] = val
            except Exception as e:
                logger.warning(f"Structure calculation failed for {audio_path.name}: {e}")
                metrics['structure_error'] = str(e)

            # Compute CLAP score if text prompt provided
            if text_prompt and self.clap_score:
                try:
                    clap_metrics = self.clap_score.compute([audio_path], [text_prompt])
                    for key, val in clap_metrics.items():
                        if key != 'individual_scores':
                            metrics[f'clap_{key}'] = val
                except Exception as e:
                    logger.warning(f"CLAP score calculation failed for {audio_path.name}: {e}")
                    metrics['clap_error'] = str(e)

            metrics['success'] = True
            metrics['computation_time'] = time.time() - start_time

        except Exception as e:
            logger.error(f"Error processing {audio_path.name}: {e}")
            metrics['error'] = str(e)
            metrics['computation_time'] = time.time() - start_time

        return metrics

    def _compute_fad_metrics(
        self,
        reference_paths: List[Path],
        generated_paths: List[Path]
    ) -> Dict[str, float]:
        """
        Compute FAD metrics comparing reference and generated audio.

        Args:
            reference_paths: List of reference audio paths
            generated_paths: List of generated audio paths

        Returns:
            Dictionary of FAD metrics
        """
        fad_metrics = {}

        # Compute VGGish FAD
        if self.fad_vggish and len(reference_paths) >= 2 and len(generated_paths) >= 2:
            try:
                logger.info("Computing VGGish FAD...")
                start_time = time.time()
                result = self.fad_vggish.compute(reference_paths, generated_paths)
                fad_metrics['fad_vggish_score'] = result['fad_score']
                fad_metrics['fad_vggish_time'] = time.time() - start_time
                logger.info(f"VGGish FAD: {result['fad_score']:.4f}")
            except Exception as e:
                logger.error(f"VGGish FAD computation failed: {e}")
                fad_metrics['fad_vggish_error'] = str(e)

        # Compute CLAP FAD
        if self.fad_clap and len(reference_paths) >= 2 and len(generated_paths) >= 2:
            try:
                logger.info("Computing CLAP FAD...")
                start_time = time.time()
                result = self.fad_clap.compute(reference_paths, generated_paths)
                fad_metrics['fad_clap_score'] = result['fad_score']
                fad_metrics['fad_clap_time'] = time.time() - start_time
                logger.info(f"CLAP FAD: {result['fad_score']:.4f}")
            except Exception as e:
                logger.error(f"CLAP FAD computation failed: {e}")
                fad_metrics['fad_clap_error'] = str(e)

        return fad_metrics

    def evaluate_batch(
        self,
        audio_paths: List[Path],
        text_prompts: Optional[List[str]] = None,
        reference_paths: Optional[List[Path]] = None
    ) -> pd.DataFrame:
        """
        Evaluate a batch of audio files.

        Args:
            audio_paths: List of audio file paths
            text_prompts: Optional list of text prompts (must match audio_paths length)
            reference_paths: Optional reference audio for FAD computation

        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"Evaluating {len(audio_paths)} audio files")

        # Initialize calculators in main process
        if self.tempo is None:
            self._init_calculators()

        # Prepare arguments for parallel processing
        if text_prompts:
            if len(text_prompts) != len(audio_paths):
                logger.warning("Text prompts length doesn't match audio files, ignoring prompts")
                text_prompts = None

        # Process files in parallel with progress bar
        results = []
        with Pool(processes=self.n_workers, initializer=self._init_worker) as pool:
            if text_prompts:
                tasks = [(path, prompt) for path, prompt in zip(audio_paths, text_prompts)]
            else:
                tasks = [(path, None) for path in audio_paths]

            # Use imap_unordered for better performance with progress bar
            for result in tqdm(
                pool.imap_unordered(self._process_single_file_wrapper, tasks),
                total=len(tasks),
                desc="Processing audio files"
            ):
                results.append(result)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Compute FAD metrics if reference provided
        if reference_paths:
            logger.info("Computing FAD metrics...")
            fad_metrics = self._compute_fad_metrics(reference_paths, audio_paths)

            # Add FAD metrics as global columns
            for key, val in fad_metrics.items():
                df[key] = val

        # Log summary
        success_rate = df['success'].sum() / len(df) if len(df) > 0 else 0
        logger.info(f"Evaluation complete: {success_rate:.1%} success rate")

        return df

    @staticmethod
    def _init_worker():
        """Initialize worker process (if needed for multiprocessing)."""
        pass

    @staticmethod
    def _process_single_file_wrapper(args: Tuple[Path, Optional[str]]) -> Dict:
        """
        Wrapper for processing single file (for multiprocessing).

        Args:
            args: Tuple of (audio_path, text_prompt)

        Returns:
            Dictionary of metrics
        """
        audio_path, text_prompt = args

        # Create temporary evaluator for this worker
        evaluator = AudioEvaluator(n_workers=1)
        evaluator._init_calculators()

        return evaluator._compute_single_file_metrics(audio_path, text_prompt)

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
        audio_paths: List[Path],
        output_path: Path,
        text_prompts: Optional[List[str]] = None,
        reference_paths: Optional[List[Path]] = None,
        force_recompute: bool = False
    ) -> pd.DataFrame:
        """
        Evaluate audio files incrementally, skipping already processed files.

        Args:
            audio_paths: List of audio file paths
            output_path: Path to save results
            text_prompts: Optional text prompts
            reference_paths: Optional reference audio for FAD
            force_recompute: Force recomputation of all files

        Returns:
            DataFrame with all results (existing + new)
        """
        # Load existing results
        existing_df = None if force_recompute else self.load_existing_results(output_path)

        # Determine which files need processing
        if existing_df is not None and not existing_df.empty:
            existing_files = set(existing_df['file_path'].values)
            new_paths = [p for p in audio_paths if str(p) not in existing_files]
            logger.info(f"Found {len(new_paths)} new files to process (skipping {len(existing_files)})")
        else:
            new_paths = audio_paths
            logger.info(f"Processing all {len(new_paths)} files")

        # Process new files if any
        if new_paths:
            # Filter text prompts for new paths
            if text_prompts:
                path_to_prompt = {str(p): t for p, t in zip(audio_paths, text_prompts)}
                new_prompts = [path_to_prompt[str(p)] for p in new_paths]
            else:
                new_prompts = None

            new_df = self.evaluate_batch(new_paths, new_prompts, reference_paths)

            # Merge with existing results
            if existing_df is not None and not existing_df.empty:
                df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                df = new_df
        else:
            df = existing_df
            logger.info("No new files to process")

        return df


def load_audio_metadata(metadata_path: Path) -> Optional[pd.DataFrame]:
    """
    Load metadata file containing text prompts and other info.

    Args:
        metadata_path: Path to metadata CSV/JSON file

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
        description="Evaluate audio generations with comprehensive metrics"
    )
    parser.add_argument(
        '--wav_dir',
        type=Path,
        default=Path('runs/artifacts/wav'),
        help='Directory containing WAV files to evaluate'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('runs/logs/audio_metrics.parquet'),
        help='Output path for metrics (parquet format)'
    )
    parser.add_argument(
        '--reference_dir',
        type=Path,
        default=None,
        help='Directory containing reference audio for FAD computation'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        default=None,
        help='Path to metadata file with text prompts (CSV/JSON/parquet)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.wav',
        help='File pattern for audio files (default: *.wav)'
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
        default=8,
        help='Batch size for processing (default: 8)'
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
    if not args.wav_dir.exists():
        logger.error(f"WAV directory not found: {args.wav_dir}")
        sys.exit(1)

    # Find audio files
    audio_paths = sorted(args.wav_dir.glob(args.pattern))
    if not audio_paths:
        logger.error(f"No audio files found in {args.wav_dir} with pattern {args.pattern}")
        sys.exit(1)

    logger.info(f"Found {len(audio_paths)} audio files")

    # Load metadata if provided
    text_prompts = None
    if args.metadata:
        metadata_df = load_audio_metadata(args.metadata)
        if metadata_df is not None and 'text_prompt' in metadata_df.columns:
            # Match prompts to audio files
            path_to_prompt = dict(zip(metadata_df['file_name'], metadata_df['text_prompt']))
            text_prompts = [path_to_prompt.get(p.name, None) for p in audio_paths]
            logger.info(f"Loaded text prompts for {sum(1 for p in text_prompts if p)} files")

    # Load reference audio if provided
    reference_paths = None
    if args.reference_dir and args.reference_dir.exists():
        reference_paths = sorted(args.reference_dir.glob(args.pattern))
        logger.info(f"Found {len(reference_paths)} reference audio files")

    # Create evaluator
    evaluator = AudioEvaluator(
        device=args.device,
        n_workers=args.workers,
        batch_size=args.batch_size
    )

    # Run evaluation
    start_time = time.time()

    if args.incremental:
        results_df = evaluator.incremental_evaluate(
            audio_paths,
            args.output,
            text_prompts,
            reference_paths,
            force_recompute=args.force
        )
    else:
        results_df = evaluator.evaluate_batch(
            audio_paths,
            text_prompts,
            reference_paths
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

    # Print metric statistics
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        logger.info("\nMetric Statistics:")
        for col in numeric_cols:
            if col not in ['success', 'computation_time']:
                mean_val = results_df[col].mean()
                std_val = results_df[col].std()
                logger.info(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")

    logger.info("="*80)


if __name__ == "__main__":
    main()
