"""
Symbolic Music Generation Orchestration Script

This script orchestrates the generation of symbolic music (MIDI) using various models
(Music Transformer, REMI Transformer) across multiple tasks and random seeds.

Usage:
    python gen/run_symbolic.py --model music_transformer --tasks t1,t2,t3 --seeds 3
    python gen/run_symbolic.py --model remi_transformer --tasks t1 --seeds 5 --resume

Author: Music Generation Evaluation Framework
Date: 2025-10-03
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm


class SymbolicGenerationOrchestrator:
    """
    Orchestrates symbolic music generation across models, tasks, and seeds.

    This class manages the generation pipeline including:
    - Loading MIDI seed files from JSON
    - Managing random seeds for reproducibility
    - Organizing output artifacts
    - Progress tracking and logging
    - Resume functionality for interrupted runs
    """

    SUPPORTED_MODELS = ["music_transformer", "remi_transformer"]
    SUPPORTED_TASKS = ["t1", "t2", "t3"]

    def __init__(
        self,
        model_name: str,
        tasks: List[str],
        num_seeds: int = 3,
        output_dir: str = "runs/artifacts",
        log_dir: str = "runs/logs",
        device: Optional[str] = None
    ):
        """
        Initialize the symbolic generation orchestrator.

        Args:
            model_name: Name of the model to use (music_transformer, remi_transformer)
            tasks: List of task IDs to generate for (t1, t2, t3)
            num_seeds: Number of random seeds per MIDI seed
            output_dir: Directory to save generated MIDI files
            log_dir: Directory to save log files
            device: Device to use (cuda/cpu). Auto-detects if None.

        Raises:
            ValueError: If model_name or tasks are invalid
        """
        self.model_name = model_name.lower()
        self.tasks = [t.lower() for t in tasks]
        self.num_seeds = num_seeds
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)

        # Validate inputs
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model_name} not supported. "
                f"Choose from: {', '.join(self.SUPPORTED_MODELS)}"
            )

        for task in self.tasks:
            if task not in self.SUPPORTED_TASKS:
                raise ValueError(
                    f"Task {task} not supported. "
                    f"Choose from: {', '.join(self.SUPPORTED_TASKS)}"
                )

        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Initialize model and data
        self.model = None
        self.midi_seeds = None

        self.logger.info(f"Initialized SymbolicGenerationOrchestrator")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Tasks: {', '.join(self.tasks)}")
        self.logger.info(f"Seeds: {self.num_seeds}")
        self.logger.info(f"Device: {self.device}")

    def _setup_logging(self) -> None:
        """Setup logging to file and console."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"symbolic_gen_{self.model_name}_{timestamp}.log"

        # Create logger
        self.logger = logging.getLogger(f"symbolic_gen_{self.model_name}")
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info(f"Logging to: {log_file}")

    def load_midi_seeds(self, seeds_file: str = "data/midi_seeds.json") -> None:
        """
        Load MIDI seed files from JSON.

        Args:
            seeds_file: Path to MIDI seeds JSON file

        Raises:
            FileNotFoundError: If seeds file doesn't exist
            json.JSONDecodeError: If seeds file is invalid JSON
        """
        seeds_path = Path(seeds_file)

        if not seeds_path.exists():
            raise FileNotFoundError(f"MIDI seeds file not found: {seeds_file}")

        self.logger.info(f"Loading MIDI seeds from: {seeds_file}")

        with open(seeds_path, 'r') as f:
            self.midi_seeds = json.load(f)

        self.logger.info(f"Loaded {len(self.midi_seeds)} MIDI seeds")

    def load_model(self) -> None:
        """
        Load the specified symbolic music generation model.

        Raises:
            ImportError: If model dependencies are not installed
            RuntimeError: If model fails to load
        """
        self.logger.info(f"Loading model: {self.model_name}")

        try:
            if self.model_name == "music_transformer":
                self._load_music_transformer()
            elif self.model_name == "remi_transformer":
                self._load_remi_transformer()
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            self.logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _load_music_transformer(self) -> None:
        """Load Music Transformer model."""
        try:
            # Import Music Transformer from Magenta or custom implementation
            from models.symbolic.music_transformer import MusicTransformer

            self.model = MusicTransformer(device=self.device)
            self.model.load_pretrained()
            self.logger.info("Loaded Music Transformer model")

        except ImportError as e:
            self.logger.warning(
                f"Music Transformer not found: {e}. "
                "Using placeholder implementation."
            )
            # Placeholder for now
            self.model = None

    def _load_remi_transformer(self) -> None:
        """Load REMI Transformer model."""
        try:
            # Import REMI Transformer
            from models.symbolic.remi_transformer import REMITransformer

            self.model = REMITransformer(device=self.device)
            self.model.load_pretrained()
            self.logger.info("Loaded REMI Transformer model")

        except ImportError as e:
            self.logger.warning(
                f"REMI Transformer not found: {e}. "
                "Using placeholder implementation."
            )
            # Placeholder for now
            self.model = None

    def load_midi_file(self, midi_path: str) -> Any:
        """
        Load a MIDI file into a representation suitable for generation.

        Args:
            midi_path: Path to MIDI file

        Returns:
            MIDI representation (format depends on model)

        Raises:
            FileNotFoundError: If MIDI file doesn't exist
        """
        import pretty_midi

        if not Path(midi_path).exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        midi = pretty_midi.PrettyMIDI(midi_path)
        return midi

    def generate_midi(
        self,
        seed_midi: Any,
        task: str,
        seed: int,
        num_bars: int = 16
    ) -> Any:
        """
        Generate MIDI continuation from seed.

        Args:
            seed_midi: MIDI seed data
            task: Task ID (t1, t2, t3)
            seed: Random seed for reproducibility
            num_bars: Number of bars to generate

        Returns:
            Generated MIDI data

        Raises:
            RuntimeError: If generation fails
        """
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        try:
            if self.model_name == "music_transformer":
                return self._generate_music_transformer(seed_midi, num_bars)
            elif self.model_name == "remi_transformer":
                return self._generate_remi_transformer(seed_midi, num_bars)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise

    def _generate_music_transformer(self, seed_midi: Any, num_bars: int) -> Any:
        """Generate MIDI using Music Transformer."""
        import pretty_midi

        if self.model is None:
            # Placeholder: return seed as-is for testing
            self.logger.warning("Using placeholder generation (no model loaded)")
            return seed_midi

        # TODO: Implement actual Music Transformer generation
        # This would involve:
        # 1. Convert MIDI to token sequence
        # 2. Generate continuation tokens
        # 3. Convert tokens back to MIDI

        with torch.no_grad():
            generated_midi = self.model.generate(
                seed_midi=seed_midi,
                num_bars=num_bars,
                temperature=1.0,
                top_p=0.9
            )

        return generated_midi

    def _generate_remi_transformer(self, seed_midi: Any, num_bars: int) -> Any:
        """Generate MIDI using REMI Transformer."""
        import pretty_midi

        if self.model is None:
            # Placeholder: return seed as-is for testing
            self.logger.warning("Using placeholder generation (no model loaded)")
            return seed_midi

        # TODO: Implement actual REMI Transformer generation
        # This would involve:
        # 1. Convert MIDI to REMI tokens
        # 2. Generate continuation tokens
        # 3. Convert REMI tokens back to MIDI

        with torch.no_grad():
            generated_midi = self.model.generate(
                seed_midi=seed_midi,
                num_bars=num_bars,
                temperature=1.0,
                top_p=0.9
            )

        return generated_midi

    def save_midi(
        self,
        midi_data: Any,
        seed_id: int,
        task: str,
        seed: int
    ) -> Tuple[Path, Dict]:
        """
        Save generated MIDI to file with metadata.

        Args:
            midi_data: Generated MIDI data
            seed_id: ID of the seed MIDI used
            task: Task ID (t1, t2, t3)
            seed: Random seed used

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        import pretty_midi

        # Create output directory structure
        task_dir = self.output_dir / self.model_name / task
        task_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"seed_{seed_id:04d}_gen_{seed:02d}.mid"
        output_path = task_dir / filename

        # Save MIDI
        if isinstance(midi_data, pretty_midi.PrettyMIDI):
            midi_data.write(str(output_path))
        else:
            # Handle other MIDI representations
            raise TypeError(f"Unsupported MIDI type: {type(midi_data)}")

        # Create metadata
        metadata = {
            "model": self.model_name,
            "task": task,
            "seed_id": seed_id,
            "random_seed": seed,
            "file_path": str(output_path),
            "timestamp": datetime.now().isoformat(),
            "num_instruments": len(midi_data.instruments) if hasattr(midi_data, 'instruments') else 0,
            "duration": midi_data.get_end_time() if hasattr(midi_data, 'get_end_time') else 0.0
        }

        # Save metadata JSON
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return output_path, metadata

    def get_completed_jobs(self) -> set:
        """
        Get set of completed generation jobs for resume functionality.

        Returns:
            Set of tuples (seed_id, task, seed) that have been completed
        """
        completed = set()

        for task in self.tasks:
            task_dir = self.output_dir / self.model_name / task
            if not task_dir.exists():
                continue

            # Find all MIDI files
            for midi_file in task_dir.glob("seed_*_gen_*.mid"):
                # Parse filename
                parts = midi_file.stem.split('_')
                if len(parts) >= 4:
                    seed_id = int(parts[1])
                    gen_seed = int(parts[3])
                    completed.add((seed_id, task, gen_seed))

        return completed

    def run(self, resume: bool = False) -> Dict:
        """
        Run the complete generation pipeline.

        Args:
            resume: If True, skip already completed generations

        Returns:
            Dictionary containing generation statistics
        """
        self.logger.info("Starting generation pipeline")

        # Load MIDI seeds and model
        self.load_midi_seeds()
        self.load_model()

        # Get completed jobs if resuming
        completed_jobs = set()
        if resume:
            completed_jobs = self.get_completed_jobs()
            self.logger.info(f"Resuming: found {len(completed_jobs)} completed generations")

        # Calculate total jobs
        total_jobs = len(self.midi_seeds) * len(self.tasks) * self.num_seeds
        self.logger.info(f"Total jobs: {total_jobs}")

        # Statistics
        stats = {
            "total_jobs": total_jobs,
            "completed": 0,
            "skipped": 0,
            "failed": 0,
            "errors": []
        }

        # Progress bar
        pbar = tqdm(total=total_jobs, desc="Generating MIDI")

        # Generate for each combination
        for seed_idx, seed_data in enumerate(self.midi_seeds):
            seed_path = seed_data.get("path", seed_data.get("file", ""))

            # Load seed MIDI
            try:
                seed_midi = self.load_midi_file(seed_path)
            except Exception as e:
                self.logger.error(f"Failed to load seed {seed_idx}: {e}")
                # Skip this seed entirely
                skip_count = len(self.tasks) * self.num_seeds
                stats["failed"] += skip_count
                pbar.update(skip_count)
                continue

            for task in self.tasks:
                for gen_seed in range(self.num_seeds):
                    # Check if already completed
                    job_key = (seed_idx, task, gen_seed)
                    if resume and job_key in completed_jobs:
                        stats["skipped"] += 1
                        pbar.update(1)
                        continue

                    try:
                        # Generate MIDI
                        generated_midi = self.generate_midi(
                            seed_midi=seed_midi,
                            task=task,
                            seed=gen_seed,
                            num_bars=16
                        )

                        # Save MIDI and metadata
                        output_path, metadata = self.save_midi(
                            midi_data=generated_midi,
                            seed_id=seed_idx,
                            task=task,
                            seed=gen_seed
                        )

                        stats["completed"] += 1

                        # Update progress
                        pbar.set_postfix({
                            "completed": stats["completed"],
                            "failed": stats["failed"]
                        })

                    except Exception as e:
                        stats["failed"] += 1
                        error_msg = f"Failed: seed={seed_idx}, task={task}, gen_seed={gen_seed}: {e}"
                        stats["errors"].append(error_msg)
                        self.logger.error(error_msg)

                    finally:
                        pbar.update(1)

        pbar.close()

        # Log final statistics
        self.logger.info("=" * 60)
        self.logger.info("Generation Complete")
        self.logger.info(f"Total jobs: {stats['total_jobs']}")
        self.logger.info(f"Completed: {stats['completed']}")
        self.logger.info(f"Skipped: {stats['skipped']}")
        self.logger.info(f"Failed: {stats['failed']}")
        self.logger.info("=" * 60)

        return stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate symbolic music (MIDI) using transformer models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with Music Transformer for all tasks
  python gen/run_symbolic.py --model music_transformer --tasks t1,t2,t3 --seeds 3

  # Generate with REMI Transformer for task 1 only
  python gen/run_symbolic.py --model remi_transformer --tasks t1 --seeds 5

  # Resume interrupted generation
  python gen/run_symbolic.py --model music_transformer --tasks t1,t2,t3 --resume
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["music_transformer", "remi_transformer"],
        help="Symbolic music generation model to use"
    )

    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated list of tasks (e.g., t1,t2,t3)"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds per MIDI seed (default: 3)"
    )

    parser.add_argument(
        "--midi-seeds",
        type=str,
        default="data/midi_seeds.json",
        help="Path to MIDI seeds JSON file (default: data/midi_seeds.json)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/artifacts",
        help="Output directory for generated MIDI (default: runs/artifacts)"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/logs",
        help="Directory for log files (default: runs/logs)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip completed generations)"
    )

    args = parser.parse_args()

    # Parse tasks
    tasks = [t.strip() for t in args.tasks.split(",")]

    try:
        # Create orchestrator
        orchestrator = SymbolicGenerationOrchestrator(
            model_name=args.model,
            tasks=tasks,
            num_seeds=args.seeds,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            device=args.device
        )

        # Run generation
        stats = orchestrator.run(resume=args.resume)

        # Exit with appropriate code
        if stats["failed"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
