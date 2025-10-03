"""
Audio Generation Orchestration Script

This script orchestrates the generation of audio samples using various text-to-audio models
(MusicGen, Stable Audio) across multiple tasks and random seeds.

Usage:
    python gen/run_audio.py --model musicgen --tasks t1,t2,t3 --seeds 3
    python gen/run_audio.py --model stableaudio --tasks t1 --seeds 5 --resume

Author: Music Generation Evaluation Framework
Date: 2025-10-03
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm


class AudioGenerationOrchestrator:
    """
    Orchestrates audio generation across models, tasks, and seeds.

    This class manages the generation pipeline including:
    - Loading prompts from JSON files
    - Managing random seeds for reproducibility
    - Organizing output artifacts
    - Progress tracking and logging
    - Resume functionality for interrupted runs
    """

    SUPPORTED_MODELS = ["musicgen", "stableaudio"]
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
        Initialize the audio generation orchestrator.

        Args:
            model_name: Name of the model to use (musicgen, stableaudio)
            tasks: List of task IDs to generate for (t1, t2, t3)
            num_seeds: Number of random seeds per prompt
            output_dir: Directory to save generated audio files
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

        # Initialize model
        self.model = None
        self.prompts = None

        self.logger.info(f"Initialized AudioGenerationOrchestrator")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Tasks: {', '.join(self.tasks)}")
        self.logger.info(f"Seeds: {self.num_seeds}")
        self.logger.info(f"Device: {self.device}")

    def _setup_logging(self) -> None:
        """Setup logging to file and console."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"audio_gen_{self.model_name}_{timestamp}.log"

        # Create logger
        self.logger = logging.getLogger(f"audio_gen_{self.model_name}")
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

    def load_prompts(self, prompts_file: str = "data/prompts/prompts_text.json") -> None:
        """
        Load text prompts from JSON file.

        Args:
            prompts_file: Path to prompts JSON file

        Raises:
            FileNotFoundError: If prompts file doesn't exist
            json.JSONDecodeError: If prompts file is invalid JSON
        """
        prompts_path = Path(prompts_file)

        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

        self.logger.info(f"Loading prompts from: {prompts_file}")

        with open(prompts_path, 'r') as f:
            self.prompts = json.load(f)

        self.logger.info(f"Loaded {len(self.prompts)} prompts")

    def load_model(self) -> None:
        """
        Load the specified audio generation model.

        Raises:
            ImportError: If model dependencies are not installed
            RuntimeError: If model fails to load
        """
        self.logger.info(f"Loading model: {self.model_name}")

        try:
            if self.model_name == "musicgen":
                self._load_musicgen()
            elif self.model_name == "stableaudio":
                self._load_stableaudio()
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            self.logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _load_musicgen(self) -> None:
        """Load MusicGen model."""
        try:
            from audiocraft.models import MusicGen
            self.model = MusicGen.get_pretrained('facebook/musicgen-medium', device=self.device)
            self.logger.info("Loaded MusicGen medium model")
        except ImportError:
            raise ImportError(
                "audiocraft not installed. Install with: pip install audiocraft"
            )

    def _load_stableaudio(self) -> None:
        """Load Stable Audio model."""
        try:
            from stable_audio_tools import get_pretrained_model
            from stable_audio_tools.inference.generation import generate_diffusion_cond

            self.model, self.model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
            self.model = self.model.to(self.device)
            self.generate_fn = generate_diffusion_cond
            self.logger.info("Loaded Stable Audio Open 1.0 model")
        except ImportError:
            raise ImportError(
                "stable-audio-tools not installed. Install with: pip install stable-audio-tools"
            )

    def generate_audio(
        self,
        prompt: str,
        task: str,
        seed: int,
        duration: float = 30.0
    ) -> np.ndarray:
        """
        Generate audio from text prompt.

        Args:
            prompt: Text description of audio to generate
            task: Task ID (t1, t2, t3)
            seed: Random seed for reproducibility
            duration: Duration of audio in seconds

        Returns:
            Generated audio as numpy array

        Raises:
            RuntimeError: If generation fails
        """
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        try:
            if self.model_name == "musicgen":
                return self._generate_musicgen(prompt, duration)
            elif self.model_name == "stableaudio":
                return self._generate_stableaudio(prompt, duration)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

        except Exception as e:
            self.logger.error(f"Generation failed for prompt '{prompt[:50]}...': {e}")
            raise

    def _generate_musicgen(self, prompt: str, duration: float) -> np.ndarray:
        """Generate audio using MusicGen."""
        self.model.set_generation_params(duration=duration)

        with torch.no_grad():
            wav = self.model.generate([prompt])

        # Convert to numpy and squeeze
        audio = wav.cpu().numpy().squeeze()
        return audio

    def _generate_stableaudio(self, prompt: str, duration: float) -> np.ndarray:
        """Generate audio using Stable Audio."""
        # Prepare conditioning
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": duration
        }]

        with torch.no_grad():
            output = self.generate_fn(
                self.model,
                steps=100,
                cfg_scale=7,
                conditioning=conditioning,
                sample_rate=self.model_config["sample_rate"],
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=self.device
            )

        # Convert to numpy
        audio = output.cpu().numpy().squeeze()
        return audio

    def save_audio(
        self,
        audio: np.ndarray,
        prompt_id: int,
        task: str,
        seed: int,
        sample_rate: int = 32000
    ) -> Tuple[Path, Dict]:
        """
        Save generated audio to WAV file with metadata.

        Args:
            audio: Audio data as numpy array
            prompt_id: ID of the prompt used
            task: Task ID (t1, t2, t3)
            seed: Random seed used
            sample_rate: Sample rate of audio

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        import soundfile as sf

        # Create output directory structure
        task_dir = self.output_dir / self.model_name / task
        task_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"prompt_{prompt_id:04d}_seed_{seed:02d}.wav"
        output_path = task_dir / filename

        # Save audio
        sf.write(output_path, audio, sample_rate)

        # Create metadata
        metadata = {
            "model": self.model_name,
            "task": task,
            "prompt_id": prompt_id,
            "seed": seed,
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate,
            "file_path": str(output_path),
            "timestamp": datetime.now().isoformat()
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
            Set of tuples (prompt_id, task, seed) that have been completed
        """
        completed = set()

        for task in self.tasks:
            task_dir = self.output_dir / self.model_name / task
            if not task_dir.exists():
                continue

            # Find all WAV files
            for wav_file in task_dir.glob("prompt_*_seed_*.wav"):
                # Parse filename
                parts = wav_file.stem.split('_')
                if len(parts) >= 4:
                    prompt_id = int(parts[1])
                    seed = int(parts[3])
                    completed.add((prompt_id, task, seed))

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

        # Load prompts and model
        self.load_prompts()
        self.load_model()

        # Get completed jobs if resuming
        completed_jobs = set()
        if resume:
            completed_jobs = self.get_completed_jobs()
            self.logger.info(f"Resuming: found {len(completed_jobs)} completed generations")

        # Calculate total jobs
        total_jobs = len(self.prompts) * len(self.tasks) * self.num_seeds
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
        pbar = tqdm(total=total_jobs, desc="Generating audio")

        # Generate for each combination
        for prompt_idx, prompt_data in enumerate(self.prompts):
            prompt_text = prompt_data.get("text", prompt_data.get("prompt", ""))

            for task in self.tasks:
                for seed_idx in range(self.num_seeds):
                    # Check if already completed
                    job_key = (prompt_idx, task, seed_idx)
                    if resume and job_key in completed_jobs:
                        stats["skipped"] += 1
                        pbar.update(1)
                        continue

                    try:
                        # Generate audio
                        audio = self.generate_audio(
                            prompt=prompt_text,
                            task=task,
                            seed=seed_idx,
                            duration=30.0
                        )

                        # Save audio and metadata
                        output_path, metadata = self.save_audio(
                            audio=audio,
                            prompt_id=prompt_idx,
                            task=task,
                            seed=seed_idx
                        )

                        stats["completed"] += 1

                        # Update progress
                        pbar.set_postfix({
                            "completed": stats["completed"],
                            "failed": stats["failed"]
                        })

                    except Exception as e:
                        stats["failed"] += 1
                        error_msg = f"Failed: prompt={prompt_idx}, task={task}, seed={seed_idx}: {e}"
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
        description="Generate audio samples using text-to-audio models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with MusicGen for all tasks
  python gen/run_audio.py --model musicgen --tasks t1,t2,t3 --seeds 3

  # Generate with Stable Audio for task 1 only
  python gen/run_audio.py --model stableaudio --tasks t1 --seeds 5

  # Resume interrupted generation
  python gen/run_audio.py --model musicgen --tasks t1,t2,t3 --resume
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["musicgen", "stableaudio"],
        help="Audio generation model to use"
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
        help="Number of random seeds per prompt (default: 3)"
    )

    parser.add_argument(
        "--prompts",
        type=str,
        default="data/prompts/prompts_text.json",
        help="Path to prompts JSON file (default: data/prompts/prompts_text.json)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/artifacts",
        help="Output directory for generated audio (default: runs/artifacts)"
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
        orchestrator = AudioGenerationOrchestrator(
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
