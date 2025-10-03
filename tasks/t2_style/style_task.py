"""
T2: Style Adherence / Conditioning Task

This module implements the style adherence test for both audio and symbolic models.

For audio models:
    - Generate music from style descriptions (genre, instrumentation, tempo range)
    - Validate that the generated audio matches the specified style

For symbolic models:
    - Continue MIDI in the same genre/style as the seed
    - Validate that the continuation maintains style consistency
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import json
from dataclasses import dataclass, asdict
from enum import Enum
import pretty_midi
import numpy as np


class ModelType(Enum):
    """Model type enumeration."""
    AUDIO = "audio"
    SYMBOLIC = "symbolic"


class StyleCategory(Enum):
    """Musical style categories."""
    CLASSICAL = "classical"
    JAZZ = "jazz"
    POP = "pop"
    ROCK = "rock"
    ELECTRONIC = "electronic"
    HIP_HOP = "hip_hop"
    FOLK = "folk"
    BLUES = "blues"
    OTHER = "other"


@dataclass
class StyleSpec:
    """Specification for musical style."""
    genre: str  # Primary genre (e.g., "jazz", "pop", "classical")
    sub_genre: Optional[str] = None  # Sub-genre (e.g., "bebop", "lo-fi")
    instrumentation: Optional[List[str]] = None  # Expected instruments
    tempo_range: Optional[tuple[int, int]] = None  # (min, max) BPM
    mood: Optional[str] = None  # e.g., "upbeat", "melancholic", "energetic"
    era: Optional[str] = None  # e.g., "80s", "baroque"
    style_tags: Optional[List[str]] = None  # Additional style descriptors


@dataclass
class TaskInput:
    """Input specification for style task."""
    model_type: ModelType
    style_spec: StyleSpec

    # For audio models
    prompt: Optional[str] = None
    duration: Optional[float] = None  # seconds

    # For symbolic models (continuation task)
    seed_midi_path: Optional[str] = None
    continuation_bars: Optional[int] = None

    # Reference for style comparison
    reference_path: Optional[str] = None  # Path to reference audio/MIDI


@dataclass
class TaskResult:
    """Result from style task execution."""
    success: bool
    model_name: str
    task_name: str = "T2_style"

    # Generation outputs
    output_path: Optional[str] = None
    generation_time: Optional[float] = None

    # Metadata
    input_spec: Optional[Dict[str, Any]] = None
    generation_params: Optional[Dict[str, Any]] = None

    # Style validation results
    style_adherence_score: Optional[float] = None  # 0-1, higher is better
    genre_detected: Optional[str] = None
    tempo_detected: Optional[float] = None
    tempo_in_range: Optional[bool] = None
    instrumentation_match: Optional[Dict[str, Any]] = None

    # Similarity to reference (if provided)
    reference_similarity: Optional[float] = None  # 0-1, higher is better

    # Error information
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = asdict(self)
        # Handle enum serialization
        if self.input_spec and 'model_type' in self.input_spec:
            if isinstance(self.input_spec['model_type'], ModelType):
                result['input_spec']['model_type'] = self.input_spec['model_type'].value
        return result


class StyleTaskExecutor:
    """
    Executor for style adherence/conditioning task.

    This class handles both audio and symbolic model testing for style consistency.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the task executor.

        Args:
            output_dir: Directory to save outputs (default: runs/artifacts)
        """
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent.parent / "runs" / "artifacts"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        model_wrapper: Any,
        task_input: TaskInput,
        seed: Optional[int] = None,
        **generation_kwargs
    ) -> TaskResult:
        """
        Execute the style adherence task.

        Args:
            model_wrapper: Model wrapper instance (MusicGenWrapper, etc.)
            task_input: Task input specification
            seed: Random seed for reproducibility
            **generation_kwargs: Additional generation parameters

        Returns:
            TaskResult with generation outputs and validation results
        """
        import time

        # Validate input
        validation_error = self._validate_input(task_input)
        if validation_error:
            return TaskResult(
                success=False,
                model_name=getattr(model_wrapper, '__class__.__name__', 'unknown'),
                error=validation_error,
                input_spec=self._serialize_input(task_input)
            )

        # Execute based on model type
        start_time = time.time()

        try:
            if task_input.model_type == ModelType.AUDIO:
                result = self._execute_audio(
                    model_wrapper,
                    task_input,
                    seed=seed,
                    **generation_kwargs
                )
            else:  # SYMBOLIC
                result = self._execute_symbolic(
                    model_wrapper,
                    task_input,
                    seed=seed,
                    **generation_kwargs
                )

            result.generation_time = time.time() - start_time
            result.input_spec = self._serialize_input(task_input)

            return result

        except Exception as e:
            return TaskResult(
                success=False,
                model_name=getattr(model_wrapper, '__class__.__name__', 'unknown'),
                error=f"Execution error: {str(e)}",
                input_spec=self._serialize_input(task_input),
                generation_time=time.time() - start_time
            )

    def _validate_input(self, task_input: TaskInput) -> Optional[str]:
        """
        Validate task input specification.

        Returns:
            Error message if validation fails, None otherwise
        """
        if task_input.model_type == ModelType.AUDIO:
            if not task_input.prompt:
                return "Audio models require a text prompt"
            if not task_input.duration or task_input.duration <= 0:
                return "Audio models require positive duration"
        else:  # SYMBOLIC
            if not task_input.seed_midi_path:
                return "Symbolic models require seed MIDI path"
            if not Path(task_input.seed_midi_path).exists():
                return f"Seed MIDI file not found: {task_input.seed_midi_path}"
            if not task_input.continuation_bars or task_input.continuation_bars <= 0:
                return "Symbolic models require positive continuation_bars"

        # Check reference if provided
        if task_input.reference_path and not Path(task_input.reference_path).exists():
            return f"Reference file not found: {task_input.reference_path}"

        return None

    def _serialize_input(self, task_input: TaskInput) -> Dict[str, Any]:
        """Serialize task input for storage."""
        input_dict = asdict(task_input)
        input_dict['model_type'] = task_input.model_type.value
        return input_dict

    def _execute_audio(
        self,
        model_wrapper: Any,
        task_input: TaskInput,
        seed: Optional[int] = None,
        **generation_kwargs
    ) -> TaskResult:
        """
        Execute style task for audio models.

        Audio models receive style information through text prompts.
        """
        import time

        model_name = model_wrapper.__class__.__name__

        # Prepare output path
        output_dir = self.output_dir / "wav" / "t2_style"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        output_path = output_dir / f"{model_name}_style_{timestamp}.wav"

        # Generate audio
        generation_params = {
            'prompt': task_input.prompt,
            'duration': task_input.duration,
            'seed': seed,
            **generation_kwargs
        }

        metadata = model_wrapper.generate_and_save(
            output_path=str(output_path),
            **generation_params
        )

        # Perform basic tempo detection if tempo range specified
        tempo_in_range = None
        tempo_detected = None

        if task_input.style_spec.tempo_range:
            try:
                tempo_detected = self._estimate_audio_tempo(str(output_path))
                min_tempo, max_tempo = task_input.style_spec.tempo_range
                tempo_in_range = min_tempo <= tempo_detected <= max_tempo
            except Exception as e:
                print(f"Warning: Could not detect tempo: {e}")

        result = TaskResult(
            success=True,
            model_name=model_name,
            output_path=str(output_path),
            generation_params=generation_params,
            tempo_detected=tempo_detected,
            tempo_in_range=tempo_in_range,
            style_adherence_score=None,  # To be filled by metrics
            genre_detected=None  # To be filled by metrics
        )

        return result

    def _execute_symbolic(
        self,
        model_wrapper: Any,
        task_input: TaskInput,
        seed: Optional[int] = None,
        **generation_kwargs
    ) -> TaskResult:
        """
        Execute style task for symbolic models.

        Symbolic models continue from a seed MIDI in the same style.
        """
        import time

        model_name = model_wrapper.__class__.__name__

        # Load seed MIDI to extract style properties
        seed_midi = pretty_midi.PrettyMIDI(task_input.seed_midi_path)

        # Extract seed style features
        seed_features = self._extract_midi_style_features(seed_midi)

        # Prepare output path
        output_dir = self.output_dir / "midi" / "t2_style"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        output_path = output_dir / f"{model_name}_style_{timestamp}.mid"

        # Calculate total length needed
        events_per_bar = 16
        continuation_events = task_input.continuation_bars * events_per_bar

        # Generate continuation
        generation_params = {
            'primer_midi': task_input.seed_midi_path,
            'total_length': continuation_events,
            'seed': seed,
            **generation_kwargs
        }

        metadata = model_wrapper.generate_and_save(
            output_path=str(output_path),
            **generation_params
        )

        # Load generated MIDI to validate style consistency
        generated_midi = pretty_midi.PrettyMIDI(str(output_path))
        generated_features = self._extract_midi_style_features(generated_midi)

        # Compute style consistency score
        style_adherence_score = self._compute_style_similarity(
            seed_features,
            generated_features
        )

        # Check tempo if specified
        tempo_in_range = None
        if task_input.style_spec.tempo_range:
            min_tempo, max_tempo = task_input.style_spec.tempo_range
            tempo_detected = generated_features.get('tempo', 120.0)
            tempo_in_range = min_tempo <= tempo_detected <= max_tempo

        result = TaskResult(
            success=True,
            model_name=model_name,
            output_path=str(output_path),
            generation_params=generation_params,
            style_adherence_score=style_adherence_score,
            tempo_detected=generated_features.get('tempo'),
            tempo_in_range=tempo_in_range,
            instrumentation_match={
                'seed_instruments': seed_features.get('instruments', []),
                'generated_instruments': generated_features.get('instruments', [])
            }
        )

        return result

    def _extract_midi_style_features(self, midi: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
        """
        Extract style-relevant features from MIDI.

        Args:
            midi: PrettyMIDI object

        Returns:
            Dictionary of style features
        """
        features = {}

        # Tempo
        if midi.get_tempo_changes()[0].size > 0:
            tempos = midi.get_tempo_changes()[1]
            features['tempo'] = float(np.mean(tempos))
        else:
            features['tempo'] = 120.0

        # Time signature
        if midi.time_signature_changes:
            ts = midi.time_signature_changes[0]
            features['time_signature'] = f"{ts.numerator}/{ts.denominator}"
        else:
            features['time_signature'] = "4/4"

        # Instruments
        features['instruments'] = [inst.program for inst in midi.instruments if not inst.is_drum]
        features['has_drums'] = any(inst.is_drum for inst in midi.instruments)

        # Pitch statistics
        all_pitches = []
        all_velocities = []
        all_durations = []

        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    all_pitches.append(note.pitch)
                    all_velocities.append(note.velocity)
                    all_durations.append(note.end - note.start)

        if all_pitches:
            features['pitch_mean'] = float(np.mean(all_pitches))
            features['pitch_std'] = float(np.std(all_pitches))
            features['pitch_range'] = int(max(all_pitches) - min(all_pitches))

            # Pitch class histogram (for harmony/tonality)
            pitch_classes = [p % 12 for p in all_pitches]
            pc_hist, _ = np.histogram(pitch_classes, bins=12, range=(0, 12), density=True)
            features['pitch_class_histogram'] = pc_hist.tolist()

        if all_velocities:
            features['velocity_mean'] = float(np.mean(all_velocities))
            features['velocity_std'] = float(np.std(all_velocities))

        if all_durations:
            features['duration_mean'] = float(np.mean(all_durations))
            features['duration_std'] = float(np.std(all_durations))

        # Note density (notes per second)
        if midi.get_end_time() > 0:
            total_notes = sum(len(inst.notes) for inst in midi.instruments if not inst.is_drum)
            features['note_density'] = total_notes / midi.get_end_time()
        else:
            features['note_density'] = 0.0

        return features

    def _compute_style_similarity(
        self,
        seed_features: Dict[str, Any],
        generated_features: Dict[str, Any]
    ) -> float:
        """
        Compute style similarity between seed and generated music.

        Args:
            seed_features: Features extracted from seed
            generated_features: Features extracted from generated music

        Returns:
            Similarity score (0-1, higher is better)
        """
        scores = []

        # Tempo similarity (normalize by percentage difference)
        if 'tempo' in seed_features and 'tempo' in generated_features:
            tempo_diff = abs(seed_features['tempo'] - generated_features['tempo'])
            tempo_score = max(0.0, 1.0 - (tempo_diff / 50.0))  # 50 BPM tolerance
            scores.append(tempo_score)

        # Pitch range similarity
        if 'pitch_range' in seed_features and 'pitch_range' in generated_features:
            range_diff = abs(seed_features['pitch_range'] - generated_features['pitch_range'])
            range_score = max(0.0, 1.0 - (range_diff / 24.0))  # 2 octaves tolerance
            scores.append(range_score)

        # Pitch class histogram similarity (cosine similarity)
        if 'pitch_class_histogram' in seed_features and 'pitch_class_histogram' in generated_features:
            seed_pc = np.array(seed_features['pitch_class_histogram'])
            gen_pc = np.array(generated_features['pitch_class_histogram'])

            # Cosine similarity
            if np.linalg.norm(seed_pc) > 0 and np.linalg.norm(gen_pc) > 0:
                pc_similarity = np.dot(seed_pc, gen_pc) / (np.linalg.norm(seed_pc) * np.linalg.norm(gen_pc))
                scores.append(float(pc_similarity))

        # Note density similarity
        if 'note_density' in seed_features and 'note_density' in generated_features:
            density_ratio = min(
                seed_features['note_density'] / max(generated_features['note_density'], 0.01),
                generated_features['note_density'] / max(seed_features['note_density'], 0.01)
            )
            scores.append(density_ratio)

        # Velocity similarity
        if 'velocity_mean' in seed_features and 'velocity_mean' in generated_features:
            vel_diff = abs(seed_features['velocity_mean'] - generated_features['velocity_mean'])
            vel_score = max(0.0, 1.0 - (vel_diff / 64.0))  # Half velocity range tolerance
            scores.append(vel_score)

        # Overall score is mean of individual scores
        if scores:
            return float(np.mean(scores))
        else:
            return 0.5  # Default if no comparable features

    def _estimate_audio_tempo(self, audio_path: str) -> float:
        """
        Estimate tempo from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Estimated tempo in BPM
        """
        try:
            import librosa

            y, sr = librosa.load(audio_path)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            return float(tempo)

        except ImportError:
            print("Warning: librosa not installed, cannot estimate tempo")
            return 120.0
        except Exception as e:
            print(f"Warning: Error estimating tempo: {e}")
            return 120.0

    def save_result(self, result: TaskResult, output_path: Union[str, Path]) -> None:
        """
        Save task result to JSON file.

        Args:
            result: Task result to save
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)


def main():
    """Example usage of style task executor."""
    from pathlib import Path

    # Example 1: Audio model with style specification
    print("Example 1: Audio model style task")

    audio_style = StyleSpec(
        genre="jazz",
        sub_genre="bebop",
        instrumentation=["piano", "bass", "drums"],
        tempo_range=(160, 200),
        mood="energetic",
        era="1940s"
    )

    audio_input = TaskInput(
        model_type=ModelType.AUDIO,
        style_spec=audio_style,
        prompt="fast bebop jazz, piano trio, 180 BPM, 1940s style, energetic improvisation",
        duration=30.0
    )

    print(f"Genre: {audio_style.genre} ({audio_style.sub_genre})")
    print(f"Tempo range: {audio_style.tempo_range} BPM")
    print(f"Prompt: {audio_input.prompt}")

    # Example 2: Symbolic model with style continuation
    print("\nExample 2: Symbolic model style task")

    symbolic_style = StyleSpec(
        genre="classical",
        sub_genre="baroque",
        tempo_range=(100, 130),
        mood="formal",
        era="baroque"
    )

    # Note: This would require an actual seed MIDI file
    symbolic_input = TaskInput(
        model_type=ModelType.SYMBOLIC,
        style_spec=symbolic_style,
        seed_midi_path="/path/to/baroque_seed.mid",
        continuation_bars=16
    )

    print(f"Genre: {symbolic_style.genre} ({symbolic_style.sub_genre})")
    print(f"Continuation: {symbolic_input.continuation_bars} bars")

    # Example 3: With reference for comparison
    print("\nExample 3: Style task with reference")

    style_with_ref = StyleSpec(
        genre="electronic",
        sub_genre="lo-fi",
        tempo_range=(80, 95)
    )

    input_with_ref = TaskInput(
        model_type=ModelType.AUDIO,
        style_spec=style_with_ref,
        prompt="lo-fi hip-hop, chill beats, 85 BPM, vinyl crackle",
        duration=30.0,
        reference_path="/path/to/reference_lofi.wav"
    )

    print(f"Genre: {style_with_ref.genre}")
    print(f"Reference: {input_with_ref.reference_path}")

    print("\nNote: To run actual generation, provide model wrappers to executor.execute()")


if __name__ == "__main__":
    main()
