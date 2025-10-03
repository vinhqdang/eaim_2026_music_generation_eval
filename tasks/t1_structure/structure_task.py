"""
T1: Structure-Aware Continuation Task

This module implements the structure-aware continuation test for both audio and symbolic models.

For audio models:
    - Generate music from textual form specifications (e.g., "AABA, 8 bars each")
    - Validate that the generated audio exhibits the requested structure

For symbolic models:
    - Continue a seed MIDI for N bars while preserving key, tempo, and meter
    - Validate that the continuation maintains structural consistency
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


@dataclass
class StructureSpec:
    """Specification for musical structure."""
    form: str  # e.g., "AABA", "verse-chorus-verse-chorus"
    section_lengths: List[int]  # Length of each section in bars
    tempo: Optional[int] = None  # Tempo in BPM (for symbolic)
    key: Optional[str] = None  # Musical key (for symbolic)
    time_signature: Optional[str] = None  # e.g., "4/4"


@dataclass
class TaskInput:
    """Input specification for structure task."""
    model_type: ModelType
    structure_spec: StructureSpec

    # For audio models
    prompt: Optional[str] = None
    duration: Optional[float] = None  # seconds

    # For symbolic models
    seed_midi_path: Optional[str] = None
    continuation_bars: Optional[int] = None


@dataclass
class TaskResult:
    """Result from structure task execution."""
    success: bool
    model_name: str
    task_name: str = "T1_structure"

    # Generation outputs
    output_path: Optional[str] = None
    generation_time: Optional[float] = None

    # Metadata
    input_spec: Optional[Dict[str, Any]] = None
    generation_params: Optional[Dict[str, Any]] = None

    # Validation results
    structure_detected: Optional[Dict[str, Any]] = None
    key_preserved: Optional[bool] = None
    tempo_preserved: Optional[bool] = None
    meter_preserved: Optional[bool] = None

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


class StructureTaskExecutor:
    """
    Executor for structure-aware continuation task.

    This class handles both audio and symbolic model testing for structure awareness.
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
        Execute the structure-aware continuation task.

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
        Execute structure task for audio models.

        Audio models receive structure information through text prompts.
        """
        model_name = model_wrapper.__class__.__name__

        # Prepare output path
        output_dir = self.output_dir / "wav" / "t1_structure"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        output_path = output_dir / f"{model_name}_structure_{timestamp}.wav"

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

        # For now, we return success without validation
        # Actual structure validation would be done by metrics modules
        result = TaskResult(
            success=True,
            model_name=model_name,
            output_path=str(output_path),
            generation_params=generation_params,
            structure_detected=None  # To be filled by metrics
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
        Execute structure task for symbolic models.

        Symbolic models continue from a seed MIDI while preserving key, tempo, and meter.
        """
        model_name = model_wrapper.__class__.__name__

        # Load seed MIDI to extract structural properties
        seed_midi = pretty_midi.PrettyMIDI(task_input.seed_midi_path)

        # Extract seed properties
        seed_properties = self._extract_midi_properties(seed_midi)

        # Prepare output path
        output_dir = self.output_dir / "midi" / "t1_structure"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        output_path = output_dir / f"{model_name}_structure_{timestamp}.mid"

        # Calculate total length needed
        # Estimate events per bar (rough heuristic: 16 events per bar for 4/4 time)
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

        # Load generated MIDI to validate properties
        generated_midi = pretty_midi.PrettyMIDI(str(output_path))
        generated_properties = self._extract_midi_properties(generated_midi)

        # Check if properties are preserved
        key_preserved = self._check_key_preservation(
            seed_properties.get('key'),
            generated_properties.get('key')
        )

        tempo_preserved = self._check_tempo_preservation(
            seed_properties.get('tempo'),
            generated_properties.get('tempo')
        )

        meter_preserved = self._check_meter_preservation(
            seed_properties.get('time_signature'),
            generated_properties.get('time_signature')
        )

        result = TaskResult(
            success=True,
            model_name=model_name,
            output_path=str(output_path),
            generation_params=generation_params,
            key_preserved=key_preserved,
            tempo_preserved=tempo_preserved,
            meter_preserved=meter_preserved,
            structure_detected={
                'seed_properties': seed_properties,
                'generated_properties': generated_properties
            }
        )

        return result

    def _extract_midi_properties(self, midi: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
        """
        Extract structural properties from MIDI.

        Args:
            midi: PrettyMIDI object

        Returns:
            Dictionary of properties (key, tempo, time_signature)
        """
        properties = {}

        # Estimate tempo
        if midi.get_tempo_changes()[0].size > 0:
            tempos = midi.get_tempo_changes()[1]
            properties['tempo'] = float(np.mean(tempos))
        else:
            properties['tempo'] = 120.0  # Default

        # Get time signature
        if midi.time_signature_changes:
            ts = midi.time_signature_changes[0]
            properties['time_signature'] = f"{ts.numerator}/{ts.denominator}"
        else:
            properties['time_signature'] = "4/4"  # Default

        # Estimate key using pitch class distribution
        # This is a simple heuristic - production code would use better key estimation
        if midi.instruments:
            all_pitches = []
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    all_pitches.extend([note.pitch for note in instrument.notes])

            if all_pitches:
                pitch_classes = [p % 12 for p in all_pitches]
                most_common_pc = max(set(pitch_classes), key=pitch_classes.count)

                # Simple major/minor detection
                key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                properties['key'] = key_names[most_common_pc]
            else:
                properties['key'] = 'C'
        else:
            properties['key'] = 'C'

        # Get duration in bars (rough estimate)
        if properties.get('tempo') and properties.get('time_signature'):
            numerator = int(properties['time_signature'].split('/')[0])
            beats_per_bar = numerator
            seconds_per_beat = 60.0 / properties['tempo']
            seconds_per_bar = seconds_per_beat * beats_per_bar
            properties['duration_bars'] = midi.get_end_time() / seconds_per_bar

        return properties

    def _check_key_preservation(
        self,
        seed_key: Optional[str],
        generated_key: Optional[str],
        tolerance: int = 1
    ) -> bool:
        """
        Check if key is preserved (with tolerance for relative major/minor).

        Args:
            seed_key: Key of seed MIDI
            generated_key: Key of generated MIDI
            tolerance: Number of semitones tolerance

        Returns:
            True if key is preserved within tolerance
        """
        if seed_key is None or generated_key is None:
            return True  # Cannot validate

        key_to_semitone = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }

        seed_semitone = key_to_semitone.get(seed_key, 0)
        generated_semitone = key_to_semitone.get(generated_key, 0)

        # Calculate circular distance
        distance = min(
            abs(seed_semitone - generated_semitone),
            12 - abs(seed_semitone - generated_semitone)
        )

        return distance <= tolerance

    def _check_tempo_preservation(
        self,
        seed_tempo: Optional[float],
        generated_tempo: Optional[float],
        tolerance_percent: float = 10.0
    ) -> bool:
        """
        Check if tempo is preserved within tolerance.

        Args:
            seed_tempo: Tempo of seed MIDI (BPM)
            generated_tempo: Tempo of generated MIDI (BPM)
            tolerance_percent: Percentage tolerance

        Returns:
            True if tempo is preserved within tolerance
        """
        if seed_tempo is None or generated_tempo is None:
            return True  # Cannot validate

        tolerance = seed_tempo * (tolerance_percent / 100.0)
        return abs(seed_tempo - generated_tempo) <= tolerance

    def _check_meter_preservation(
        self,
        seed_meter: Optional[str],
        generated_meter: Optional[str]
    ) -> bool:
        """
        Check if time signature is preserved.

        Args:
            seed_meter: Time signature of seed MIDI
            generated_meter: Time signature of generated MIDI

        Returns:
            True if meter is preserved
        """
        if seed_meter is None or generated_meter is None:
            return True  # Cannot validate

        return seed_meter == generated_meter

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
    """Example usage of structure task executor."""
    from pathlib import Path

    # Example 1: Audio model with structure specification
    print("Example 1: Audio model structure task")

    audio_structure = StructureSpec(
        form="AABA",
        section_lengths=[8, 8, 8, 8],  # 8 bars each
        tempo=120,
        time_signature="4/4"
    )

    audio_input = TaskInput(
        model_type=ModelType.AUDIO,
        structure_spec=audio_structure,
        prompt="lo-fi hip-hop, AABA form, 8 bars each section, 120 BPM, piano and drums",
        duration=30.0  # seconds
    )

    print(f"Structure spec: {audio_structure.form}")
    print(f"Prompt: {audio_input.prompt}")
    print(f"Duration: {audio_input.duration}s")

    # Example 2: Symbolic model with continuation
    print("\nExample 2: Symbolic model structure task")

    symbolic_structure = StructureSpec(
        form="continuation",
        section_lengths=[16],  # Continue for 16 bars
        key="C",
        tempo=120,
        time_signature="4/4"
    )

    # Note: This would require an actual seed MIDI file
    symbolic_input = TaskInput(
        model_type=ModelType.SYMBOLIC,
        structure_spec=symbolic_structure,
        seed_midi_path="/path/to/seed.mid",
        continuation_bars=16
    )

    print(f"Structure spec: {symbolic_structure.form}")
    print(f"Continuation: {symbolic_input.continuation_bars} bars")

    print("\nNote: To run actual generation, provide model wrappers to executor.execute()")


if __name__ == "__main__":
    main()
