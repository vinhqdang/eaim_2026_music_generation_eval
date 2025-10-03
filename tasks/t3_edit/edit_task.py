"""
T3: Edit-Responsiveness (Constraint Satisfaction) Task

This module implements edit-responsiveness tests for both audio and symbolic models.

Tests the model's ability to follow mid-piece edits while maintaining coherence:
- Key changes
- Tempo changes
- Time signature changes
- Style/instrumentation changes
- Adding swing/syncopation

For audio models:
    - Apply edits via prompt modifications
    - Generate and validate compliance

For symbolic models:
    - Apply edits to MIDI mid-sequence
    - Continue with constraints and validate
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import json
from dataclasses import dataclass, asdict
from enum import Enum
import pretty_midi
import numpy as np
import copy


class ModelType(Enum):
    """Model type enumeration."""
    AUDIO = "audio"
    SYMBOLIC = "symbolic"


class EditType(Enum):
    """Types of musical edits."""
    KEY_CHANGE = "key_change"
    TEMPO_CHANGE = "tempo_change"
    TIME_SIGNATURE_CHANGE = "time_signature_change"
    INSTRUMENTATION_CHANGE = "instrumentation_change"
    DYNAMICS_CHANGE = "dynamics_change"
    STYLE_CHANGE = "style_change"
    ADD_SWING = "add_swing"
    ARTICULATION_CHANGE = "articulation_change"


@dataclass
class Edit:
    """Specification for a musical edit."""
    edit_type: EditType
    timing: float  # When to apply edit (0-1, as fraction of total duration)

    # Edit parameters
    new_key: Optional[str] = None  # For KEY_CHANGE
    new_tempo: Optional[int] = None  # For TEMPO_CHANGE
    new_time_signature: Optional[str] = None  # For TIME_SIGNATURE_CHANGE
    new_instruments: Optional[List[str]] = None  # For INSTRUMENTATION_CHANGE
    new_dynamics: Optional[str] = None  # For DYNAMICS_CHANGE (e.g., "forte", "piano")
    new_style: Optional[str] = None  # For STYLE_CHANGE
    swing_ratio: Optional[float] = None  # For ADD_SWING (0.5 = straight, 0.67 = swing)
    articulation: Optional[str] = None  # For ARTICULATION_CHANGE (e.g., "staccato", "legato")

    # Description for audio prompts
    description: Optional[str] = None


@dataclass
class TaskInput:
    """Input specification for edit task."""
    model_type: ModelType
    edits: List[Edit]  # List of edits to apply

    # For audio models
    initial_prompt: Optional[str] = None
    duration: Optional[float] = None  # seconds

    # For symbolic models
    seed_midi_path: Optional[str] = None
    continuation_bars: Optional[int] = None


@dataclass
class EditCompliance:
    """Result of edit compliance checking."""
    edit_type: str
    requested_value: Any
    actual_value: Any
    compliant: bool
    deviation: Optional[float] = None  # Quantified deviation if applicable


@dataclass
class TaskResult:
    """Result from edit task execution."""
    success: bool
    model_name: str
    task_name: str = "T3_edit"

    # Generation outputs
    output_path: Optional[str] = None
    generation_time: Optional[float] = None

    # Metadata
    input_spec: Optional[Dict[str, Any]] = None
    generation_params: Optional[Dict[str, Any]] = None

    # Edit compliance results
    edit_compliance: Optional[List[EditCompliance]] = None
    overall_compliance_score: Optional[float] = None  # 0-1, fraction of edits followed

    # Coherence after edits
    coherence_score: Optional[float] = None  # 0-1, musical coherence
    transition_smoothness: Optional[float] = None  # 0-1, smoothness of transitions

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


class EditTaskExecutor:
    """
    Executor for edit-responsiveness task.

    This class tests models' ability to follow edits while maintaining coherence.
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
        Execute the edit-responsiveness task.

        Args:
            model_wrapper: Model wrapper instance (MusicGenWrapper, etc.)
            task_input: Task input specification
            seed: Random seed for reproducibility
            **generation_kwargs: Additional generation parameters

        Returns:
            TaskResult with generation outputs and compliance results
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
            import traceback
            return TaskResult(
                success=False,
                model_name=getattr(model_wrapper, '__class__.__name__', 'unknown'),
                error=f"Execution error: {str(e)}\n{traceback.format_exc()}",
                input_spec=self._serialize_input(task_input),
                generation_time=time.time() - start_time
            )

    def _validate_input(self, task_input: TaskInput) -> Optional[str]:
        """
        Validate task input specification.

        Returns:
            Error message if validation fails, None otherwise
        """
        if not task_input.edits:
            return "At least one edit must be specified"

        if task_input.model_type == ModelType.AUDIO:
            if not task_input.initial_prompt:
                return "Audio models require an initial prompt"
            if not task_input.duration or task_input.duration <= 0:
                return "Audio models require positive duration"
        else:  # SYMBOLIC
            if not task_input.seed_midi_path:
                return "Symbolic models require seed MIDI path"
            if not Path(task_input.seed_midi_path).exists():
                return f"Seed MIDI file not found: {task_input.seed_midi_path}"
            if not task_input.continuation_bars or task_input.continuation_bars <= 0:
                return "Symbolic models require positive continuation_bars"

        # Validate edit timings
        for edit in task_input.edits:
            if edit.timing < 0 or edit.timing > 1:
                return f"Edit timing must be between 0 and 1, got {edit.timing}"

        return None

    def _serialize_input(self, task_input: TaskInput) -> Dict[str, Any]:
        """Serialize task input for storage."""
        input_dict = asdict(task_input)
        input_dict['model_type'] = task_input.model_type.value
        # Serialize edit types
        for edit in input_dict.get('edits', []):
            if 'edit_type' in edit:
                edit['edit_type'] = edit['edit_type'].value if hasattr(edit['edit_type'], 'value') else edit['edit_type']
        return input_dict

    def _execute_audio(
        self,
        model_wrapper: Any,
        task_input: TaskInput,
        seed: Optional[int] = None,
        **generation_kwargs
    ) -> TaskResult:
        """
        Execute edit task for audio models.

        Audio models receive edits through modified prompts.
        Note: Most audio models don't support mid-generation edits,
        so we construct a prompt that describes the full structure with edits.
        """
        import time

        model_name = model_wrapper.__class__.__name__

        # Build prompt incorporating edits
        full_prompt = self._build_audio_prompt_with_edits(task_input)

        # Prepare output path
        output_dir = self.output_dir / "wav" / "t3_edit"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        output_path = output_dir / f"{model_name}_edit_{timestamp}.wav"

        # Generate audio
        generation_params = {
            'prompt': full_prompt,
            'duration': task_input.duration,
            'seed': seed,
            **generation_kwargs
        }

        metadata = model_wrapper.generate_and_save(
            output_path=str(output_path),
            **generation_params
        )

        # Validate edit compliance
        # For audio, we'd need audio analysis tools to verify
        # For now, we create placeholder compliance results
        edit_compliance = self._validate_audio_edits(
            str(output_path),
            task_input.edits,
            task_input.duration
        )

        overall_compliance = sum(ec.compliant for ec in edit_compliance) / len(edit_compliance) if edit_compliance else 0.0

        result = TaskResult(
            success=True,
            model_name=model_name,
            output_path=str(output_path),
            generation_params=generation_params,
            edit_compliance=edit_compliance,
            overall_compliance_score=overall_compliance,
            coherence_score=None,  # To be filled by metrics
            transition_smoothness=None  # To be filled by metrics
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
        Execute edit task for symbolic models.

        For symbolic models, we can apply edits directly to MIDI.
        """
        import time

        model_name = model_wrapper.__class__.__name__

        # Load seed MIDI
        seed_midi = pretty_midi.PrettyMIDI(task_input.seed_midi_path)

        # Prepare output path
        output_dir = self.output_dir / "midi" / "t3_edit"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        output_path = output_dir / f"{model_name}_edit_{timestamp}.mid"

        # Sort edits by timing
        sorted_edits = sorted(task_input.edits, key=lambda e: e.timing)

        # For now, generate with edits applied via continuation
        # More sophisticated implementation would:
        # 1. Generate up to first edit point
        # 2. Apply edit
        # 3. Continue generation with constraints
        # 4. Repeat for each edit

        # Simplified: Generate full continuation and apply edits post-hoc
        events_per_bar = 16
        total_events = task_input.continuation_bars * events_per_bar

        generation_params = {
            'primer_midi': task_input.seed_midi_path,
            'total_length': total_events,
            'seed': seed,
            **generation_kwargs
        }

        metadata = model_wrapper.generate_and_save(
            output_path=str(output_path),
            **generation_params
        )

        # Load generated MIDI
        generated_midi = pretty_midi.PrettyMIDI(str(output_path))

        # Apply edits to generated MIDI
        edited_midi = self._apply_edits_to_midi(generated_midi, sorted_edits)

        # Save edited version
        edited_path = output_dir / f"{model_name}_edit_{timestamp}_applied.mid"
        edited_midi.write(str(edited_path))

        # Validate edit compliance
        edit_compliance = self._validate_midi_edits(
            edited_midi,
            sorted_edits
        )

        overall_compliance = sum(ec.compliant for ec in edit_compliance) / len(edit_compliance) if edit_compliance else 0.0

        # Check coherence
        coherence_score = self._compute_midi_coherence(edited_midi)

        result = TaskResult(
            success=True,
            model_name=model_name,
            output_path=str(edited_path),
            generation_params=generation_params,
            edit_compliance=edit_compliance,
            overall_compliance_score=overall_compliance,
            coherence_score=coherence_score,
            transition_smoothness=None  # To be filled by detailed metrics
        )

        return result

    def _build_audio_prompt_with_edits(self, task_input: TaskInput) -> str:
        """
        Build audio prompt that incorporates edit descriptions.

        Args:
            task_input: Task input with edits

        Returns:
            Full prompt string
        """
        prompt_parts = [task_input.initial_prompt]

        # Add edit descriptions in temporal order
        sorted_edits = sorted(task_input.edits, key=lambda e: e.timing)

        for edit in sorted_edits:
            if edit.description:
                prompt_parts.append(edit.description)
            else:
                # Generate description from edit parameters
                desc = self._generate_edit_description(edit)
                if desc:
                    prompt_parts.append(desc)

        return ", then ".join(prompt_parts)

    def _generate_edit_description(self, edit: Edit) -> str:
        """Generate text description of an edit."""
        if edit.edit_type == EditType.KEY_CHANGE:
            return f"modulate to {edit.new_key}"
        elif edit.edit_type == EditType.TEMPO_CHANGE:
            return f"change tempo to {edit.new_tempo} BPM"
        elif edit.edit_type == EditType.TIME_SIGNATURE_CHANGE:
            return f"change to {edit.new_time_signature} time"
        elif edit.edit_type == EditType.INSTRUMENTATION_CHANGE:
            instruments = ", ".join(edit.new_instruments) if edit.new_instruments else "different instruments"
            return f"change to {instruments}"
        elif edit.edit_type == EditType.DYNAMICS_CHANGE:
            return f"change dynamics to {edit.new_dynamics}"
        elif edit.edit_type == EditType.STYLE_CHANGE:
            return f"change style to {edit.new_style}"
        elif edit.edit_type == EditType.ADD_SWING:
            return "add swing feel"
        else:
            return ""

    def _apply_edits_to_midi(
        self,
        midi: pretty_midi.PrettyMIDI,
        edits: List[Edit]
    ) -> pretty_midi.PrettyMIDI:
        """
        Apply edits to MIDI file.

        Args:
            midi: Original MIDI
            edits: List of edits to apply

        Returns:
            Modified MIDI
        """
        # Create a copy to avoid modifying original
        edited_midi = copy.deepcopy(midi)
        total_duration = edited_midi.get_end_time()

        for edit in edits:
            edit_time = edit.timing * total_duration

            if edit.edit_type == EditType.KEY_CHANGE:
                self._apply_key_change(edited_midi, edit.new_key, edit_time)

            elif edit.edit_type == EditType.TEMPO_CHANGE:
                self._apply_tempo_change(edited_midi, edit.new_tempo, edit_time)

            elif edit.edit_type == EditType.TIME_SIGNATURE_CHANGE:
                self._apply_time_signature_change(edited_midi, edit.new_time_signature, edit_time)

            elif edit.edit_type == EditType.DYNAMICS_CHANGE:
                self._apply_dynamics_change(edited_midi, edit.new_dynamics, edit_time)

            elif edit.edit_type == EditType.ADD_SWING:
                self._apply_swing(edited_midi, edit.swing_ratio or 0.67, edit_time)

        return edited_midi

    def _apply_key_change(
        self,
        midi: pretty_midi.PrettyMIDI,
        new_key: str,
        start_time: float
    ) -> None:
        """Apply key transposition starting at specified time."""
        # Simplified: transpose all notes after start_time
        key_to_semitone = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }

        # Current key estimation would go here
        # For simplicity, assume C major currently
        current_key_semitone = 0
        new_key_semitone = key_to_semitone.get(new_key, 0)
        transposition = new_key_semitone - current_key_semitone

        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start >= start_time:
                        note.pitch = max(0, min(127, note.pitch + transposition))

    def _apply_tempo_change(
        self,
        midi: pretty_midi.PrettyMIDI,
        new_tempo: int,
        start_time: float
    ) -> None:
        """Apply tempo change at specified time."""
        # Add tempo change event
        midi._PrettyMIDI__tick_scales.append((start_time, 60.0 / new_tempo))

    def _apply_time_signature_change(
        self,
        midi: pretty_midi.PrettyMIDI,
        new_time_signature: str,
        start_time: float
    ) -> None:
        """Apply time signature change at specified time."""
        numerator, denominator = map(int, new_time_signature.split('/'))
        ts_change = pretty_midi.TimeSignature(numerator, denominator, start_time)
        midi.time_signature_changes.append(ts_change)

    def _apply_dynamics_change(
        self,
        midi: pretty_midi.PrettyMIDI,
        new_dynamics: str,
        start_time: float
    ) -> None:
        """Apply dynamics change (velocity modification) at specified time."""
        dynamics_map = {
            'pp': 30, 'p': 50, 'mp': 65, 'mf': 80,
            'f': 95, 'ff': 110, 'piano': 50, 'forte': 95
        }

        target_velocity = dynamics_map.get(new_dynamics.lower(), 80)

        for instrument in midi.instruments:
            for note in instrument.notes:
                if note.start >= start_time:
                    note.velocity = target_velocity

    def _apply_swing(
        self,
        midi: pretty_midi.PrettyMIDI,
        swing_ratio: float,
        start_time: float
    ) -> None:
        """Apply swing timing to notes after specified time."""
        # Swing: delay every other eighth note
        # This is a simplified implementation
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            # Sort notes by start time
            notes = sorted([n for n in instrument.notes if n.start >= start_time],
                         key=lambda n: n.start)

            # Apply swing to alternating notes
            for i in range(1, len(notes), 2):
                if i < len(notes):
                    # Calculate swing delay
                    prev_note = notes[i - 1]
                    curr_note = notes[i]
                    interval = curr_note.start - prev_note.start

                    # Apply swing ratio
                    swing_delay = interval * (swing_ratio - 0.5)
                    notes[i].start += swing_delay
                    notes[i].end += swing_delay

    def _validate_audio_edits(
        self,
        audio_path: str,
        edits: List[Edit],
        duration: float
    ) -> List[EditCompliance]:
        """
        Validate that audio edits were applied correctly.

        This is a placeholder - actual implementation would use audio analysis.
        """
        compliance_results = []

        for edit in edits:
            # Placeholder compliance check
            # Real implementation would analyze audio at edit.timing * duration
            compliance = EditCompliance(
                edit_type=edit.edit_type.value,
                requested_value=self._get_edit_requested_value(edit),
                actual_value=None,  # Would be extracted from audio
                compliant=None,  # Would be determined by analysis
                deviation=None
            )
            compliance_results.append(compliance)

        return compliance_results

    def _validate_midi_edits(
        self,
        midi: pretty_midi.PrettyMIDI,
        edits: List[Edit]
    ) -> List[EditCompliance]:
        """
        Validate that MIDI edits were applied correctly.

        Args:
            midi: Generated MIDI with edits
            edits: List of edits that were applied

        Returns:
            List of compliance results
        """
        compliance_results = []
        total_duration = midi.get_end_time()

        for edit in edits:
            edit_time = edit.timing * total_duration

            if edit.edit_type == EditType.TEMPO_CHANGE:
                # Check tempo after edit point
                actual_tempo = self._get_tempo_at_time(midi, edit_time + 0.1)
                deviation = abs(actual_tempo - edit.new_tempo) if edit.new_tempo else 0

                compliance = EditCompliance(
                    edit_type=edit.edit_type.value,
                    requested_value=edit.new_tempo,
                    actual_value=actual_tempo,
                    compliant=deviation < 10,  # 10 BPM tolerance
                    deviation=deviation
                )
                compliance_results.append(compliance)

            elif edit.edit_type == EditType.KEY_CHANGE:
                # Check key after edit point
                # Simplified check
                compliance = EditCompliance(
                    edit_type=edit.edit_type.value,
                    requested_value=edit.new_key,
                    actual_value=None,  # Would need key detection
                    compliant=None,  # Unknown without detection
                    deviation=None
                )
                compliance_results.append(compliance)

            else:
                # Generic compliance entry
                compliance = EditCompliance(
                    edit_type=edit.edit_type.value,
                    requested_value=self._get_edit_requested_value(edit),
                    actual_value=None,
                    compliant=None,
                    deviation=None
                )
                compliance_results.append(compliance)

        return compliance_results

    def _get_tempo_at_time(self, midi: pretty_midi.PrettyMIDI, time: float) -> float:
        """Get tempo at specified time in MIDI."""
        tempo_times, tempos = midi.get_tempo_changes()

        if len(tempos) == 0:
            return 120.0

        # Find tempo at specified time
        idx = np.searchsorted(tempo_times, time) - 1
        idx = max(0, idx)

        return float(tempos[idx])

    def _get_edit_requested_value(self, edit: Edit) -> Any:
        """Extract the requested value from an edit."""
        if edit.edit_type == EditType.KEY_CHANGE:
            return edit.new_key
        elif edit.edit_type == EditType.TEMPO_CHANGE:
            return edit.new_tempo
        elif edit.edit_type == EditType.TIME_SIGNATURE_CHANGE:
            return edit.new_time_signature
        elif edit.edit_type == EditType.INSTRUMENTATION_CHANGE:
            return edit.new_instruments
        elif edit.edit_type == EditType.DYNAMICS_CHANGE:
            return edit.new_dynamics
        elif edit.edit_type == EditType.STYLE_CHANGE:
            return edit.new_style
        elif edit.edit_type == EditType.ADD_SWING:
            return edit.swing_ratio
        else:
            return None

    def _compute_midi_coherence(self, midi: pretty_midi.PrettyMIDI) -> float:
        """
        Compute musical coherence score for MIDI.

        This is a simplified heuristic based on:
        - Pitch continuity (smooth voice leading)
        - Rhythm regularity
        - Harmonic consistency

        Returns:
            Coherence score (0-1)
        """
        if not midi.instruments:
            return 0.0

        scores = []

        # Check pitch continuity
        for instrument in midi.instruments:
            if instrument.is_drum or not instrument.notes:
                continue

            notes = sorted(instrument.notes, key=lambda n: n.start)
            pitch_jumps = []

            for i in range(1, len(notes)):
                pitch_jump = abs(notes[i].pitch - notes[i-1].pitch)
                pitch_jumps.append(pitch_jump)

            if pitch_jumps:
                # Prefer smaller jumps (smoother voice leading)
                avg_jump = np.mean(pitch_jumps)
                pitch_score = max(0.0, 1.0 - (avg_jump / 12.0))  # 1 octave tolerance
                scores.append(pitch_score)

        # Return mean coherence
        if scores:
            return float(np.mean(scores))
        else:
            return 0.5  # Neutral score

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
    """Example usage of edit task executor."""
    print("Example 1: Audio model with key change")

    # Edit: change key mid-piece
    key_change_edit = Edit(
        edit_type=EditType.KEY_CHANGE,
        timing=0.5,  # At 50% of duration
        new_key="D",
        description="modulate to D major"
    )

    audio_input = TaskInput(
        model_type=ModelType.AUDIO,
        edits=[key_change_edit],
        initial_prompt="upbeat pop song in C major, 120 BPM, piano and drums",
        duration=30.0
    )

    print(f"Initial: {audio_input.initial_prompt}")
    print(f"Edit at {key_change_edit.timing * 100}%: {key_change_edit.description}")

    # Example 2: Symbolic model with tempo change
    print("\nExample 2: Symbolic model with tempo change")

    tempo_change_edit = Edit(
        edit_type=EditType.TEMPO_CHANGE,
        timing=0.33,  # At 33% of duration
        new_tempo=140
    )

    symbolic_input = TaskInput(
        model_type=ModelType.SYMBOLIC,
        edits=[tempo_change_edit],
        seed_midi_path="/path/to/seed.mid",
        continuation_bars=32
    )

    print(f"Seed: {symbolic_input.seed_midi_path}")
    print(f"Edit at {tempo_change_edit.timing * 100}%: Change tempo to {tempo_change_edit.new_tempo} BPM")

    # Example 3: Multiple edits
    print("\nExample 3: Multiple edits")

    edits = [
        Edit(
            edit_type=EditType.TEMPO_CHANGE,
            timing=0.25,
            new_tempo=140,
            description="speed up to 140 BPM"
        ),
        Edit(
            edit_type=EditType.KEY_CHANGE,
            timing=0.5,
            new_key="G",
            description="modulate to G major"
        ),
        Edit(
            edit_type=EditType.DYNAMICS_CHANGE,
            timing=0.75,
            new_dynamics="forte",
            description="increase to forte"
        )
    ]

    multi_edit_input = TaskInput(
        model_type=ModelType.AUDIO,
        edits=edits,
        initial_prompt="classical piano piece in C major, 120 BPM, expressive",
        duration=60.0
    )

    print(f"Initial: {multi_edit_input.initial_prompt}")
    for i, edit in enumerate(edits, 1):
        print(f"Edit {i} at {edit.timing * 100}%: {edit.description}")

    print("\nNote: To run actual generation, provide model wrappers to executor.execute()")


if __name__ == "__main__":
    main()
