"""
Stable Audio Open 1.0 model wrapper for text-to-audio generation.
Uses Stability AI's stable-audio-open-1.0 from HuggingFace.
"""
import torch
from pathlib import Path
import scipy.io.wavfile as wavfile
import numpy as np


class StableAudioWrapper:
    """Wrapper for Stability AI's Stable Audio Open 1.0 model."""

    def __init__(self, model_name="stabilityai/stable-audio-open-1.0", device=None):
        """
        Initialize Stable Audio model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on (cuda/cpu)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading Stable Audio model on {self.device}...")

        try:
            from stable_audio_tools import get_pretrained_model
            from stable_audio_tools.inference.generation import generate_diffusion_cond

            self.model, self.model_config = get_pretrained_model(model_name)
            self.model.to(self.device)
            self.generate_diffusion_cond = generate_diffusion_cond

            # Get sampling rate from config
            self.sampling_rate = self.model_config["sample_rate"]
            self.max_duration = 47  # Maximum duration in seconds for stable-audio-open-1.0

            print(f"Model loaded successfully (max duration: {self.max_duration}s)")
        except ImportError:
            raise ImportError(
                "stable-audio-tools not installed. Install with: "
                "pip install stable-audio-tools"
            )

    def generate(self, prompt, duration=30, steps=100, cfg_scale=7.0, seed=None):
        """
        Generate audio from text prompt.

        Args:
            prompt: Text description of desired audio
            duration: Duration in seconds (default 30s, max 47s)
            steps: Number of diffusion steps (default 100)
            cfg_scale: Classifier-free guidance scale (default 7.0)
            seed: Random seed for reproducibility

        Returns:
            Generated audio as numpy array (sampling_rate, audio)
        """
        if duration > self.max_duration:
            print(f"Warning: Duration {duration}s exceeds max {self.max_duration}s. "
                  f"Clamping to {self.max_duration}s")
            duration = self.max_duration

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        # Generate audio
        with torch.no_grad():
            audio = self.generate_diffusion_cond(
                self.model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning={
                    "prompt": prompt,
                    "seconds_start": 0,
                    "seconds_total": duration
                },
                sample_size=int(duration * self.sampling_rate),
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=self.device
            )

        # Convert to numpy
        audio_np = audio.cpu().numpy().squeeze()

        return self.sampling_rate, audio_np

    def generate_and_save(self, prompt, output_path, duration=30, **kwargs):
        """
        Generate audio and save to file.

        Args:
            prompt: Text description
            output_path: Output WAV file path
            duration: Duration in seconds (max 47s)
            **kwargs: Additional generation parameters

        Returns:
            Metadata dictionary
        """
        sampling_rate, audio = self.generate(prompt, duration=duration, **kwargs)

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize and save
        audio = np.squeeze(audio)
        audio = audio / np.max(np.abs(audio) + 1e-8)  # Normalize with epsilon
        audio = (audio * 32767).astype(np.int16)

        wavfile.write(output_path, sampling_rate, audio)

        metadata = {
            "model": "stable-audio-open-1.0",
            "prompt": prompt,
            "duration": min(duration, self.max_duration),
            "sampling_rate": sampling_rate,
            "output_path": str(output_path),
            **kwargs
        }

        return metadata


def main():
    """Test the wrapper."""
    wrapper = StableAudioWrapper()

    test_prompt = "128 BPM electronic dance music with synthesizers and drums"
    print(f"Generating audio for prompt: {test_prompt}")

    output_dir = Path(__file__).parent.parent.parent / "runs" / "artifacts" / "wav"
    output_path = output_dir / "test_stableaudio.wav"

    metadata = wrapper.generate_and_save(
        prompt=test_prompt,
        output_path=output_path,
        duration=30,
        seed=42
    )

    print(f"Generated audio saved to: {metadata['output_path']}")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()
