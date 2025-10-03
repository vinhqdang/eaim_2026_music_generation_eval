"""
MusicGen-Large model wrapper for text-to-music generation.
Uses Hugging Face transformers for inference.
"""
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from pathlib import Path
import scipy.io.wavfile as wavfile
import numpy as np


class MusicGenWrapper:
    """Wrapper for Meta's MusicGen-Large model."""

    def __init__(self, model_name="facebook/musicgen-large", device=None):
        """
        Initialize MusicGen model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on (cuda/cpu)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading MusicGen model on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)
        print("Model loaded successfully")

    def generate(self, prompt, duration=30, temperature=1.0, top_k=250,
                 top_p=0.0, guidance_scale=3.0, seed=None):
        """
        Generate music from text prompt.

        Args:
            prompt: Text description of desired music
            duration: Duration in seconds (default 30s)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility

        Returns:
            Generated audio as numpy array (sampling_rate, audio)
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Process inputs
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Calculate max_new_tokens based on duration
        # MusicGen uses 50Hz frame rate
        max_new_tokens = int(duration * 50)

        # Generate
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                guidance_scale=guidance_scale,
            )

        # Convert to numpy
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        audio_np = audio_values[0].cpu().numpy()

        return sampling_rate, audio_np

    def generate_and_save(self, prompt, output_path, duration=30, **kwargs):
        """
        Generate music and save to file.

        Args:
            prompt: Text description
            output_path: Output WAV file path
            duration: Duration in seconds
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
        audio = audio / np.max(np.abs(audio))  # Normalize
        audio = (audio * 32767).astype(np.int16)

        wavfile.write(output_path, sampling_rate, audio)

        metadata = {
            "model": "musicgen-large",
            "prompt": prompt,
            "duration": duration,
            "sampling_rate": sampling_rate,
            "output_path": str(output_path),
            **kwargs
        }

        return metadata


def main():
    """Test the wrapper."""
    wrapper = MusicGenWrapper()

    test_prompt = "lo-fi hip-hop, 85 BPM, piano and drums, relaxed vibe"
    print(f"Generating music for prompt: {test_prompt}")

    output_dir = Path(__file__).parent.parent.parent / "runs" / "artifacts" / "wav"
    output_path = output_dir / "test_musicgen.wav"

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
