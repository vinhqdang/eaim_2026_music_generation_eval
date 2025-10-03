"""
Fréchet Audio Distance (FAD) implementation.

This module calculates FAD using both VGGish and CLAP embeddings to measure
the quality of generated audio by comparing feature distributions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Literal

import numpy as np
import torch
import librosa
from scipy import linalg
from transformers import ClapModel, ClapProcessor


class FADCalculator:
    """
    Calculate Fréchet Audio Distance (FAD) between reference and generated audio.

    FAD measures the distance between two distributions of audio embeddings
    using the Fréchet distance (also known as Wasserstein-2 distance for Gaussians).
    Lower FAD scores indicate better quality and more similar distributions.

    Attributes:
        embedding_type: Type of embedding to use ('vggish' or 'clap')
        sample_rate: Target sample rate for audio processing
        device: PyTorch device for computation
    """

    def __init__(
        self,
        embedding_type: Literal["vggish", "clap"] = "clap",
        sample_rate: int = 48000,
        device: Optional[str] = None,
    ):
        """
        Initialize FAD calculator.

        Args:
            embedding_type: Type of embedding ('vggish' or 'clap')
            sample_rate: Target sample rate for audio
            device: Device for computation ('cuda' or 'cpu')
        """
        self.embedding_type = embedding_type
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model based on embedding_type."""
        if self.embedding_type == "clap":
            try:
                model_name = "laion/larger_clap_music"
                self._model = ClapModel.from_pretrained(model_name).to(self.device)
                self._processor = ClapProcessor.from_pretrained(model_name)
                self._model.eval()
            except Exception as e:
                raise RuntimeError(f"Failed to load CLAP model: {e}")

        elif self.embedding_type == "vggish":
            try:
                # Using torch.hub VGGish implementation
                self._model = torch.hub.load(
                    'harritaylor/torchvggish',
                    'vggish',
                    preprocess=False,
                    postprocess=False
                )
                self._model = self._model.to(self.device)
                self._model.eval()
            except Exception as e:
                raise RuntimeError(f"Failed to load VGGish model: {e}")

        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

    def _load_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Load audio file and resample to target sample rate.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio waveform as numpy array

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If audio loading fails
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return audio
        except Exception as e:
            raise RuntimeError(f"Failed to load audio {audio_path}: {e}")

    def _extract_vggish_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract VGGish embeddings from audio.

        Args:
            audio: Audio waveform

        Returns:
            VGGish embeddings (N, 128)
        """
        # VGGish expects 16kHz audio
        if self.sample_rate != 16000:
            audio = librosa.resample(
                audio,
                orig_sr=self.sample_rate,
                target_sr=16000
            )

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            embeddings = self._model(audio_tensor)

        return embeddings.cpu().numpy()

    def _extract_clap_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract CLAP embeddings from audio.

        Args:
            audio: Audio waveform

        Returns:
            CLAP embeddings (1, 512)
        """
        # CLAP expects 48kHz audio
        if self.sample_rate != 48000:
            audio = librosa.resample(
                audio,
                orig_sr=self.sample_rate,
                target_sr=48000
            )

        # Process audio through CLAP processor
        inputs = self._processor(
            audios=[audio],
            return_tensors="pt",
            sampling_rate=48000
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self._model.get_audio_features(**inputs)

        return embeddings.cpu().numpy()

    def extract_embeddings(
        self,
        audio_paths: List[Union[str, Path]]
    ) -> np.ndarray:
        """
        Extract embeddings from multiple audio files.

        Args:
            audio_paths: List of audio file paths

        Returns:
            Embeddings array (N, D) where N is number of samples and D is embedding dim
        """
        all_embeddings = []

        for audio_path in audio_paths:
            audio = self._load_audio(audio_path)

            if self.embedding_type == "vggish":
                embeddings = self._extract_vggish_embedding(audio)
            else:  # clap
                embeddings = self._extract_clap_embedding(audio)

            all_embeddings.append(embeddings)

        # Concatenate all embeddings
        embeddings_array = np.vstack(all_embeddings)
        return embeddings_array

    @staticmethod
    def calculate_frechet_distance(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """
        Calculate Fréchet distance between two Gaussian distributions.

        The Fréchet distance between two multivariate Gaussians is:
        d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))

        Args:
            mu1: Mean of first distribution
            sigma1: Covariance of first distribution
            mu2: Mean of second distribution
            sigma2: Covariance of second distribution
            eps: Small constant for numerical stability

        Returns:
            Fréchet distance
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m} too large")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    def compute(
        self,
        reference_paths: List[Union[str, Path]],
        generated_paths: List[Union[str, Path]],
    ) -> Dict[str, float]:
        """
        Compute FAD between reference and generated audio.

        Args:
            reference_paths: List of reference audio file paths
            generated_paths: List of generated audio file paths

        Returns:
            Dictionary containing:
                - fad_score: Fréchet Audio Distance
                - embedding_type: Type of embedding used
                - n_reference: Number of reference samples
                - n_generated: Number of generated samples

        Raises:
            ValueError: If input lists are empty or have insufficient samples
        """
        if not reference_paths or not generated_paths:
            raise ValueError("Both reference and generated paths must be non-empty")

        if len(reference_paths) < 2 or len(generated_paths) < 2:
            raise ValueError("Need at least 2 samples for each distribution")

        # Extract embeddings
        print(f"Extracting embeddings for {len(reference_paths)} reference samples...")
        ref_embeddings = self.extract_embeddings(reference_paths)

        print(f"Extracting embeddings for {len(generated_paths)} generated samples...")
        gen_embeddings = self.extract_embeddings(generated_paths)

        # Calculate statistics
        mu_ref = np.mean(ref_embeddings, axis=0)
        sigma_ref = np.cov(ref_embeddings, rowvar=False)

        mu_gen = np.mean(gen_embeddings, axis=0)
        sigma_gen = np.cov(gen_embeddings, rowvar=False)

        # Calculate FAD
        fad_score = self.calculate_frechet_distance(
            mu_ref, sigma_ref, mu_gen, sigma_gen
        )

        return {
            "fad_score": fad_score,
            "embedding_type": self.embedding_type,
            "n_reference": len(reference_paths),
            "n_generated": len(generated_paths),
            "embedding_dim": ref_embeddings.shape[1],
        }


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python fad.py <reference_dir> <generated_dir>")
        sys.exit(1)

    ref_dir = Path(sys.argv[1])
    gen_dir = Path(sys.argv[2])

    # Get audio files
    ref_files = sorted(ref_dir.glob("*.wav")) + sorted(ref_dir.glob("*.mp3"))
    gen_files = sorted(gen_dir.glob("*.wav")) + sorted(gen_dir.glob("*.mp3"))

    if not ref_files or not gen_files:
        print("No audio files found in provided directories")
        sys.exit(1)

    # Calculate FAD with CLAP
    print("\nCalculating FAD with CLAP embeddings...")
    calculator = FADCalculator(embedding_type="clap")
    result = calculator.compute(ref_files, gen_files)

    print("\nResults:")
    for key, value in result.items():
        print(f"  {key}: {value}")
