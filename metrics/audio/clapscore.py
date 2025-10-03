"""
CLAP Score implementation for text-audio alignment evaluation.

This module uses CLAP (Contrastive Language-Audio Pretraining) to measure
the alignment between text descriptions and generated audio.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import librosa
from transformers import ClapModel, ClapProcessor


class CLAPScoreCalculator:
    """
    Calculate CLAP Score for text-audio alignment.

    CLAP Score measures how well audio matches text descriptions using
    contrastive language-audio embeddings. Higher scores indicate better alignment.

    Attributes:
        model_name: Name of the pretrained CLAP model
        sample_rate: Target sample rate for audio processing
        device: PyTorch device for computation
    """

    def __init__(
        self,
        model_name: str = "laion/larger_clap_music",
        sample_rate: int = 48000,
        device: Optional[str] = None,
    ):
        """
        Initialize CLAP Score calculator.

        Args:
            model_name: HuggingFace model name for CLAP
            sample_rate: Target sample rate for audio
            device: Device for computation ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model()

    def _load_model(self) -> None:
        """Load the CLAP model and processor."""
        try:
            self.model = ClapModel.from_pretrained(self.model_name).to(self.device)
            self.processor = ClapProcessor.from_pretrained(self.model_name)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load CLAP model {self.model_name}: {e}")

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

    def get_audio_embeddings(
        self,
        audio_paths: List[Union[str, Path]]
    ) -> torch.Tensor:
        """
        Extract audio embeddings for multiple files.

        Args:
            audio_paths: List of audio file paths

        Returns:
            Audio embeddings tensor (N, embedding_dim)
        """
        audios = [self._load_audio(path) for path in audio_paths]

        inputs = self.processor(
            audios=audios,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_embeds = self.model.get_audio_features(**inputs)

        return audio_embeds

    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Extract text embeddings for multiple text descriptions.

        Args:
            texts: List of text descriptions

        Returns:
            Text embeddings tensor (N, embedding_dim)
        """
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)

        return text_embeds

    def compute_similarity(
        self,
        audio_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute cosine similarity between audio and text embeddings.

        Args:
            audio_embeds: Audio embeddings (N, D)
            text_embeds: Text embeddings (M, D)
            temperature: Temperature for scaling similarity scores

        Returns:
            Similarity matrix (N, M)
        """
        # Normalize embeddings
        audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = (audio_embeds @ text_embeds.T) / temperature

        return similarity

    def compute(
        self,
        audio_paths: List[Union[str, Path]],
        text_descriptions: List[str],
        return_individual_scores: bool = False,
    ) -> Dict[str, Union[float, List[float], np.ndarray]]:
        """
        Compute CLAP Score between audio and text descriptions.

        Args:
            audio_paths: List of audio file paths
            text_descriptions: List of text descriptions (must match audio_paths length)
            return_individual_scores: Whether to return scores for each pair

        Returns:
            Dictionary containing:
                - clap_score: Average similarity score
                - std: Standard deviation of scores
                - min_score: Minimum score
                - max_score: Maximum score
                - individual_scores: List of individual scores (if requested)
                - n_samples: Number of audio-text pairs

        Raises:
            ValueError: If inputs are empty or lengths don't match
        """
        if not audio_paths or not text_descriptions:
            raise ValueError("Both audio_paths and text_descriptions must be non-empty")

        if len(audio_paths) != len(text_descriptions):
            raise ValueError(
                f"Number of audio files ({len(audio_paths)}) must match "
                f"number of text descriptions ({len(text_descriptions)})"
            )

        # Extract embeddings
        print(f"Extracting embeddings for {len(audio_paths)} audio-text pairs...")
        audio_embeds = self.get_audio_embeddings(audio_paths)
        text_embeds = self.get_text_embeddings(text_descriptions)

        # Compute similarity
        similarity = self.compute_similarity(audio_embeds, text_embeds)

        # Extract diagonal (matching pairs)
        scores = torch.diagonal(similarity).cpu().numpy()

        result = {
            "clap_score": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "n_samples": len(audio_paths),
        }

        if return_individual_scores:
            result["individual_scores"] = scores.tolist()

        return result

    def compute_cross_similarity(
        self,
        audio_paths: List[Union[str, Path]],
        text_descriptions: List[str],
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute cross-similarity matrix between all audio-text pairs.

        Useful for evaluating whether the model correctly matches
        audio with the right text descriptions.

        Args:
            audio_paths: List of audio file paths
            text_descriptions: List of text descriptions

        Returns:
            Dictionary containing:
                - similarity_matrix: Full similarity matrix (N, M)
                - diagonal_mean: Mean of diagonal (correct pairs)
                - off_diagonal_mean: Mean of off-diagonal (incorrect pairs)
                - accuracy: Percentage where diagonal is maximum in each row

        Raises:
            ValueError: If inputs are empty
        """
        if not audio_paths or not text_descriptions:
            raise ValueError("Both audio_paths and text_descriptions must be non-empty")

        # Extract embeddings
        audio_embeds = self.get_audio_embeddings(audio_paths)
        text_embeds = self.get_text_embeddings(text_descriptions)

        # Compute full similarity matrix
        similarity = self.compute_similarity(audio_embeds, text_embeds)
        similarity_np = similarity.cpu().numpy()

        # Calculate metrics
        n_audio = len(audio_paths)
        n_text = len(text_descriptions)

        if n_audio == n_text:
            diagonal = np.diagonal(similarity_np)
            diagonal_mean = float(np.mean(diagonal))

            # Off-diagonal mean
            mask = ~np.eye(n_audio, dtype=bool)
            off_diagonal_mean = float(np.mean(similarity_np[mask]))

            # Accuracy: how often is diagonal the max in each row?
            max_indices = np.argmax(similarity_np, axis=1)
            correct = np.sum(max_indices == np.arange(n_audio))
            accuracy = float(correct / n_audio)
        else:
            diagonal_mean = None
            off_diagonal_mean = None
            accuracy = None

        return {
            "similarity_matrix": similarity_np,
            "diagonal_mean": diagonal_mean,
            "off_diagonal_mean": off_diagonal_mean,
            "accuracy": accuracy,
            "n_audio": n_audio,
            "n_text": n_text,
        }

    def rank_texts_for_audio(
        self,
        audio_path: Union[str, Path],
        candidate_texts: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Rank candidate text descriptions for a given audio file.

        Args:
            audio_path: Path to audio file
            candidate_texts: List of candidate text descriptions
            top_k: Return only top k results (None for all)

        Returns:
            List of (text, score) tuples sorted by score (descending)

        Raises:
            ValueError: If candidate_texts is empty
        """
        if not candidate_texts:
            raise ValueError("candidate_texts must be non-empty")

        # Extract embeddings
        audio_embeds = self.get_audio_embeddings([audio_path])
        text_embeds = self.get_text_embeddings(candidate_texts)

        # Compute similarity
        similarity = self.compute_similarity(audio_embeds, text_embeds)
        scores = similarity[0].cpu().numpy()

        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]

        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]

        return [(candidate_texts[i], float(scores[i])) for i in sorted_indices]


if __name__ == "__main__":
    # Example usage
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python clapscore.py <audio_file1,audio_file2,...> <text1|text2|...>")
        print("Example: python clapscore.py 'song1.wav,song2.wav' 'piano melody|guitar riff'")
        sys.exit(1)

    audio_files = [Path(p.strip()) for p in sys.argv[1].split(",")]
    texts = [t.strip() for t in sys.argv[2].split("|")]

    if len(audio_files) != len(texts):
        print(f"Warning: Number of audio files ({len(audio_files)}) != number of texts ({len(texts)})")

    # Calculate CLAP Score
    print("\nCalculating CLAP Score...")
    calculator = CLAPScoreCalculator()

    if len(audio_files) == len(texts):
        result = calculator.compute(audio_files, texts, return_individual_scores=True)

        print("\nResults:")
        print(f"  Average CLAP Score: {result['clap_score']:.4f}")
        print(f"  Std: {result['std']:.4f}")
        print(f"  Min: {result['min_score']:.4f}")
        print(f"  Max: {result['max_score']:.4f}")

        if result.get("individual_scores"):
            print("\n  Individual scores:")
            for i, score in enumerate(result["individual_scores"]):
                print(f"    {audio_files[i].name}: {score:.4f}")
    else:
        # Cross-similarity
        result = calculator.compute_cross_similarity(audio_files, texts)
        print("\nCross-similarity matrix:")
        print(result["similarity_matrix"])
        if result["accuracy"] is not None:
            print(f"\nAccuracy: {result['accuracy']:.2%}")
