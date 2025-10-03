"""
Generate 100 structured text prompts for audio models.
Spans genre, instrumentation, structure cues, BPM, and effects.
"""
import json
import random
from pathlib import Path


class PromptGenerator:
    """Generates structured prompts for music generation evaluation."""

    GENRES = [
        "lo-fi hip-hop", "jazz", "classical piano", "rock", "electronic",
        "ambient", "folk", "blues", "reggae", "metal", "pop", "funk",
        "soul", "techno", "house", "drum and bass", "dubstep", "trap"
    ]

    INSTRUMENTS = [
        ["piano", "bass", "drums"],
        ["guitar", "bass", "drums"],
        ["strings", "piano"],
        ["synthesizer", "bass", "drums"],
        ["acoustic guitar", "vocals"],
        ["electric guitar", "bass", "drums", "keyboard"],
        ["saxophone", "piano", "bass", "drums"],
        ["violin", "cello", "piano"],
    ]

    STRUCTURES = [
        "8-bar intro, 16-bar verse, 16-bar chorus",
        "AABA, 8 bars each",
        "verse-chorus-verse-chorus-bridge-chorus",
        "32-bar standard form",
        "intro-A-B-A",
        "4-bar intro, 32-bar development",
    ]

    EFFECTS = [
        "vinyl crackle", "reverb", "echo", "distortion", "lo-fi texture",
        "tape saturation", "bit-crush", "chorus", "phaser", "clean production"
    ]

    BPM_RANGES = {
        "slow": (60, 80),
        "medium": (90, 120),
        "fast": (130, 160),
        "very fast": (170, 200)
    }

    def __init__(self, seed=42):
        """Initialize with random seed for reproducibility."""
        random.seed(seed)

    def generate_prompt(self, prompt_id):
        """Generate a single structured prompt."""
        genre = random.choice(self.GENRES)
        instruments = ", ".join(random.choice(self.INSTRUMENTS))
        structure = random.choice(self.STRUCTURES)
        effect = random.choice(self.EFFECTS)

        tempo_label = random.choice(list(self.BPM_RANGES.keys()))
        bpm_range = self.BPM_RANGES[tempo_label]
        bpm = random.randint(bpm_range[0], bpm_range[1])

        # Build prompt
        prompt_parts = [
            f"{genre}",
            f"{bpm} BPM",
            f"instrumentation: {instruments}",
            f"structure: {structure}",
            f"{effect}"
        ]

        prompt_text = ", ".join(prompt_parts)

        return {
            "id": prompt_id,
            "text": prompt_text,
            "genre": genre,
            "bpm": bpm,
            "instruments": instruments,
            "structure": structure,
            "effect": effect
        }

    def generate_all(self, num_prompts=100):
        """Generate all prompts."""
        prompts = []
        for i in range(num_prompts):
            prompt = self.generate_prompt(f"prompt_{i:03d}")
            prompts.append(prompt)
        return prompts


def main():
    """Main execution."""
    # Set paths
    output_dir = Path(__file__).parent.parent / "data" / "prompts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "prompts_text.json"

    # Generate prompts
    generator = PromptGenerator(seed=42)
    prompts = generator.generate_all(num_prompts=100)

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)

    print(f"Generated {len(prompts)} prompts")
    print(f"Saved to: {output_file}")
    print(f"\nExample prompts:")
    for i in range(3):
        print(f"  {prompts[i]['id']}: {prompts[i]['text']}")


if __name__ == "__main__":
    main()
