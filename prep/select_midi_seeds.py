"""
Select 100 MIDI seed clips (8-16 bars) from MAESTRO and POP909 datasets.
These seeds are used for continuation and style-adherence tasks.
"""
import json
import random
from pathlib import Path
import shutil


class MIDISeedSelector:
    """Select representative MIDI seeds from datasets."""

    def __init__(self, maestro_path=None, pop909_path=None, seed=42):
        """
        Initialize selector with dataset paths.

        Args:
            maestro_path: Path to MAESTRO dataset root
            pop909_path: Path to POP909 dataset root
            seed: Random seed for reproducibility
        """
        self.maestro_path = Path(maestro_path) if maestro_path else None
        self.pop909_path = Path(pop909_path) if pop909_path else None
        random.seed(seed)

    def scan_maestro(self):
        """Scan MAESTRO directory for MIDI files."""
        if not self.maestro_path or not self.maestro_path.exists():
            print(f"Warning: MAESTRO path not found: {self.maestro_path}")
            return []

        midi_files = list(self.maestro_path.rglob("*.midi")) + \
                     list(self.maestro_path.rglob("*.mid"))
        print(f"Found {len(midi_files)} MAESTRO MIDI files")
        return midi_files

    def scan_pop909(self):
        """Scan POP909 directory for MIDI files."""
        if not self.pop909_path or not self.pop909_path.exists():
            print(f"Warning: POP909 path not found: {self.pop909_path}")
            return []

        midi_files = list(self.pop909_path.rglob("*.midi")) + \
                     list(self.pop909_path.rglob("*.mid"))
        print(f"Found {len(midi_files)} POP909 MIDI files")
        return midi_files

    def select_seeds(self, num_maestro=50, num_pop909=50):
        """
        Select seed MIDI files.

        Args:
            num_maestro: Number of MAESTRO seeds
            num_pop909: Number of POP909 seeds

        Returns:
            List of selected seed metadata
        """
        maestro_files = self.scan_maestro()
        pop909_files = self.scan_pop909()

        # Sample from available files
        selected_maestro = random.sample(
            maestro_files,
            min(num_maestro, len(maestro_files))
        )
        selected_pop909 = random.sample(
            pop909_files,
            min(num_pop909, len(pop909_files))
        )

        # Create metadata
        seeds = []
        seed_id = 0

        for midi_file in selected_maestro:
            seeds.append({
                "id": f"seed_{seed_id:03d}",
                "source": "maestro",
                "path": str(midi_file),
                "filename": midi_file.name,
                "genre": "classical"
            })
            seed_id += 1

        for midi_file in selected_pop909:
            seeds.append({
                "id": f"seed_{seed_id:03d}",
                "source": "pop909",
                "path": str(midi_file),
                "filename": midi_file.name,
                "genre": "pop"
            })
            seed_id += 1

        return seeds

    def copy_seeds(self, seeds, output_dir):
        """Copy selected MIDI files to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            src = Path(seed["path"])
            if src.exists():
                dest = output_dir / seed["source"] / f"{seed['id']}.mid"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                seed["output_path"] = str(dest)
            else:
                print(f"Warning: Source file not found: {src}")

        return seeds


def main():
    """Main execution."""
    # Configuration
    # These paths should be updated to point to actual dataset locations
    maestro_path = Path("./datasets/maestro-v3.0.0")
    pop909_path = Path("./datasets/POP909")

    output_dir = Path(__file__).parent.parent / "data" / "midi_seeds"
    metadata_file = Path(__file__).parent.parent / "data" / "midi_seeds.json"

    # Check if datasets exist
    if not maestro_path.exists():
        print(f"MAESTRO dataset not found at {maestro_path}")
        print("Please download from: https://magenta.withgoogle.com/datasets/maestro")
        print("Creating placeholder structure...")
        maestro_files_available = False
    else:
        maestro_files_available = True

    if not pop909_path.exists():
        print(f"POP909 dataset not found at {pop909_path}")
        print("Please download from: https://github.com/music-x-lab/POP909-Dataset")
        print("Creating placeholder structure...")
        pop909_files_available = False
    else:
        pop909_files_available = True

    # Initialize selector
    selector = MIDISeedSelector(
        maestro_path=maestro_path if maestro_files_available else None,
        pop909_path=pop909_path if pop909_files_available else None,
        seed=42
    )

    # Select seeds
    seeds = selector.select_seeds(num_maestro=50, num_pop909=50)

    # If datasets available, copy files
    if maestro_files_available or pop909_files_available:
        seeds = selector.copy_seeds(seeds, output_dir)

    # Save metadata
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(seeds, f, indent=2)

    print(f"\nSelected {len(seeds)} MIDI seeds")
    print(f"Metadata saved to: {metadata_file}")

    if not (maestro_files_available or pop909_files_available):
        print("\n" + "="*60)
        print("IMPORTANT: No dataset files were found.")
        print("Please download the datasets and rerun this script.")
        print("="*60)


if __name__ == "__main__":
    main()
