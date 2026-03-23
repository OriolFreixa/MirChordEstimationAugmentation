from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_augmented_jams(
    augmented_audio_dir: Path,
    source_jams_dir: Path,
    output_jams_dir: Path,
    audio_extensions: tuple[str, ...] = (".mp3", ".wav", ".flac"),
) -> tuple[int, int, int]:
    """Create JAMS files for augmented audio by copying original JAMS annotations.

    Returns:
        (created_count, overwritten_count, missing_original_count)
    """
    output_jams_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    overwritten = 0
    missing = 0

    audio_files = sorted(
        p for p in augmented_audio_dir.iterdir() if p.is_file() and p.suffix.lower() in audio_extensions
    )

    for audio_file in audio_files:
        # Example augmented stem: TRYNQFC149E3E14844_add_noise
        # Original JAMS stem expected: TRYNQFC149E3E14844
        augmented_stem = audio_file.stem
        original_stem = augmented_stem.split("_")[0]

        source_jams = source_jams_dir / f"{original_stem}.jams"
        target_jams = output_jams_dir / f"{augmented_stem}.jams"

        if not source_jams.exists():
            missing += 1
            print(f"[MISSING] {source_jams}")
            continue

        if target_jams.exists():
            overwritten += 1
        else:
            created += 1

        shutil.copy2(source_jams, target_jams)

    return created, overwritten, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create augmented JAMS files by mapping augmented audio names to original JAMS IDs."
    )
    parser.add_argument(
        "--augmented-audio-dir",
        type=Path,
        default=Path("augmentations_test_outputs"),
        help="Folder containing augmented audio files.",
    )
    parser.add_argument(
        "--source-jams-dir",
        type=Path,
        default=Path("jams"),
        help="Folder containing original JAMS files.",
    )
    parser.add_argument(
        "--output-jams-dir",
        type=Path,
        default=Path("training_plus_augmentaiton") / "augmented_jams",
        help="Folder where generated augmented JAMS files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.augmented_audio_dir.exists():
        raise FileNotFoundError(f"Augmented audio directory not found: {args.augmented_audio_dir}")
    if not args.source_jams_dir.exists():
        raise FileNotFoundError(f"Source JAMS directory not found: {args.source_jams_dir}")

    created, overwritten, missing = build_augmented_jams(
        augmented_audio_dir=args.augmented_audio_dir,
        source_jams_dir=args.source_jams_dir,
        output_jams_dir=args.output_jams_dir,
    )

    total = created + overwritten + missing
    print("\n=== JAMS generation summary ===")
    print(f"Total augmented audio files scanned: {total}")
    print(f"Created: {created}")
    print(f"Overwritten: {overwritten}")
    print(f"Missing originals: {missing}")
    print(f"Output folder: {args.output_jams_dir}")


if __name__ == "__main__":
    main()
