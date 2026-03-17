from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def _slugify(value: str) -> str:
    """Create filesystem-friendly suffixes for IR names."""
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()


def _iter_audio_files(folder: Path, extensions: tuple[str, ...]) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions
    )


def _iter_ir_files(folder: Path, extensions: tuple[str, ...]) -> list[Path]:
    excluded_dirs = {"examples", "images", "__macosx"}
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensions
        and not any(part.lower() in excluded_dirs for part in p.parts)
    )


def _copy_originals(input_dir: Path, output_dir: Path, extensions: tuple[str, ...]) -> int:
    copied = 0
    for audio_path in _iter_audio_files(input_dir, extensions):
        target = output_dir / audio_path.name
        if not target.exists():
            shutil.copy2(audio_path, target)
            copied += 1
    return copied


def generate_reverb_augments(
    input_dir: Path,
    ir_dir: Path,
    output_dir: Path,
    wet_dry_mix: float,
    sample_rate: int,
    audio_extensions: tuple[str, ...],
    ir_extensions: tuple[str, ...],
    copy_originals: bool = False,
) -> tuple[int, int]:
    """Create one reverberated audio file per (input audio, IR) pair."""
    from data_augmentation import apply_reverb

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = _iter_audio_files(input_dir, audio_extensions)
    ir_files = _iter_ir_files(ir_dir, ir_extensions)

    if not audio_files:
        raise FileNotFoundError(f"No input audio files found in {input_dir}")
    if not ir_files:
        raise FileNotFoundError(f"No IR files found in {ir_dir}")

    copied = _copy_originals(input_dir, output_dir, audio_extensions) if copy_originals else 0

    created = 0
    for audio_path in audio_files:
        for ir_path in ir_files:
            relative_parent = ir_path.parent.relative_to(ir_dir)
            parent_suffix = _slugify(str(relative_parent)) if str(relative_parent) != "." else ""
            ir_suffix = _slugify(ir_path.stem)
            if parent_suffix:
                ir_suffix = f"{parent_suffix}-{ir_suffix}"
            output_stem = f"{audio_path.stem}_reverb_{ir_suffix}"
            output_path = output_dir / f"{output_stem}.mp3"

            apply_reverb(
                str(audio_path),
                str(output_path),
                {
                    "ir_path": str(ir_path),
                    "sr": sample_rate,
                    "wet_dry_mix": wet_dry_mix,
                },
            )
            created += 1
            print(f"[OK] {audio_path.name} + {ir_path.name} -> {output_path.name}")

    return copied, created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reverberated audio files from a folder of impulse responses."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("test_audio"),
        help="Folder containing the clean source audio files.",
    )
    parser.add_argument(
        "--ir-dir",
        type=Path,
        required=True,
        help="Folder containing impulse response audio files (.wav, .flac, .mp3, .aiff).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("augmentations_test_outputs"),
        help="Destination folder for the generated reverb examples.",
    )
    parser.add_argument(
        "--wet-dry-mix",
        type=float,
        default=0.3,
        help="Wet/dry ratio for convolution reverb. 0.0=dry, 1.0=fully wet.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate used when loading the audio and IR files.",
    )
    parser.add_argument(
        "--copy-originals",
        action="store_true",
        help="Also copy the clean input files into the output directory if missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input audio directory not found: {args.input_dir}")
    if not args.ir_dir.exists():
        raise FileNotFoundError(f"IR directory not found: {args.ir_dir}")

    copied, created = generate_reverb_augments(
        input_dir=args.input_dir,
        ir_dir=args.ir_dir,
        output_dir=args.output_dir,
        wet_dry_mix=args.wet_dry_mix,
        sample_rate=args.sample_rate,
        audio_extensions=(".mp3", ".wav", ".flac"),
        ir_extensions=(".wav", ".flac", ".mp3", ".aiff", ".aif"),
        copy_originals=args.copy_originals,
    )

    print("\n=== Reverb augmentation summary ===")
    print(f"Copied originals: {copied}")
    print(f"Created reverberated files: {created}")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
