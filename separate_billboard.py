"""
Separates Billboard dataset files from the rest.

Section 1 — chord_data_1217:
  Reads all .jams in references_v2, identifies Billboard ones,
  then moves the corresponding .jams, audio, and .pt files to billboard_for_inference/.

Section 2 — final_augmentation:
  Reads all .jams in data_augmentation_jams, identifies Billboard ones,
  then moves the corresponding .jams and audio files to billboard_extraction/.
"""

import json
import shutil
from pathlib import Path


def is_billboard(jams_path: Path) -> bool:
    with open(jams_path, encoding="utf-8") as f:
        data = json.load(f)
    for annotation in data.get("annotations", []):
        corpus = annotation.get("annotation_metadata", {}).get("corpus", "")
        if corpus == "Billboard-Chords":
            return True
    return False


def move_file(src: Path, dst_dir: Path) -> bool:
    if src.exists():
        shutil.move(str(src), str(dst_dir / src.name))
        return True
    return False


# ---------------------------------------------------------------------------
# Section 1: chord_data_1217
# ---------------------------------------------------------------------------

def process_chord_data():
    print("=" * 60)
    print("Section 1: chord_data_1217")
    print("=" * 60)

    BASE = Path("chord_data_1217")
    JAMS_SRC = BASE / "references_v2"
    AUDIO_SRC = BASE / "audio"
    PT_SRC = BASE / "preprocessed_data_og"

    DST_JAMS = BASE / "billboard_for_inference" / "jams"
    DST_AUDIO = BASE / "billboard_for_inference" / "audio"
    DST_PT = BASE / "billboard_for_inference" / "preprocesed_billboard_data"

    for d in (DST_JAMS, DST_AUDIO, DST_PT):
        d.mkdir(parents=True, exist_ok=True)

    jams_files = sorted(JAMS_SRC.glob("*.jams"))
    print(f"Total JAMS files found: {len(jams_files)}")

    billboard_ids = [j.stem for j in jams_files if is_billboard(j)]
    print(f"Billboard tracks identified: {len(billboard_ids)}")

    moved_jams = moved_audio = moved_pt = 0
    missing_audio, missing_pt = [], []

    for track_id in billboard_ids:
        if move_file(JAMS_SRC / f"{track_id}.jams", DST_JAMS):
            moved_jams += 1

        audio_moved = False
        for ext in (".mp3", ".wav", ".flac", ".ogg"):
            if move_file(AUDIO_SRC / f"{track_id}{ext}", DST_AUDIO):
                moved_audio += 1
                audio_moved = True
                break
        if not audio_moved:
            missing_audio.append(track_id)

        pt_files = list(PT_SRC.glob(f"*{track_id}*.pt"))
        if pt_files:
            for pt in pt_files:
                shutil.move(str(pt), str(DST_PT / pt.name))
                moved_pt += 1
        else:
            missing_pt.append(track_id)

    print(f"\nMoved {moved_jams} JAMS  -> {DST_JAMS}")
    print(f"Moved {moved_audio} audio -> {DST_AUDIO}")
    print(f"Moved {moved_pt} .pt    -> {DST_PT}")

    if missing_audio:
        print(f"\n[WARN] No audio for {len(missing_audio)} tracks: {missing_audio}")
    if missing_pt:
        print(f"\n[WARN] No .pt files for {len(missing_pt)} tracks: {missing_pt}")


# ---------------------------------------------------------------------------
# Section 2: final_augmentation
# ---------------------------------------------------------------------------

def process_final_augmentation():
    print("\n" + "=" * 60)
    print("Section 2: final_augmentation")
    print("=" * 60)

    BASE = Path("final_augmentation")
    JAMS_SRC = BASE / "data_augmentation_jams"
    AUDIO_SRC = BASE / "data_augmentation_files"

    DST_JAMS = BASE / "billboard_extraction" / "jams"
    DST_AUDIO = BASE / "billboard_extraction" / "audio"

    for d in (DST_JAMS, DST_AUDIO):
        d.mkdir(parents=True, exist_ok=True)

    jams_files = sorted(JAMS_SRC.glob("*.jams"))
    print(f"Total JAMS files found: {len(jams_files)}")

    billboard_stems = [j.stem for j in jams_files if is_billboard(j)]
    print(f"Billboard files identified: {len(billboard_stems)}")

    moved_jams = moved_audio = 0
    missing_audio = []

    for stem in billboard_stems:
        if move_file(JAMS_SRC / f"{stem}.jams", DST_JAMS):
            moved_jams += 1

        audio_moved = False
        for ext in (".mp3", ".wav", ".flac", ".ogg"):
            if move_file(AUDIO_SRC / f"{stem}{ext}", DST_AUDIO):
                moved_audio += 1
                audio_moved = True
                break
        if not audio_moved:
            missing_audio.append(stem)

    print(f"\nMoved {moved_jams} JAMS  -> {DST_JAMS}")
    print(f"Moved {moved_audio} audio -> {DST_AUDIO}")

    if missing_audio:
        print(f"\n[WARN] No audio for {len(missing_audio)} files: {missing_audio[:10]}{'...' if len(missing_audio) > 10 else ''}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    process_chord_data()
    process_final_augmentation()
    print("\nAll done.")
