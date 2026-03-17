"""
Local ACE training wrapper with leakage-safe dataset splitting.

`GROUP_SUFFIXES` controls which filename suffixes should be treated as belonging to
the same underlying source track. This affects two places:

1. Train/validation splitting:
   files such as `billboard_song_add_noise_t0000_p+0.pt` and
   `billboard_song_t0000_p+0.pt` are grouped together before the split.
2. Validation/test filtering:
   when `augmentation=False`, files whose track IDs end with one of the configured
   suffixes are excluded from validation/test so those splits stay clean.

Override `GROUP_SUFFIXES` with gin:

    ChocoAudioDataModule.GROUP_SUFFIXES = (
        "add_noise",
        "apply_compression",
        "randomly_eq",
        "apply_reverb",
    )

Override `GROUP_SUFFIXES` from the notebook/API:

    ace_train_model(
        model_name="conformer",
        run_name="experiment",
        params={
            "ChocoAudioDataModule.GROUP_SUFFIXES": (
                "add_noise",
                "apply_compression",
                "randomly_eq",
                "apply_reverb",
            ),
        },
    )
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import gin
import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks.early_stopping import EarlyStopping as _EarlyStopping
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint as _ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

DEFAULT_GROUP_SUFFIXES = (
    "add_noise",
    "apply_compression",
    "randomly_eq",
    "apply_reverb",
)


def _track_id_from_file(file_path: Path) -> str:
    return "_".join(file_path.stem.split("_")[:-2])


def _glob_case_variants(data_path: Path, prefix: str) -> list[Path]:
    return sorted(
        {
            *data_path.glob(f"{prefix}*.pt"),
            *data_path.glob(f"{prefix.lower()}*.pt"),
        }
    )


def _glob_all_pt_files(data_path: Path) -> list[Path]:
    return sorted(data_path.glob("*.pt"))


def _group_suffix_regex(group_suffixes: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(
        rf"_(?:{'|'.join(re.escape(name) for name in group_suffixes)})$"
    )


def _canonical_track_id(track_id: str, group_suffixes: tuple[str, ...]) -> str:
    if not group_suffixes:
        return track_id
    return _group_suffix_regex(group_suffixes).sub("", track_id)


def _is_baked_augmentation(track_id: str, group_suffixes: tuple[str, ...]) -> bool:
    return _canonical_track_id(track_id, group_suffixes) != track_id


class LeakageSafeChocoAudioDataset(Dataset):
    """
    Dataset wrapper that optionally removes grouped augmentation variants.

    Parameters
    ----------
    GROUP_SUFFIXES:
        Tuple of suffixes that identify cached files derived from the same source
        track, for example `("add_noise", "apply_compression")`. These suffixes
        are matched at the end of the track ID, before `_tXXXX_p+Y`.
    """

    def __init__(
        self,
        data_path: Path,
        track_ids: list[str],
        data_dict: dict[str, list[Path]],
        GROUP_SUFFIXES: tuple[str, ...] = DEFAULT_GROUP_SUFFIXES,
        augmentation: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.augmentation = augmentation
        self.data_dict = data_dict
        self.group_suffixes = tuple(GROUP_SUFFIXES)
        self.data_list = self._get_data_list(track_ids)

    def __len__(self) -> int:
        return len(self.data_list)

    def _get_data_list(self, track_ids: list[str]) -> list[Path]:
        data_list: list[Path] = []
        for track_id in track_ids:
            data_list.extend(self.data_dict[track_id])

        if self.augmentation is False:
            data_list = [
                file_path
                for file_path in data_list
                if not _is_baked_augmentation(
                    _track_id_from_file(file_path), self.group_suffixes
                )
            ]
            data_list = [x for x in data_list if x.stem.endswith("_p+0")]
            data_list = [
                x
                for x in data_list
                if int(x.stem.split("_")[-2].replace("t", "")) % 20 == 0
            ]

        return data_list

    def __getitem__(self, index):
        return torch.load(
            self.data_list[index],
            weights_only=False,
            map_location="cpu",
        )


class _LeakageSafeChocoAudioDataModule(L.LightningDataModule):
    """
    Datamodule with source-aware splitting for cached ACE datasets.

    Parameters
    ----------
    GROUP_SUFFIXES:
        Tuple of track-ID suffixes that should be grouped with the original source
        when creating train/validation splits. Example:
        `("add_noise", "apply_compression", "randomly_eq", "apply_reverb")`.

        If a cached file is named `billboard_song_add_noise_t0000_p+0.pt`, the
        grouping key becomes `billboard_song`. This prevents one augmentation
        variant from appearing in train while another variant of the same song
        appears in validation.

        To disable suffix-based grouping entirely, pass an empty tuple `()`.
    """

    def __init__(
        self,
        data_path: str | Path,
        test_data_path: str | Path | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
        augmentation: bool = False,
        GROUP_SUFFIXES: tuple[str, ...] = DEFAULT_GROUP_SUFFIXES,
        train_ratio: float | None = 0.6,
        val_ratio: float | None = 0.5,
        random_seed: int | None = None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.test_data_path = Path(test_data_path) if test_data_path is not None else self.data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.group_suffixes = tuple(GROUP_SUFFIXES)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed

        self.vocabularies = {
            "simplified": 50,
            "root": 13,
            "bass": 13,
            "mode": 8,
            "majmin": 26,
            "onehot": 12,
            "complete": 170,
        }

        isophonics_files = _glob_case_variants(self.data_path, "Isophonics")
        billboard_files = _glob_case_variants(self.data_path, "Billboard")
        if test_data_path is None:
            test_files = _glob_case_variants(self.test_data_path, "MARL")
        else:
            test_files = _glob_all_pt_files(self.test_data_path)

        self.train_list = isophonics_files + billboard_files
        self.test_list = test_files

        self.data_dict_train = self._precompute_data_dict(self.train_list)
        self.data_dict_test = self._precompute_data_dict(self.test_list)

        print(
            f"Found {len(self.train_list)} training files "
            f"and {len(self.test_list)} test files."
        )
        print(f"Training data path: {self.data_path}")
        print(f"Test data path: {self.test_data_path}")
        print(f"Training IDs: {list(self.data_dict_train.keys())[:5]}... ")
        print(f"Test IDs: {list(self.data_dict_test.keys())[:5]}... ")

    def _precompute_data_dict(self, data_list: list[Path]) -> dict[str, list[Path]]:
        data_dict: dict[str, list[Path]] = defaultdict(list)
        for file_path in data_list:
            track_id = _track_id_from_file(file_path)
            data_dict[track_id].append(file_path)
        return data_dict

    def _group_track_ids_by_source(self, data_dict: dict[str, list[Path]]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for track_id in data_dict:
            grouped[_canonical_track_id(track_id, self.group_suffixes)].append(track_id)
        return grouped

    def _split_dataset(self):
        grouped_train = self._group_track_ids_by_source(self.data_dict_train)
        source_ids = list(grouped_train.keys())

        train_sources, val_sources = train_test_split(
            source_ids,
            train_size=self.train_ratio,
            random_state=self.random_seed,
            shuffle=True,
        )

        grouped_test = self._group_track_ids_by_source(self.data_dict_test)
        test_sources = list(grouped_test.keys())

        train_track_ids = sorted(
            track_id for source_id in train_sources for track_id in grouped_train[source_id]
        )
        val_track_ids = sorted(
            track_id for source_id in val_sources for track_id in grouped_train[source_id]
        )
        test_track_ids = sorted(
            track_id for source_id in test_sources for track_id in grouped_test[source_id]
        )

        print(
            "Leakage-safe split:"
            f" {len(train_sources)} train sources, {len(val_sources)} val sources,"
            f" {len(test_sources)} test sources"
        )

        return train_track_ids, val_track_ids, test_track_ids

    def prepare_data(self):
        self.train_track_ids, self.val_track_ids, self.test_track_ids = (
            self._split_dataset()
        )

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = LeakageSafeChocoAudioDataset(
                self.data_path,
                self.train_track_ids,
                self.data_dict_train,
                GROUP_SUFFIXES=self.group_suffixes,
                augmentation=self.augmentation,
            )
            self.val_dataset = LeakageSafeChocoAudioDataset(
                self.data_path,
                self.val_track_ids,
                self.data_dict_train,
                GROUP_SUFFIXES=self.group_suffixes,
                augmentation=False,
            )

        if stage == "test":
            self.test_dataset = LeakageSafeChocoAudioDataset(
                self.test_data_path,
                self.test_track_ids,
                self.data_dict_test,
                GROUP_SUFFIXES=self.group_suffixes,
                augmentation=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


@gin.configurable
def wandb_logger(project: str, name: str, group: str | None = None) -> WandbLogger:
    return WandbLogger(project=project, name=name, log_model=False, group=group)


@gin.configurable
class ModelCheckpoint(_ModelCheckpoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@gin.configurable
class EarlyStopping(_EarlyStopping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@gin.configurable
def train(
    model_class: type[L.LightningModule],
    data_path: str | Path,
    test_data_path: str | Path | None = None,
    run_name: str = "default_run",
    max_epochs: int = 100,
    precision: str = "16-mixed",
    accelerator: str = "gpu",
    devices: int = 1,
    log_every_n_steps: int = 10,
    check_val_every_n_epoch: int = 10,
    accumulate_grad_batches: int = 1,
    gradient_clip_val: float = 0.0,
    vocab_path: str | Path = "./ACE/chords_vocab.joblib",
):
    torch.set_float32_matmul_precision("medium")

    checkpoint_callback = gin.get_configurable(ModelCheckpoint)()
    early_stop_callback = gin.get_configurable(EarlyStopping)()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = wandb_logger(name=run_name)  # type: ignore[arg-type]
    datamodule = ChocoAudioDataModule(
        data_path=data_path,
        test_data_path=test_data_path,
    )

    print(f"Train data path: {datamodule.data_path}")
    print(f"Test data path: {datamodule.test_data_path}")

    model = gin.get_configurable(model_class)(
        vocabularies=datamodule.vocabularies, vocab_path=vocab_path
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,  # type: ignore[arg-type]
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        deterministic=True,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    wandb.finish()


@gin.configurable
def test(
    model_class: type[L.LightningModule],
    data_path: str | Path,
    checkpoint_path: str | Path,
    test_data_path: str | Path | None = None,
    run_name: str = "default_run",
    precision: str = "16-mixed",
    accelerator: str = "gpu",
    devices: int = 1,
    vocab_path: str | Path = "./ACE/chords_vocab.joblib",
):
    torch.set_float32_matmul_precision("medium")

    logger = wandb_logger(name=run_name)  # type: ignore[arg-type]
    datamodule = ChocoAudioDataModule(
        data_path=data_path,
        test_data_path=test_data_path,
    )

    print(f"Train data path: {datamodule.data_path}")
    print(f"Test data path: {datamodule.test_data_path}")
    print(f"Checkpoint path: {checkpoint_path}")

    model = gin.get_configurable(model_class)(
        vocabularies=datamodule.vocabularies, vocab_path=vocab_path
    )

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,  # type: ignore[arg-type]
        logger=logger,
        deterministic=True,
    )

    trainer.test(model=model, datamodule=datamodule, ckpt_path=str(checkpoint_path))
    wandb.finish()


def bind_overrides(params: dict | None):
    if params is None:
        return
    for key, value in params.items():
        if value is not None:
            gin.bind_parameter(key, value)


@gin.configurable
class ChocoAudioDataModule(_LeakageSafeChocoAudioDataModule):
    """
    Gin-compatible alias that preserves the upstream configurable name.

    Use `ChocoAudioDataModule.GROUP_SUFFIXES` in gin or in the `params` override
    dictionary when calling `main(...)`.
    """


LeakageSafeChocoAudioDataModule = ChocoAudioDataModule


@gin.configurable
def main(
    model_name: str,
    run_name: str,
    mode: str = "train",
    checkpoint_path: str | Path | None = None,
    params: dict | None = None,
):
    gin.clear_config()
    gin.parse_config_file("ACE/trainer.gin")

    from ACE.models.conformer import ConformerModel
    from ACE.models.conformer_decomposed import ConformerDecomposedModel

    model_registry = {
        "conformer": ConformerModel,
        "conformer_decomposed": ConformerDecomposedModel,
    }

    model_class = model_registry.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")

    bind_overrides(params)
    if mode == "train":
        train(model_class=model_class, run_name=run_name)
        return
    if mode == "test":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when mode='test'")
        test(
            model_class=model_class,
            run_name=run_name,
            checkpoint_path=checkpoint_path,
        )
        return
    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--vocab_path", type=str)
    parser.add_argument("--accelerator", type=str, choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--precision", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--checkpoint_save_top_k", type=int)
    parser.add_argument("--checkpoint_monitor", type=str)
    parser.add_argument("--earlystop_patience", type=int)
    parser.add_argument("--earlystop_monitor", type=str)
    args = parser.parse_args()

    cli_overrides = {
        "train.data_path": args.data_path,
        "train.test_data_path": args.test_data_path,
        "test.data_path": args.data_path,
        "test.test_data_path": args.test_data_path,
        "train.vocab_path": args.vocab_path,
        "test.vocab_path": args.vocab_path,
        "train.max_epochs": args.max_epochs,
        "train.precision": args.precision,
        "test.precision": args.precision,
        "train.accelerator": args.accelerator,
        "test.accelerator": args.accelerator,
        "ModelCheckpoint.dirpath": args.checkpoint_dir,
        "ModelCheckpoint.save_top_k": args.checkpoint_save_top_k,
        "ModelCheckpoint.monitor": args.checkpoint_monitor,
        "EarlyStopping.patience": args.earlystop_patience,
        "EarlyStopping.monitor": args.earlystop_monitor,
    }

    main(
        model_name=args.model,
        run_name=args.name,
        mode=args.mode,
        checkpoint_path=args.checkpoint_path,
        params=cli_overrides,
    )
