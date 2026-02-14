from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    train_images_dir: Path
    train_masks_dir: Path
    val_images_dir: Path | None = None
    val_masks_dir: Path | None = None
    test_images_dir: Path | None = None
    test_masks_dir: Path | None = None
    val_split: float = 0.2
    split_by_patient: bool = True
    image_size: int = 256
    batch_size: int = 8
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    random_seed: int = 42
    mixed_precision: bool = True
    early_stopping_patience: int = 7
    use_augmentation: bool = True
    loss: str = "bce_dice"
    dice_weight: float = 0.5
    output_dir: Path = Path("artifacts")

    def __post_init__(self) -> None:
        self.train_images_dir = Path(self.train_images_dir)
        self.train_masks_dir = Path(self.train_masks_dir)
        self.val_images_dir = Path(self.val_images_dir) if self.val_images_dir else None
        self.val_masks_dir = Path(self.val_masks_dir) if self.val_masks_dir else None
        self.test_images_dir = Path(self.test_images_dir) if self.test_images_dir else None
        self.test_masks_dir = Path(self.test_masks_dir) if self.test_masks_dir else None
        self.output_dir = Path(self.output_dir)
