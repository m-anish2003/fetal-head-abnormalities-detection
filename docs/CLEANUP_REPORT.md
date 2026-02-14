# Project Cleanup Report

## Necessary features kept
- Paired image-mask dataset handling with strict filename alignment checks.
- Baseline UNet segmentation model for reproducible training.
- Dice and IoU metrics implemented in code and used in validation loop.
- Data quality controls: image-extension filtering, hidden/system-file exclusion, train/val split with seed.

## Problems detected and fixed
1. **Hardcoded Windows-only paths** in data scripts prevented reproducible runs.
   - Replaced with CLI arguments and repository-relative defaults.
2. **System artifacts mixed with dataset** (`Thumbs.db`) could pollute loaders.
   - All loaders/scripts now filter strictly by image extension.
3. **Notebook-only workflow** reduced production readiness.
   - Introduced a Python package (`src/fhad`) with train/evaluate entry modules.
4. **No early stopping in baseline path** can hurt generalization and Dice/IoU.
   - Added validation-based early stopping and best-checkpoint saving.

## Legacy/non-production components
- `notebooks/*.ipynb` remain useful for experimentation but are not required for production training.
- `notebooks/01_organize_dataset.py` duplicates script logic and is superseded by `scripts/organize_dataset.py`.
- `Untitled.ipynb` and `.ipynb_checkpoints/` are non-essential artifacts and should be removed in a future cleanup commit.

## Why these changes help IoU/Dice
- Better data integrity (correct pairing + extension filtering) reduces noisy supervision.
- Simple augmentations increase robustness and reduce overfitting.
- Validation monitoring + early stopping preserve the best generalizing checkpoint.
- Consistent preprocessing and deterministic splitting improve reproducibility of metric gains.
