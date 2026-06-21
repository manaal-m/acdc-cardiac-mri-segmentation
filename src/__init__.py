"""ACDC cardiac MRI segmentation -- encoder backbone comparison.

Importable building blocks used by both notebooks/01_train.ipynb and
notebooks/02_evaluate.ipynb:

    src.dataset    -- ACDC loading, preprocessing, Lightning DataModule
    src.model      -- SMP U-Net wrapper + LightningModule
    src.losses     -- Dice/CE loss, Dice score, per-class Dice
    src.benchmark  -- latency timing, results.json read/write
    src.visualize  -- comparison grid + metrics bar chart plotting
"""
