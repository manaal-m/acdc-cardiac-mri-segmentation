# Checkpoints

Trained model weights are not committed to this repository (they're too large for git).

To reproduce the results in `results/all_results.json`:

1. Run `notebooks/01_train_part1.ipynb` and `notebooks/02_train_part2.ipynb`. This will train all four encoders across 3 seeds (0, 42, 11) and save the PyTorch Lightning checkpoints (`.ckpt`) into this folder automatically.

   **OR**

2. Download our pretrained checkpoints from [this Kaggle Dataset](https://www.kaggle.com/datasets/manaalmay/acdc-multiseed-checkpoints), and extract the zipped `.ckpt` files here:
   - `resnet18` checkpoints (seeds 0, 42, 11)
   - `vgg16` checkpoints (seeds 0, 42, 11)
   - `mobilenet_v2` checkpoints (seeds 0, 42, 11)
   - `resnet50` checkpoints (seeds 0, 42, 11)

*(Make sure they are unzipped into `.ckpt` files in this directory before proceeding).*

Then run `notebooks/03_compile_and_evaluate.ipynb` to evaluate the ensemble, measure CPU/GPU latency, and regenerate the figures and per-class Dice numbers.
