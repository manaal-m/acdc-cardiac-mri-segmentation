# Checkpoints

Trained model weights are not committed to this repository (they're too
large for git, and `.gitignore` excludes `*.pth` files here).

To reproduce the results in `results/all_results.json`:

1. Run `notebooks/01_train.ipynb` end to end. It will train all four
   encoders and save `<encoder>_best.pth` into this folder automatically.

   **OR**

2. Download our pretrained checkpoints from [this Kaggle Dataset](https://www.kaggle.com/datasets/manaalmay/acdc-model-checkpoints), and place the four files
   here:
   - `resnet18_best.pth`
   - `vgg16_best.pth`
   - `mobilenet_v2_best.pth`
   - `resnet50_best.pth`

Then run `notebooks/02_evaluate.ipynb` to regenerate the figures and
per-slice latency / per-class Dice numbers.
