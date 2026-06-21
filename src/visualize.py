"""
Plotting helpers for qualitative segmentation comparisons and quantitative
metric bar charts. Every function here takes data as an argument -- nothing
is hardcoded -- so figures regenerate correctly if results/all_results.json
changes.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

COLORS = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 0, 255], 3: [0, 255, 0]}
LABELS = {0: "Background", 1: "LV Cavity", 2: "RV Cavity", 3: "Myocardium"}


def mask_to_rgb(mask):
    """Convert an integer class mask (H, W) to an RGB image (H, W, 3) using COLORS."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        rgb[mask == cls] = color
    return rgb


def tight_crop(img, mask, pad=20):
    """Crop `img` and `mask` to a tight bounding box around the foreground
    (mask > 0) plus `pad` pixels of margin. Returns (img, mask) unchanged if
    the mask is empty."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return img, mask
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def find_best_slice(dataset, target_classes=(1, 2, 3)):
    """Find the index of the first slice in `dataset` that contains all of
    `target_classes` (used to pick a representative slice for the
    qualitative comparison figure)."""
    best_idx, best_score = None, 0
    for i in range(len(dataset)):
        _, mask = dataset[i]
        present = set(torch.unique(mask).tolist())
        score = len([c for c in target_classes if c in present])
        if score > best_score:
            best_idx, best_score = i, score
        if best_score == len(target_classes):
            break
    return best_idx


def plot_segmentation_comparison(
    models_by_encoder,
    sample,
    dice_by_encoder,
    encoders,
    device="cuda",
    save_path=None,
):
    """Grid figure: one row per encoder, columns = [input MRI, ground truth, prediction].

    Args:
        models_by_encoder: dict {encoder_name: loaded eval-mode model}.
        sample: (image_tensor, mask_tensor) single example from the dataset
            (image_tensor is (1, H, W), mask_tensor is (H, W)).
        dice_by_encoder: dict {encoder_name: float} test-set Dice, shown under
            each row's prediction column.
        encoders: ordered list of encoder names (row order).
        save_path: if given, the figure is saved here (dpi=150).
    """
    x, y = sample
    x_input = x.unsqueeze(0).to(device)
    img = x.squeeze().numpy()
    gt = y.numpy()
    img_crop, gt_crop = tight_crop(img, gt)

    fig, axes = plt.subplots(len(encoders), 3, figsize=(10, 3.5 * len(encoders)))
    fig.suptitle(
        "Cardiac MRI Segmentation: Ground Truth vs Model Predictions",
        fontsize=13,
        fontweight="bold",
    )

    for col, title in enumerate(["Input MRI", "Ground Truth", "Prediction"]):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold")

    for row, encoder in enumerate(encoders):
        model = models_by_encoder[encoder]
        with torch.no_grad():
            pred = torch.argmax(model(x_input), dim=1).squeeze().cpu().numpy()
        _, pred_crop = tight_crop(img, pred)

        axes[row][0].imshow(img_crop, cmap="gray")
        axes[row][1].imshow(mask_to_rgb(gt_crop))
        axes[row][2].imshow(mask_to_rgb(pred_crop))

        for col in range(3):
            axes[row][col].axis("off")

        axes[row][0].text(
            -0.15, 0.5, encoder, transform=axes[row][0].transAxes,
            fontsize=10, fontweight="bold", va="center", ha="center", rotation=90,
        )
        axes[row][2].text(
            0.5, -0.08, f"Dice: {dice_by_encoder[encoder]:.4f}",
            transform=axes[row][2].transAxes, fontsize=9, va="top", ha="center",
        )

    patches = [mpatches.Patch(color=np.array(c) / 255, label=l) for c, l in zip(COLORS.values(), LABELS.values())]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_comparison(results, encoders, latency_key="inference_ms_per_slice_mean", save_path=None):
    """2x2 bar chart: Dice, mIoU, inference latency, model size -- one bar
    per encoder, read directly from the `results` dict (see
    src.benchmark for the schema) so the figure can never drift from
    results/all_results.json.

    Args:
        results: the loaded all_results.json dict.
        encoders: ordered list of encoder names (x-axis order).
        latency_key: which latency field to plot -- defaults to the
            per-slice (batch_size=1) mean. Pass "inference_ms_batch8_mean"
            to plot batch-8 latency instead.
    """
    display_names = {
        "resnet18": "ResNet18",
        "vgg16": "VGG16",
        "mobilenet_v2": "MobileNetV2",
        "resnet50": "ResNet50",
    }
    models = [display_names.get(e, e) for e in encoders]
    dice = [results[e]["dice"] for e in encoders]
    miou = [results[e]["iou"] for e in encoders]
    latency = [results[e][latency_key] for e in encoders]
    size = [results[e]["model_size_mb"] for e in encoders]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(encoders)]
    x = np.arange(len(models))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Encoder Architecture Comparison on ACDC Dataset", fontsize=14, fontweight="bold")

    def bar_plot(ax, values, title, ylabel, highlight_max=True):
        bars = ax.bar(x, values, color=colors, width=0.5, edgecolor="white")
        best = values.index(max(values) if highlight_max else min(values))
        bars[best].set_edgecolor("black")
        bars[best].set_linewidth(2)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001 * max(values),
                f"{val}", ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylim(min(values) * 0.97, max(values) * 1.05)

    latency_label = "ms/slice" if "per_slice" in latency_key else "ms/batch (8 slices)"

    bar_plot(axes[0][0], dice, "Dice Score (\u2191)", "Dice")
    bar_plot(axes[0][1], miou, "Mean IoU (\u2191)", "mIoU")
    bar_plot(axes[1][0], latency, "Inference Latency (\u2193)", latency_label, highlight_max=False)
    bar_plot(axes[1][1], size, "Model Size MB (\u2193)", "MB", highlight_max=False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
