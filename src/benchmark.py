"""
Latency benchmarking and results file I/O.

This module is the single place that defines the results schema. Both
notebooks/01_train.ipynb and notebooks/02_evaluate.ipynb read and write
through these functions so that every table and figure in the repo is
generated from the same JSON file on disk -- nothing is ever hand-typed
into a notebook cell again.

Schema written to results/all_results.json:
{
  "<encoder>": {
      "encoder": str,
      "dice": float,                       # macro Dice on the held-out test set
      "iou": float,                        # macro mIoU on the held-out test set
      "model_size_mb": float,               # size of the saved state_dict on disk
      "inference_ms_batch8_mean": float,    # latency per forward pass, batch size 8
      "inference_ms_per_slice_mean": float, # latency per forward pass, batch size 1
      "inference_ms_per_slice_median": float,
      "inference_ms_per_slice_std": float,
  },
  ...
}
"""

import json
import os
import time

import numpy as np
import torch


def measure_batch_latency(model, dataloader, device="cuda", n_batches=50):
    """Time `n_batches` forward passes through `dataloader` (whatever batch
    size the dataloader was built with) and return the mean latency in ms.

    Uses wall-clock timing with cuda synchronize -- adequate for a coarse
    batch-level comparison. For precise single-sample timing use
    measure_per_slice_latency instead.
    """
    model.eval()
    times = []
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            x = x.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)
    return float(np.mean(times))


def measure_per_slice_latency(model, dataset, device="cuda", n_warmup=20, n_measure=100):
    """Time individual (batch_size=1) forward passes using CUDA events, which
    are more accurate than wall-clock time.time() for GPU timing.

    Args:
        model: a model already moved to `device` and set to eval().
        dataset: a torch Dataset (NOT a DataLoader) of (image, mask) pairs.
        n_warmup: number of initial forward passes discarded (GPU warm-up).
        n_measure: number of timed forward passes after warm-up.

    Returns:
        dict with keys mean, median, std (all in ms), and n (slices actually timed).
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    times = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)

            if i < n_warmup:
                _ = model(x)
                continue

            if device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
                _ = model(x)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
            else:
                start = time.time()
                _ = model(x)
                times.append((time.time() - start) * 1000)

            if len(times) >= n_measure:
                break

    return {
        "mean": round(float(np.mean(times)), 3),
        "median": round(float(np.median(times)), 3),
        "std": round(float(np.std(times)), 3),
        "n": len(times),
    }


def model_size_mb(model, tmp_path="/tmp/_size_check.pth"):
    """Save a model's state_dict to disk and return its size in MB."""
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return round(size_mb, 2)


def load_results(path):
    """Load the results dict from disk, or return {} if it doesn't exist yet."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_results(results, path):
    """Write the results dict to disk as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def update_result(results, encoder, **fields):
    """Merge `fields` into results[encoder], creating the entry if needed.
    Mutates and returns `results`."""
    entry = results.setdefault(encoder, {"encoder": encoder})
    entry.update(fields)
    return results
