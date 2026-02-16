"""
Evaluate trained models on the test split.

Supports all modes: classification (single/multi-param), regression (single/multi/triple).
Optionally generates scatter plots (regression) or confusion matrices (classification).

Usage:
    # From experiment name (loads config and checkpoint automatically):
    python scripts/evaluate.py --name exp01_baseline
    python scripts/evaluate.py --name exp01_baseline --plot

    # Manual override:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint path/to/best_model.pt
"""

import argparse
import math
import os
import glob
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml
from tqdm import tqdm

from tape_id.models.encoder import SpectralEncoder
from tape_id.models.controller import ParameterController
from tape_id.data.dataset import hard_clipping, tape_saturation, ja_saturation, wow_flutter
from tape_id.utils import split_dataset, conform_length, linear_fade


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config: dict, checkpoint_path: str, device: str = "cpu"):
    """Load encoder + controller from config and checkpoint.

    Args:
        config: Experiment config dict.
        checkpoint_path: Path to .pt checkpoint.
        device: Device to load on.

    Returns:
        (encoder, controller, epoch)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder = SpectralEncoder(
        num_params=1,
        sample_rate=config["sample_rate"],
        embed_dim=config["embed_dim"],
        width_mult=2,
    ).to(device)

    degradation_model = config.get("degradation_model", "ja")
    regression = config.get("regression", False)
    triple_param = (degradation_model == "ja_wf")
    wf_target_param = config.get("wf_target_param", "depth")

    if triple_param:
        controller = ParameterController(
            embed_dim=config["embed_dim"], hidden_dim=config["hidden_dim"],
            regression=True, triple_param=True,
        )
    elif wf_target_param == "both":
        controller = ParameterController(
            embed_dim=config["embed_dim"], hidden_dim=config["hidden_dim"],
            num_classes_depth=config["num_classes_depth"],
            num_classes_rate=config["num_classes_rate"],
            regression=regression,
        )
    else:
        controller = ParameterController(
            num_classes=config.get("num_classes", 10),
            embed_dim=config["embed_dim"], hidden_dim=config["hidden_dim"],
            regression=regression,
        )

    controller = controller.to(device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    controller.load_state_dict(checkpoint["controller_state"])
    encoder.eval()
    controller.eval()

    return encoder, controller, checkpoint.get("epoch", "?")


# ---------------------------------------------------------------------------
# Test file discovery
# ---------------------------------------------------------------------------

def get_test_files(audio_dir, input_dirs=None, ext="mp3", train_frac=0.8):
    """Get test split files using the same deterministic split as the dataset."""
    filepaths = []
    if input_dirs is None:
        for entry in sorted(os.listdir(audio_dir)):
            subdir = os.path.join(audio_dir, entry)
            if os.path.isdir(subdir):
                filepaths += glob.glob(os.path.join(subdir, f"*.{ext}"))
        filepaths += glob.glob(os.path.join(audio_dir, f"*.{ext}"))
    else:
        for input_dir in input_dirs:
            search_path = os.path.join(audio_dir, input_dir, f"*.{ext}")
            filepaths += glob.glob(search_path)
        if len(filepaths) == 0:
            filepaths += glob.glob(os.path.join(audio_dir, f"*.{ext}"))
    filepaths = sorted(filepaths)

    rng_split = random.Random(42)
    rng_split.shuffle(filepaths)
    return split_dataset(filepaths, "test", train_frac)


# ---------------------------------------------------------------------------
# Audio and degradation helpers
# ---------------------------------------------------------------------------

def _load_and_prepare_audio(fpath, sample_rate, audio_length):
    """Load an audio file and prepare it for evaluation."""
    audio, sr = torchaudio.load(fpath)
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0)
    else:
        audio = audio.squeeze(0)
    if audio.shape[0] > audio_length:
        audio = audio[:audio_length]
    audio = audio / (audio.abs().max() + 1e-8)
    return audio


def _apply_degradation(x, param, config, wow_rate_override=None):
    """Apply the configured degradation to audio."""
    model = config["degradation_model"]
    sample_rate = config["sample_rate"]

    if model == "wow_flutter":
        depth = param
        rate = config.get("wow_rate", 0.4)
        if wow_rate_override is not None:
            rate = wow_rate_override
        elif config.get("wf_target_param") == "rate":
            depth = config.get("wf_fixed_depth", 0.5)
            rate = param
        return wow_flutter(x, depth, sample_rate=sample_rate,
                           wow_rate=rate,
                           flutter_rate=config.get("flutter_rate", 0.5),
                           enable_ou=config.get("enable_ou", True),
                           interpolation=config.get("wf_interpolation", "linear"))
    elif model == "hard_clipping":
        return hard_clipping(x, param)
    elif model == "ja":
        return ja_saturation(x, param, sample_rate=sample_rate)
    else:
        return tape_saturation(x, param)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_regression_metrics(true_arr, pred_arr, min_p, max_p, num_bins=5):
    """Compute regression metrics: MAE, RMSE, R², and per-bin breakdown.

    Returns:
        dict with keys: mae, rmse, r2, max_error, median_error, bins
    """
    errors = pred_arr - true_arr
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = math.sqrt(np.mean(errors ** 2))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((true_arr - np.mean(true_arr)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Per-bin breakdown
    bin_edges = np.linspace(min_p, max_p, num_bins + 1)
    bins = []
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (true_arr >= lo) & (true_arr < hi if i < num_bins - 1 else true_arr <= hi)
        n = mask.sum()
        if n > 0:
            bins.append({
                "lo": lo, "hi": hi, "n": int(n),
                "mae": float(np.mean(abs_errors[mask])),
                "rmse": float(math.sqrt(np.mean(errors[mask] ** 2))),
            })
        else:
            bins.append({"lo": lo, "hi": hi, "n": 0, "mae": 0.0, "rmse": 0.0})

    return {
        "mae": float(mae), "rmse": float(rmse), "r2": float(r2),
        "max_error": float(np.max(abs_errors)),
        "median_error": float(np.median(abs_errors)),
        "bins": bins,
    }


def print_regression_metrics(metrics, name, min_p, max_p):
    """Print regression metrics for a single parameter."""
    print(f"\n--- {name} [{min_p:.3f}, {max_p:.3f}] ---")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  Max error: {metrics['max_error']:.4f}")
    print(f"  Median error: {metrics['median_error']:.4f}")

    bins = metrics["bins"]
    if bins:
        print(f"  Error by range:")
        print(f"    {'Range':>20}  {'N':>5}  {'MAE':>8}  {'RMSE':>8}")
        for b in bins:
            if b["n"] > 0:
                print(f"    [{b['lo']:6.3f}, {b['hi']:6.3f}]  {b['n']:5d}  {b['mae']:8.4f}  {b['rmse']:8.4f}")
            else:
                print(f"    [{b['lo']:6.3f}, {b['hi']:6.3f}]  {b['n']:5d}       -         -")


def print_confusion_matrix(confusion, labels, name):
    """Print a confusion matrix with per-class accuracy."""
    num_classes = len(labels)
    total = confusion.sum()
    correct = np.trace(confusion)
    accuracy = 100 * correct / total if total > 0 else 0

    print(f"\n--- {name} ---")
    print(f"  Accuracy: {correct}/{total} = {accuracy:.1f}%")

    print(f"\n  Per-class accuracy:")
    for i in range(num_classes):
        class_total = confusion[i].sum()
        class_correct = confusion[i][i]
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0
        print(f"    p={labels[i]:>6}: {class_correct}/{class_total} = {class_acc:.1f}%")

    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    header = "".join(f"{labels[i]:>10}" for i in range(num_classes))
    print(f"    {'':>10}{header}")
    for i in range(num_classes):
        row = "".join(f"{confusion[i][j]:>10d}" for j in range(num_classes))
        print(f"    {'p=' + labels[i]:>10}{row}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_scatter_plots(results, epoch, output_path):
    """Save scatter plots for regression results.

    Args:
        results: list of (name, true_arr, pred_arr, min_p, max_p) tuples
        epoch: Training epoch for the title
        output_path: Path to save the PNG
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_params = len(results)
    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5))
    if n_params == 1:
        axes = [axes]

    for ax, (name, true_arr, pred_arr, lo, hi) in zip(axes, results):
        errors = pred_arr - true_arr
        mae = np.mean(np.abs(errors))
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((true_arr - np.mean(true_arr)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        ax.scatter(true_arr, pred_arr, alpha=0.4, s=15, c='steelblue', edgecolors='none')
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='ideal')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('True', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title(f'{name}\nMAE={mae:.4f}  R²={r2:.4f}', fontsize=13)
        ax.set_aspect('equal')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Evaluation (epoch {epoch}, {len(results[0][1])} samples)',
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlot saved to {output_path}")


# ---------------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------------

def evaluate_triple_param(encoder, controller, test_files, config, device):
    """Evaluate triple-param regression: JA drive + WF depth + WF rate."""
    sample_rate = config["sample_rate"]
    audio_length = config["audio_length"]
    min_ja, max_ja = config["min_param"], config["max_param"]
    min_d, max_d = config["min_depth"], config["max_depth"]
    min_r, max_r = config["min_rate"], config["max_rate"]

    all_true = {"ja": [], "depth": [], "rate": []}
    all_pred = {"ja": [], "depth": [], "rate": []}

    pbar = tqdm(test_files, ncols=80)
    for fpath in pbar:
        audio = _load_and_prepare_audio(fpath, sample_rate, audio_length)

        ja_val = random.uniform(min_ja, max_ja)
        depth_val = random.uniform(min_d, max_d)
        rate_val = random.uniform(min_r, max_r)

        x = audio.unsqueeze(0)
        y = ja_saturation(x, ja_val, sample_rate=sample_rate)
        y = wow_flutter(y, depth_val, sample_rate=sample_rate,
                        wow_rate=rate_val,
                        flutter_rate=config.get("flutter_rate", 0.5),
                        enable_ou=config.get("enable_ou", True),
                        interpolation=config.get("wf_interpolation", "linear"))
        y = conform_length(y, audio_length)
        y = linear_fade(y, sample_rate=sample_rate)

        y_in = y.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = controller(encoder(y_in))

        all_true["ja"].append(ja_val)
        all_true["depth"].append(depth_val)
        all_true["rate"].append(rate_val)
        all_pred["ja"].append(pred["ja"].item() * (max_ja - min_ja) + min_ja)
        all_pred["depth"].append(pred["depth"].item() * (max_d - min_d) + min_d)
        all_pred["rate"].append(pred["rate"].item() * (max_r - min_r) + min_r)

        if len(all_true["ja"]) > 1:
            mae_ja = np.mean(np.abs(np.array(all_true["ja"]) - np.array(all_pred["ja"])))
            mae_d = np.mean(np.abs(np.array(all_true["depth"]) - np.array(all_pred["depth"])))
            mae_r = np.mean(np.abs(np.array(all_true["rate"]) - np.array(all_pred["rate"])))
            pbar.set_postfix(ja=f"{mae_ja:.3f}", d=f"{mae_d:.3f}", r=f"{mae_r:.3f}")

    # Compute and print metrics for each parameter
    params = [
        ("JA Drive", "ja", min_ja, max_ja),
        ("WF Depth", "depth", min_d, max_d),
        ("WF Rate", "rate", min_r, max_r),
    ]
    print("=" * 60)
    print(f"Triple-param regression results ({len(all_true['ja'])} samples)")

    results_for_plot = []
    summary = []
    for name, key, lo, hi in params:
        true_arr = np.array(all_true[key])
        pred_arr = np.array(all_pred[key])
        metrics = compute_regression_metrics(true_arr, pred_arr, lo, hi)
        print_regression_metrics(metrics, name, lo, hi)
        summary.append((name, metrics["mae"], metrics["r2"]))
        results_for_plot.append((name, true_arr, pred_arr, lo, hi))

    print(f"\n--- Summary ---")
    print(f"  {'Param':>10}  {'MAE':>8}  {'R²':>8}")
    for name, mae, r2 in summary:
        print(f"  {name:>10}  {mae:8.4f}  {r2:8.4f}")

    return results_for_plot


def evaluate_regression(encoder, controller, test_files, config, device):
    """Evaluate single-param regression."""
    sample_rate = config["sample_rate"]
    audio_length = config["audio_length"]
    min_p = config["min_param"]
    max_p = config["max_param"]
    param_range = max_p - min_p

    all_true, all_pred = [], []

    pbar = tqdm(test_files, ncols=80)
    for fpath in pbar:
        audio = _load_and_prepare_audio(fpath, sample_rate, audio_length)
        param = random.uniform(min_p, max_p)

        x = audio.unsqueeze(0)
        audio_deg = _apply_degradation(x, param, config)
        audio_deg = conform_length(audio_deg, audio_length)
        audio_deg = linear_fade(audio_deg, sample_rate=sample_rate)

        y = audio_deg.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = controller(encoder(y)).item()

        all_true.append(param)
        all_pred.append(pred_norm * param_range + min_p)

        if len(all_true) > 1:
            mae_so_far = np.mean(np.abs(np.array(all_true) - np.array(all_pred)))
            pbar.set_postfix(mae=f"{mae_so_far:.3f}")

    true_arr = np.array(all_true)
    pred_arr = np.array(all_pred)
    metrics = compute_regression_metrics(true_arr, pred_arr, min_p, max_p)

    print("=" * 60)
    print(f"Regression results ({len(all_true)} samples)")
    print_regression_metrics(metrics, config["degradation_model"], min_p, max_p)

    # Sample predictions
    print(f"\nSample predictions (first 20):")
    print(f"  {'True':>8}  {'Pred':>8}  {'Error':>8}")
    errors = pred_arr - true_arr
    for i in range(min(20, len(all_true))):
        print(f"  {all_true[i]:8.3f}  {all_pred[i]:8.3f}  {errors[i]:+8.3f}")

    return [(config["degradation_model"], true_arr, pred_arr, min_p, max_p)]


def evaluate_classification(encoder, controller, test_files, config, device):
    """Evaluate single-param classification."""
    sample_rate = config["sample_rate"]
    audio_length = config["audio_length"]
    num_classes = config.get("num_classes", 10)
    min_p = config["min_param"]
    max_p = config["max_param"]

    if config.get("log_scale", False):
        param_values = np.geomspace(min_p, max_p, num_classes)
    else:
        param_values = np.linspace(min_p, max_p, num_classes)
    param_labels = [f"{g:.3f}" for g in param_values]
    print(f"Param values: {param_values.tolist()}")

    confusion = np.zeros((num_classes, num_classes), dtype=int)

    pbar = tqdm(test_files, ncols=80)
    correct, total = 0, 0
    for fpath in pbar:
        audio = _load_and_prepare_audio(fpath, sample_rate, audio_length)

        class_idx = random.randint(0, num_classes - 1)
        param = param_values[class_idx]

        x = audio.unsqueeze(0)
        audio_deg = _apply_degradation(x, param, config)
        audio_deg = conform_length(audio_deg, audio_length)
        audio_deg = linear_fade(audio_deg, sample_rate=sample_rate)

        y = audio_deg.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = controller(encoder(y))
            pred_idx = torch.argmax(logits, dim=-1).item()

        if pred_idx == class_idx:
            correct += 1
        total += 1
        confusion[class_idx][pred_idx] += 1
        pbar.set_postfix(acc=f"{100 * correct / total:.1f}%")

    print("=" * 60)
    print_confusion_matrix(confusion, param_labels, config["degradation_model"])
    return None  # No scatter plot for classification


def evaluate_multi_param(encoder, controller, test_files, config, device):
    """Evaluate dual-param classification: depth + rate."""
    sample_rate = config["sample_rate"]
    audio_length = config["audio_length"]
    nc_d = config["num_classes_depth"]
    nc_r = config["num_classes_rate"]
    depth_values = np.linspace(config["min_depth"], config["max_depth"], nc_d)
    rate_values = np.linspace(config["min_rate"], config["max_rate"], nc_r)
    depth_labels = [f"{v:.3f}" for v in depth_values]
    rate_labels = [f"{v:.3f}" for v in rate_values]

    print(f"Depth values ({nc_d} classes): {depth_labels}")
    print(f"Rate values ({nc_r} classes): {rate_labels}")

    confusion_depth = np.zeros((nc_d, nc_d), dtype=int)
    confusion_rate = np.zeros((nc_r, nc_r), dtype=int)
    correct_both, total = 0, 0

    pbar = tqdm(test_files, ncols=80)
    for fpath in pbar:
        audio = _load_and_prepare_audio(fpath, sample_rate, audio_length)

        depth_idx = random.randint(0, nc_d - 1)
        rate_idx = random.randint(0, nc_r - 1)

        x = audio.unsqueeze(0)
        audio_deg = _apply_degradation(x, depth_values[depth_idx], config,
                                       wow_rate_override=rate_values[rate_idx])
        audio_deg = conform_length(audio_deg, audio_length)
        audio_deg = linear_fade(audio_deg, sample_rate=sample_rate)

        y = audio_deg.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = controller(encoder(y))
            pred_d = torch.argmax(logits["depth"], dim=-1).item()
            pred_r = torch.argmax(logits["rate"], dim=-1).item()

        confusion_depth[depth_idx][pred_d] += 1
        confusion_rate[rate_idx][pred_r] += 1
        if pred_d == depth_idx and pred_r == rate_idx:
            correct_both += 1
        total += 1

        acc_d = 100 * np.trace(confusion_depth) / total
        acc_r = 100 * np.trace(confusion_rate) / total
        pbar.set_postfix(d=f"{acc_d:.0f}%", r=f"{acc_r:.0f}%")

    print("=" * 60)
    print(f"Combined accuracy: {correct_both}/{total} = {100 * correct_both / total:.1f}%")
    print_confusion_matrix(confusion_depth, depth_labels, "Depth")
    print_confusion_matrix(confusion_rate, rate_labels, "Rate")
    return None  # No scatter plot for classification


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate on test split")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (loads config from outputs/<name>/config.yaml)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (overrides --name config)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (overrides --name checkpoint)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_files", type=int, default=0, help="Max test files (0=all)")
    parser.add_argument("--plot", action="store_true", help="Generate scatter plots")
    args = parser.parse_args()

    # Resolve config and checkpoint paths
    if args.name:
        exp_dir = Path("outputs") / args.name
        config_path = args.config or str(exp_dir / "config.yaml")
        checkpoint_path = args.checkpoint or str(exp_dir / "checkpoints" / "best_model.pt")
    elif args.config and args.checkpoint:
        config_path = args.config
        checkpoint_path = args.checkpoint
        exp_dir = None
    else:
        parser.error("Provide --name or both --config and --checkpoint")
        return

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")

    # Detect mode
    degradation_model = config.get("degradation_model", "ja")
    regression = config.get("regression", False)
    triple_param = (degradation_model == "ja_wf")
    multi_param = (config.get("wf_target_param") == "both")

    if triple_param:
        mode = "regression triple-param (ja + depth + rate)"
    elif regression:
        mode = "regression multi-param" if multi_param else "regression"
    elif multi_param:
        mode = "classification multi-param"
    else:
        mode = "classification"
    print(f"Mode: {mode}")
    print(f"Device: {args.device}")

    # Load model
    encoder, controller, epoch = load_model(config, checkpoint_path, args.device)
    print(f"Model loaded (epoch {epoch})")

    # Get test files
    test_files = get_test_files(
        config["audio_dir"],
        config.get("input_dirs"),
        config.get("ext", "mp3"),
    )
    print(f"Test split: {len(test_files)} files")

    if args.max_files > 0 and len(test_files) > args.max_files:
        test_files = test_files[:args.max_files]
        print(f"Using first {args.max_files} files")

    # Run evaluation
    if triple_param:
        plot_data = evaluate_triple_param(encoder, controller, test_files, config, args.device)
    elif regression:
        plot_data = evaluate_regression(encoder, controller, test_files, config, args.device)
    elif multi_param:
        plot_data = evaluate_multi_param(encoder, controller, test_files, config, args.device)
    else:
        plot_data = evaluate_classification(encoder, controller, test_files, config, args.device)

    # Generate plots
    if args.plot and plot_data:
        if args.name:
            plot_path = Path("outputs") / args.name / "eval" / "scatter.png"
        else:
            plot_path = Path("outputs") / "eval_scatter.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        save_scatter_plots(plot_data, epoch, str(plot_path))


if __name__ == "__main__":
    main()
