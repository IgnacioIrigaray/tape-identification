"""
Script de evaluación sobre el split de test.

Usa la misma lógica de split determinista que el dataset (seed=42, train_frac=0.8)
para obtener el 10% de archivos de test, aplica la degradación con una clase al azar,
y evalúa el modelo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import glob
import random
import torch
import torchaudio
import argparse
import numpy as np
from tqdm import tqdm

from tape_id.models.encoder import SpectralEncoder
from tape_id.models.controller import ParameterController
from tape_id.data.dataset import hard_clipping, tape_saturation, ja_saturation, wow_flutter
from tape_id.utils import split_dataset, conform_length, linear_fade


def load_model(checkpoint_path: str, device: str = "cpu", num_classes: int = 10,
               sample_rate: int = 22050, multi_param: bool = False,
               num_classes_depth: int = 3, num_classes_rate: int = 3):
    """Carga modelo desde checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder = SpectralEncoder(num_params=1, sample_rate=sample_rate, embed_dim=1024, width_mult=2).to(device)

    if multi_param:
        controller = ParameterController(
            embed_dim=1024, hidden_dim=256,
            num_classes_depth=num_classes_depth,
            num_classes_rate=num_classes_rate,
        ).to(device)
    else:
        controller = ParameterController(num_classes=num_classes, embed_dim=1024, hidden_dim=256).to(device)

    encoder.load_state_dict(checkpoint["encoder_state"])
    controller.load_state_dict(checkpoint["controller_state"])

    encoder.eval()
    controller.eval()

    return encoder, controller, checkpoint.get("epoch", "?")


def get_test_files(audio_dir, input_dirs, ext="mp3", train_frac=0.8):
    """Obtiene los archivos del split de test usando la misma lógica que el dataset."""
    filepaths = []
    for input_dir in input_dirs:
        search_path = os.path.join(audio_dir, input_dir, f"*.{ext}")
        filepaths += glob.glob(search_path)
    filepaths = sorted(filepaths)

    if len(filepaths) == 0:
        search_path = os.path.join(audio_dir, f"*.{ext}")
        filepaths = sorted(glob.glob(search_path))

    # Mismo shuffle determinista que dataset.py
    rng_split = random.Random(42)
    rng_split.shuffle(filepaths)
    return split_dataset(filepaths, "test", train_frac)


def apply_degradation(x, param, args, wow_rate_override=None):
    """Aplica la degradación configurada al audio."""
    if args.degradation_model == "wow_flutter":
        depth = param
        rate = args.wow_rate
        if args.multi_param and wow_rate_override is not None:
            rate = wow_rate_override
        elif args.wf_target_param == "rate":
            depth = args.wf_fixed_depth
            rate = param
        return wow_flutter(x, depth, sample_rate=args.sample_rate,
                           wow_rate=rate, flutter_rate=args.flutter_rate,
                           enable_ou=args.enable_ou, interpolation=args.wf_interpolation)
    elif args.degradation_model == "hard_clipping":
        return hard_clipping(x, param)
    elif args.degradation_model == "ja":
        return ja_saturation(x, param, sample_rate=args.sample_rate)
    else:
        return tape_saturation(x, param)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on test split")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, default="/mnt/data/working_datasets/jamendo")
    parser.add_argument("--input_dirs", nargs="+", default=["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"])
    parser.add_argument("--ext", type=str, default="mp3")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--min_gain", type=float, default=0.05)
    parser.add_argument("--max_gain", type=float, default=1.0)
    parser.add_argument("--log_scale", action="store_true", help="Use log (geomspace) instead of linear")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--max_files", type=int, default=0, help="Max test files (0=all)")
    parser.add_argument("--audio_length", type=int, default=65536, help="Audio length in samples")
    # Degradation model
    parser.add_argument("--degradation_model", type=str, default="wow_flutter",
                        choices=["hard_clipping", "tanh", "ja", "wow_flutter"])
    # Wow/flutter params
    parser.add_argument("--wow_rate", type=float, default=0.4)
    parser.add_argument("--flutter_rate", type=float, default=0.5)
    parser.add_argument("--enable_ou", action="store_true", default=True)
    parser.add_argument("--no_ou", action="store_true", help="Disable OU process")
    parser.add_argument("--wf_interpolation", type=str, default="linear", choices=["linear", "lagrange3"])
    parser.add_argument("--wf_target_param", type=str, default="depth", choices=["depth", "rate", "both"])
    parser.add_argument("--wf_fixed_depth", type=float, default=0.5)
    # Multi-param
    parser.add_argument("--multi_param", action="store_true", help="Dual-param (depth+rate) mode")
    parser.add_argument("--num_classes_depth", type=int, default=3)
    parser.add_argument("--num_classes_rate", type=int, default=3)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=0.8)
    parser.add_argument("--min_rate", type=float, default=0.1)
    parser.add_argument("--max_rate", type=float, default=0.8)
    args = parser.parse_args()

    if args.no_ou:
        args.enable_ou = False

    # Auto-detect multi_param from wf_target_param
    if args.wf_target_param == "both":
        args.multi_param = True

    device = args.device
    print(f"Device: {device}")
    print(f"Degradation: {args.degradation_model}")
    print(f"Mode: {'multi-param (depth+rate)' if args.multi_param else 'single-param'}")

    # Cargar modelo
    encoder, controller, epoch = load_model(
        args.checkpoint, device, args.num_classes, args.sample_rate,
        multi_param=args.multi_param,
        num_classes_depth=args.num_classes_depth,
        num_classes_rate=args.num_classes_rate,
    )
    print(f"Model loaded (epoch {epoch})")

    # Obtener archivos de test
    test_files = get_test_files(args.audio_dir, args.input_dirs, args.ext)
    print(f"Test split: {len(test_files)} files")

    if args.max_files > 0 and len(test_files) > args.max_files:
        test_files = test_files[:args.max_files]
        print(f"Using first {args.max_files} files")

    sample_rate = args.sample_rate
    audio_length = args.audio_length

    if args.multi_param:
        _evaluate_multi_param(encoder, controller, test_files, args, device, sample_rate, audio_length)
    else:
        _evaluate_single_param(encoder, controller, test_files, args, device, sample_rate, audio_length)


def _evaluate_single_param(encoder, controller, test_files, args, device, sample_rate, audio_length):
    """Evaluación single-param (backward compat)."""
    if args.log_scale:
        param_values = torch.tensor(np.geomspace(args.min_gain, args.max_gain, args.num_classes), dtype=torch.float32)
    else:
        param_values = torch.tensor(np.linspace(args.min_gain, args.max_gain, args.num_classes), dtype=torch.float32)
    param_labels = [f"{g:.3f}" for g in param_values.tolist()]
    print(f"Param values: {param_values.tolist()}")

    num_classes = args.num_classes
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    correct = 0
    total = 0

    pbar = tqdm(test_files, ncols=80)
    for fpath in pbar:
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

        class_idx = random.randint(0, num_classes - 1)
        param = param_values[class_idx].item()
        x = audio.unsqueeze(0)
        audio_deg = apply_degradation(x, param, args)
        audio_deg = conform_length(audio_deg, audio_length)
        audio_deg = linear_fade(audio_deg, sample_rate=sample_rate)

        y = audio_deg.unsqueeze(0).to(device)
        with torch.no_grad():
            e_y = encoder(y)
            logits = controller(e_y)
            pred_idx = torch.argmax(logits, dim=-1).item()

        if pred_idx == class_idx:
            correct += 1
        total += 1
        confusion[class_idx][pred_idx] += 1

        accuracy_so_far = 100 * correct / total if total > 0 else 0
        pbar.set_postfix(acc=f"{accuracy_so_far:.1f}%")

    accuracy = 100 * correct / total if total > 0 else 0
    print("=" * 60)
    print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")

    print(f"\nPer-class accuracy:")
    for i in range(num_classes):
        class_total = confusion[i].sum()
        class_correct = confusion[i][i]
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0
        print(f"  p={param_labels[i]:>6}: {class_correct}/{class_total} = {class_acc:.1f}%")

    print(f"\nConfusion matrix (rows=real, cols=predicted):")
    header = "".join(f"{param_labels[i]:>10}" for i in range(num_classes))
    print(f"{'':>10}{header}")
    for i in range(num_classes):
        row = "".join(f"{confusion[i][j]:>10d}" for j in range(num_classes))
        print(f"{'p=' + param_labels[i]:>10}{row}")


def _evaluate_multi_param(encoder, controller, test_files, args, device, sample_rate, audio_length):
    """Evaluación dual-param: depth + rate."""
    nc_d = args.num_classes_depth
    nc_r = args.num_classes_rate
    depth_values = np.linspace(args.min_depth, args.max_depth, nc_d)
    rate_values = np.linspace(args.min_rate, args.max_rate, nc_r)
    depth_labels = [f"{v:.3f}" for v in depth_values]
    rate_labels = [f"{v:.3f}" for v in rate_values]

    print(f"Depth values ({nc_d} classes): {depth_labels}")
    print(f"Rate values ({nc_r} classes): {rate_labels}")

    confusion_depth = np.zeros((nc_d, nc_d), dtype=int)
    confusion_rate = np.zeros((nc_r, nc_r), dtype=int)
    correct_depth = 0
    correct_rate = 0
    correct_both = 0
    total = 0

    pbar = tqdm(test_files, ncols=80)
    for fpath in pbar:
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

        depth_idx = random.randint(0, nc_d - 1)
        rate_idx = random.randint(0, nc_r - 1)
        depth_val = depth_values[depth_idx]
        rate_val = rate_values[rate_idx]

        x = audio.unsqueeze(0)
        audio_deg = apply_degradation(x, depth_val, args, wow_rate_override=rate_val)
        audio_deg = conform_length(audio_deg, audio_length)
        audio_deg = linear_fade(audio_deg, sample_rate=sample_rate)

        y = audio_deg.unsqueeze(0).to(device)
        with torch.no_grad():
            e_y = encoder(y)
            logits = controller(e_y)
            pred_d = torch.argmax(logits["depth"], dim=-1).item()
            pred_r = torch.argmax(logits["rate"], dim=-1).item()

        if pred_d == depth_idx:
            correct_depth += 1
        if pred_r == rate_idx:
            correct_rate += 1
        if pred_d == depth_idx and pred_r == rate_idx:
            correct_both += 1
        total += 1

        confusion_depth[depth_idx][pred_d] += 1
        confusion_rate[rate_idx][pred_r] += 1

        acc_d = 100 * correct_depth / total
        acc_r = 100 * correct_rate / total
        pbar.set_postfix(d=f"{acc_d:.0f}%", r=f"{acc_r:.0f}%")

    print("=" * 60)
    print(f"Depth accuracy:    {correct_depth}/{total} = {100*correct_depth/total:.1f}%")
    print(f"Rate accuracy:     {correct_rate}/{total} = {100*correct_rate/total:.1f}%")
    print(f"Combined accuracy: {correct_both}/{total} = {100*correct_both/total:.1f}%")

    # Depth confusion
    print(f"\n--- Depth per-class accuracy ---")
    for i in range(nc_d):
        ct = confusion_depth[i].sum()
        cc = confusion_depth[i][i]
        print(f"  depth={depth_labels[i]:>6}: {cc}/{ct} = {100*cc/ct:.1f}%" if ct > 0 else f"  depth={depth_labels[i]:>6}: 0/0")

    print(f"\nDepth confusion matrix:")
    header = "".join(f"{depth_labels[i]:>10}" for i in range(nc_d))
    print(f"{'':>10}{header}")
    for i in range(nc_d):
        row = "".join(f"{confusion_depth[i][j]:>10d}" for j in range(nc_d))
        print(f"{'d=' + depth_labels[i]:>10}{row}")

    # Rate confusion
    print(f"\n--- Rate per-class accuracy ---")
    for i in range(nc_r):
        ct = confusion_rate[i].sum()
        cc = confusion_rate[i][i]
        print(f"  rate={rate_labels[i]:>6}: {cc}/{ct} = {100*cc/ct:.1f}%" if ct > 0 else f"  rate={rate_labels[i]:>6}: 0/0")

    print(f"\nRate confusion matrix:")
    header = "".join(f"{rate_labels[i]:>10}" for i in range(nc_r))
    print(f"{'':>10}{header}")
    for i in range(nc_r):
        row = "".join(f"{confusion_rate[i][j]:>10d}" for j in range(nc_r))
        print(f"{'r=' + rate_labels[i]:>10}{row}")


if __name__ == "__main__":
    main()
