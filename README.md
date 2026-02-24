# Tape Parameter Identification

Neural network for identifying tape degradation parameters from audio. Given a degraded audio signal, the model predicts the physical parameters of the degradation model that produced it.

Simplified adaptation of [DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST) focused solely on **parameter identification**.

## Overview

The model takes a degraded audio segment as input and predicts the parameters of the degradation applied to it. Training data is generated on-the-fly by applying degradation models with random parameters to clean audio.

**Degradation models supported:**

| Model | Description | Parameters |
|-------|-------------|------------|
| `tanh` | Tanh saturation | `gain` ∈ [1, 10] |
| `hard_clipping` | Hard clipping | `gain` ∈ [1, 4] |
| `ja` | Jiles-Atherton hysteresis (full RK4) | `drive` ∈ [1, 10] |
| `wow_flutter` | Time-base distortion (wow/flutter) | `depth`, `rate` ∈ [0.1, 0.8] |
| `ja_wf` | JA hysteresis + wow/flutter | `ja`, `depth`, `rate` |
| `tape_noise` | Additive tape noise (MagTapeDB) | `SNR` ∈ [0, 50] dB |

**Controller modes:**

| Mode | Description |
|------|-------------|
| Classification | Predicts discrete parameter class |
| Regression | Predicts continuous parameter value via sigmoid + dB mapping |
| Multi-param | Predicts two parameters simultaneously |
| Triple-param | Predicts JA drive + WF depth + WF rate simultaneously |

## Architecture

```
Degraded audio (y) ──► SpectralEncoder (STFT + MobileNetV2 / EfficientNet-B2)
                              │
                              ▼
                        Embedding [1024-dim, L2-normalized]
                              │
                              ▼
                       ParameterController (MLP)
                              │
                              ▼
                    Predicted parameters (dB / normalized)
```

### Encoder

The `SpectralEncoder` supports two backbones, configurable via `encoder_model`:

**MobileNetV2** (`encoder_model: mobilenet_v2`, default)
- Log-magnitude STFT spectrogram: `20·log10(|X| + 1e-8)` with -80 dB floor
- Normalization: `(X - mean) / std` with mean=-10.0, std=19.4
- MobileNetV2 backbone with `width_mult=2`
- Input: 3-channel (spectrogram replicated) or multi-resolution STFT
- L2-normalized output embedding: 1024-dim

**EfficientNet-B2** (`encoder_model: efficient_net`)
- Same log-magnitude STFT processing
- Input: 1-channel spectrogram, image size (128, 65)
- 1×1 convolution projection from 1408 → `embed_dim`
- L2-normalized output embedding: 1024-dim

**Multi-resolution STFT** (`multi_resolution: true`, MobileNetV2 only)
- 3 STFTs with different time-frequency resolutions as 3 input channels:
  - n_fft: [4096, 2048, 1024], hop: [2048, 1024, 512]
  - Each interpolated to 128×128 via bilinear interpolation
- Provides complementary time-frequency information

### Controller

**Single-param regression:**
- MLP: `[1024] → [256] → [256] → [1]` with LeakyReLU(0.1), dropout 0.1
- Output: raw logit (sigmoid applied in trainer)

**Triple-param regression (ja_wf):**
- Shared trunk: `[1024] → [256] → [256]`
- 3 independent heads: each `[256] → [1]` for ja / depth / rate

**Classification:**
- MLP: `[1024] → [256] → [num_classes]`
- Output: logits (CrossEntropyLoss)

### Loss Functions

**Parameter loss** (always active):
- Regression: `MSELoss(pred_dB, target_dB)` — predictions mapped to dB via sigmoid
- Classification: `CrossEntropyLoss(logits, class_idx)`

**Signal loss** (optional, regression only):
- Multi-Resolution STFT Loss comparing log-magnitude at 3 resolutions
- `MR-STFT(forward_model(x_clean, predicted_params), y_degraded)`
- Enabled by setting `signal_loss_weight > 0`

**Combined loss:**
```
total = param_loss_weight * param_loss + signal_loss_weight * signal_loss
```

### Sigmoid + dB Prediction

For regression, the trainer maps raw controller outputs to physical units:
```
pred_norm = sigmoid(raw_output)
pred_dB = pred_norm × (max_param - min_param) + min_param
loss = MSE(pred_dB, target_dB)
```

This avoids the zero-gradient problem of `torch.clamp` outside [0, 1].

### Headroom

To keep sigmoid targets in the healthy gradient zone, `min_data` / `max_data` define the actual sampling range while `min_param` / `max_param` define the normalization range:
```
# Example: SNR sampled in [10, 30], normalized with [0, 50]
# → sigmoid targets land in [0.2, 0.6], well within sigmoid's linear region
min_param: 0
max_param: 50
min_data: 10
max_data: 30
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Requires Python ≥ 3.9 with PyTorch ≥ 2.0.

## Usage

### Training

```bash
python scripts/train.py --config configs/default.yaml --name my_experiment
```

All config values can be overridden from the CLI:

```bash
# Single-param JA regression
python scripts/train.py --config configs/ja.yaml --name exp_ja

# Tape noise with EfficientNet
python scripts/train.py --config configs/tape_noise.yaml --name exp_noise

# MobileNetV2 with multi-resolution STFT
python scripts/train.py --config configs/tape_noise.yaml --name exp_noise_multires \
    --encoder_model mobilenet_v2 --multi_resolution true

# Triple-param (JA + wow/flutter)
python scripts/train.py --config configs/triple_param.yaml --name exp_triple

# With signal loss
python scripts/train.py --config configs/default.yaml --name exp_signal \
    --signal_loss_weight 0.1 --param_loss_weight 1.0
```

Training auto-resumes from checkpoint if one exists in `outputs/<name>/checkpoints/`.

### Configuration

Key config parameters:

```yaml
# Data
audio_dir: /path/to/audio
ext: mp3                          # or wav
sample_rate: 22050
audio_length: 65536               # ~3 seconds
train_examples_per_epoch: 5000
val_examples_per_epoch: 500
buffer_size_gb: 1
buffer_reload_rate: 2000

# Model
degradation_model: ja             # tanh | hard_clipping | ja | wow_flutter | ja_wf | tape_noise
regression: true
encoder_model: mobilenet_v2       # mobilenet_v2 | efficient_net
multi_resolution: false           # 3-channel multi-res STFT (mobilenet_v2 only)
embed_dim: 1024
hidden_dim: 256
num_classes: 3                    # for classification mode
min_param: 1.0
max_param: 10.0

# Headroom (optional, for sigmoid regression)
min_data: 1.0                    # actual sampling range lower bound
max_data: 10.0                   # actual sampling range upper bound

# Training
batch_size: 32
num_epochs: 400
learning_rate: 5.0e-5
weight_decay: 1.0e-5
patience: 15                      # early stopping
grad_clip_norm: 1.0
scheduler_factor: 0.5
scheduler_patience: 5

# Signal loss (optional)
signal_loss_weight: 0.0           # 0 = disabled
param_loss_weight: 1.0

# Tape noise specific
noise_dir: /path/to/MagTapeDB    # required for tape_noise model
noise_ext: wav
noise_preload: true               # preload all noise files to RAM
```

For triple-param mode (`ja_wf`):

```yaml
degradation_model: ja_wf
min_depth: 0.1
max_depth: 0.8
min_rate: 0.1
max_rate: 0.8
```

### Monitoring

```bash
tensorboard --logdir=outputs/my_experiment/logs
```

Logged metrics:
- `train/loss_step`, `train/loss_epoch`: training loss
- `val/loss`: validation loss
- `train/mae_*`, `val/mae_*`: mean absolute error per parameter (regression)
- `train/rmse_*`, `val/rmse_*`: RMSE per parameter
- `val/accuracy_*`: accuracy per parameter (classification)
- `train/signal_loss_step`: signal loss (when enabled)

### Evaluation

```bash
# By experiment name (loads config + checkpoint automatically)
python scripts/evaluate.py --name my_experiment --plot --device cpu

# With explicit paths
python scripts/evaluate.py \
    --config outputs/my_experiment/config.yaml \
    --checkpoint outputs/my_experiment/checkpoints/best_model.pt \
    --plot --max_files 200

# Tape noise (requires noise directory)
python scripts/evaluate.py --name exp_noise \
    --noise_dir /path/to/MagTapeDB --noise_ext wav --plot
```

Options:
- `--plot`: generate scatter plots (`outputs/<name>/eval/scatter.png`)
- `--signal_loss`: compute MR-STFT signal reconstruction loss
- `--max_files N`: limit number of test files (0 = all)
- `--device`: `cuda` or `cpu` (default: `cpu`)

## Project Structure

```
tape-identification/
├── tape_id/
│   ├── models/
│   │   ├── encoder.py          # SpectralEncoder (STFT + MobileNetV2 / EfficientNet-B2)
│   │   ├── controller.py       # ParameterController (MLP, single/multi/triple)
│   │   ├── tape_processor.py   # Degradation processors + DifferentiableForwardModel
│   │   ├── mobilenetv2.py      # MobileNetV2 backbone
│   │   └── efficient_net/      # EfficientNet-B2 backbone
│   ├── data/
│   │   ├── dataset.py          # TapeSaturationDataset (on-the-fly generation)
│   │   └── audio.py            # AudioFile I/O helper
│   ├── training/
│   │   ├── trainer.py          # Training loop (classification / regression / triple)
│   │   └── losses.py           # MultiResolutionSTFTLoss
│   └── utils.py
├── scripts/
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Evaluation script
│   └── process_audio.py        # Apply degradation models with fixed params
├── configs/
│   ├── default.yaml            # Single-param regression (JA)
│   ├── triple_param.yaml       # Triple-param regression (JA + wow/flutter)
│   ├── tape_noise.yaml         # Tape noise SNR estimation
│   ├── ja.yaml                 # Jiles-Atherton baseline
│   ├── tanh.yaml               # Tanh saturation baseline
│   ├── hard_clipping.yaml      # Hard clipping baseline
│   └── wow_flutter.yaml        # Wow/flutter baseline
├── tests/
├── EXPERIMENTS.md              # Systematic experiment plan
└── outputs/
    └── <experiment_name>/
        ├── checkpoints/        # best_model.pt, last_model.pt
        ├── logs/               # TensorBoard logs
        ├── eval/               # Scatter plots, metrics
        └── config.yaml         # Config copy for reproducibility
```

## Audio Format

- Sample rate: 22050 Hz
- Segment length: 65536 samples (~3 seconds)
- Normalization: peak to 0 dBFS before degradation
- Fades: 50ms linear fade in/out applied after degradation

## Dataset Generation

The dataset generates `(y_degraded, *targets)` pairs on-the-fly:

1. Load random audio patch from RAM buffer
2. Normalize to 0 dBFS
3. Sample random parameter(s) uniformly from `[min_data, max_data]`
4. Apply degradation model
5. Apply length conforming + fade in/out
6. Normalize target parameter(s) to [0, 1] using `[min_param, max_param]`

When signal loss is enabled, clean audio is also returned: `(x_clean, y_degraded, *targets)`.

**Buffer management:** Audio files are preloaded into RAM in configurable chunks (`buffer_size_gb`) and refreshed periodically (`buffer_reload_rate`), avoiding disk I/O bottlenecks during training. For tape noise, `noise_preload: true` loads all MagTapeDB files into RAM.

## Training Tips

- **Signal loss weight**: Start with `signal_loss_weight: 0.1`. The MR-STFT loss is ~10-30x larger than MSE at initialization, so keep `param_loss_weight` proportionally higher.
- **JA signal loss**: The differentiable forward model uses the anhysteretic approximation (not full RK4), so there is an irreducible mismatch. Signal loss is most accurate for `tanh` and `hard_clipping`.
- **Learning rate**: `5e-5` with `ReduceLROnPlateau` (factor=0.5, patience=5) works well for most models. Tape noise may benefit from `1e-3`.
- **Early stopping**: Default patience=15 epochs. The scheduler reduces LR before stopping.
- **Buffer size**: Larger `buffer_size_gb` (up to your available RAM) improves variety per epoch.
- **Headroom**: For bounded parameters (like SNR in dB), use `min_data`/`max_data` to keep sigmoid targets away from saturation regions.
- **Multi-resolution STFT**: Use with MobileNetV2 for richer time-frequency representation. Each channel captures a different resolution trade-off.

## References

- **DeepAFx-ST**: [Differentiable Signal Processing with Black-Box Audio Effects](https://arxiv.org/abs/2105.04752) — original architecture
- **Jiles-Atherton model**: AnalogTapeModel by Jatin Chowdhury
- **MobileNetV2**: [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- **EfficientNet**: [Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- **MagTapeDB**: Magnetic tape noise database used for `tape_noise` model

## License

Uses code from [DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST) (BSD-3-Clause). See [LICENSE](LICENSE).
