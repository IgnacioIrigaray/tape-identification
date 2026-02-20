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

**Controller modes:**

| Mode | Description |
|------|-------------|
| Classification | Predicts discrete parameter class |
| Regression | Predicts continuous parameter value in [0, 1] |
| Multi-param | Predicts two parameters simultaneously |
| Triple-param | Predicts JA drive + WF depth + WF rate simultaneously |

## Architecture

```
Degraded audio (y) ──► SpectralEncoder (MobileNetV2 + STFT)
                              │
                              ▼
                        Embedding [1024-dim]
                              │
                              ▼
                       ParameterController (MLP)
                              │
                              ▼
                    Predicted parameters [0, 1]
```

### Components

**SpectralEncoder**
- STFT spectrogram: n_fft=4096, hop=2048, power compression |X|^0.3
- MobileNetV2 backbone (width_mult=2) on normalized spectrogram
- L2-normalized output embedding: 1024-dim

**ParameterController**
- Single-param regression: `[1024] → [256] → [1]` + Sigmoid
- Triple-param regression: shared trunk + 3 heads (ja / depth / rate)
- Classification: replaces Sigmoid with softmax over N classes

**DifferentiableForwardModel** (for signal loss)
- Differentiable approximation of each degradation for use in the training loop
- JA: uses anhysteretic Langevin curve `3a·L(x/a)` instead of full RK4
- Wow/flutter: sinusoidal LFO with `torch.lerp` interpolation

### Loss Functions

**Parameter loss** (always active):
- Regression: `MSELoss(predicted_params, target_params)`
- Classification: `CrossEntropyLoss(logits, class_idx)`

**Signal loss** (optional, regression only):
- Multi-Resolution STFT Loss comparing log-magnitude at 3 resolutions
- `MR-STFT(forward_model(x_clean, predicted_params), y_degraded)`
- Enabled by setting `signal_loss_weight > 0` in config

**Combined loss:**
```
total = param_loss_weight * param_loss + signal_loss_weight * signal_loss
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

### Training

```bash
python scripts/train.py --config configs/default.yaml --name my_experiment
```

All config values can be overridden from the CLI:

```bash
# Single-param JA regression
python scripts/train.py --config configs/default.yaml --name ja_reg \
    --degradation_model ja --regression true

# Triple-param (JA + wow/flutter)
python scripts/train.py --config configs/triple_param.yaml --name triple_v1

# With signal loss
python scripts/train.py --config configs/default.yaml --name ja_signal \
    --signal_loss_weight 0.1 --param_loss_weight 1.0
```

Training auto-detects existing checkpoints and offers to resume.

### Configuration

Key config parameters (`configs/default.yaml`):

```yaml
# Data
audio_dir: /path/to/audio
ext: mp3                          # or wav
sample_rate: 22050
audio_length: 65536               # ~3 seconds

# Model
degradation_model: ja             # tanh | hard_clipping | ja | wow_flutter | ja_wf
regression: true
num_classes: 3                    # for classification mode
min_param: 1.0
max_param: 10.0
embed_dim: 1024
hidden_dim: 256

# Training
batch_size: 32
num_epochs: 400
learning_rate: 5.0e-5
patience: 15                      # early stopping
grad_clip_norm: 1.0

# Signal loss (optional)
signal_loss_weight: 0.0           # 0 = disabled
param_loss_weight: 1.0
```

For triple-param mode (`ja_wf`), also configure:

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
python scripts/evaluate.py \
    --checkpoint outputs/my_experiment/checkpoints/best_model.pt \
    --config outputs/my_experiment/config.yaml \
    --audio_dir /path/to/test/audio
```

## Project Structure

```
tape-identification/
├── tape_id/
│   ├── models/
│   │   ├── encoder.py          # SpectralEncoder (MobileNetV2 + STFT)
│   │   ├── controller.py       # ParameterController (MLP, single/multi/triple)
│   │   ├── tape_processor.py   # Degradation processors + DifferentiableForwardModel
│   │   └── mobilenetv2.py      # MobileNetV2 backbone
│   ├── data/
│   │   ├── dataset.py          # TapeSaturationDataset (on-the-fly generation)
│   │   └── audio.py            # AudioFile I/O helper
│   ├── training/
│   │   ├── trainer.py          # Training loop (classification / regression / triple)
│   │   └── losses.py           # MultiResolutionSTFTLoss
│   └── utils.py
├── scripts/
│   ├── train.py                # Main training script
│   └── evaluate.py             # Evaluation script
├── configs/
│   ├── default.yaml            # Single-param regression (JA)
│   └── triple_param.yaml       # Triple-param regression (JA + wow/flutter)
├── tests/
└── outputs/
    └── <experiment_name>/
        ├── checkpoints/        # best_model.pt, last_model.pt
        ├── logs/               # TensorBoard logs
        ├── eval/               # Evaluation results
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
3. Sample random parameter(s) uniformly from configured range
4. Apply degradation model
5. Apply length conforming + fade in/out
6. Normalize target parameter(s) to [0, 1]

When signal loss is enabled, clean audio `x_clean` is also returned as the first batch element: `(x_clean, y_degraded, *targets)`.

**Buffer management:** Audio files are preloaded into RAM in configurable chunks (`buffer_size_gb`) and refreshed periodically (`buffer_reload_rate`), avoiding disk I/O bottlenecks during training.

## Training Tips

- **Signal loss weight**: Start with `signal_loss_weight: 0.1`. The MR-STFT loss is ~10-30x larger than MSE in absolute terms at initialization, so keep `param_loss_weight` proportionally higher if both are active.
- **JA signal loss**: The differentiable forward model uses the anhysteretic approximation (not full RK4), so there is an irreducible mismatch. Signal loss is most accurate for `tanh` and `hard_clipping`.
- **Learning rate**: `5e-5` with `ReduceLROnPlateau` (factor=0.5, patience=5) works well.
- **Early stopping**: Default patience=15 epochs. The scheduler reduces LR before stopping.
- **Buffer size**: Larger `buffer_size_gb` (up to your available RAM) improves variety per epoch.

## References

- **DeepAFx-ST**: [Differentiable Signal Processing with Black-Box Audio Effects](https://arxiv.org/abs/2105.04752) — original architecture
- **Jiles-Atherton model**: AnalogTapeModel by Jatin Chowdhury
- **MobileNetV2**: [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

## License

Uses code from [DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST) (BSD-3-Clause). See [LICENSE](LICENSE).
