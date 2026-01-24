# Tape Saturation Parameter Identification

Deep learning model for identifying tape saturation parameters from audio.

Simplified adaptation of [DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST) focused solely on **parameter identification** (not style transfer).

## Overview

This project implements a neural network that learns to predict the saturation depth parameter of tape saturation effects by analyzing clean and processed audio pairs.

**Key Features:**
- ✅ End-to-end parameter identification using spectral encoder + MLP controller
- ✅ On-the-fly dataset generation with buffer management for efficient training
- ✅ TensorBoard logging for real-time training monitoring
- ✅ Automatic checkpoint recovery from crashes
- ✅ Multi-resolution STFT loss for audio comparison
- ✅ Clean, simplified codebase (~500 lines vs ~3000 in original)

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU training)

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/tape-identification.git
cd tape-identification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python scripts/train.py
```

The script will:
1. Load audio dataset from `audio_dir`
2. Apply tape saturation with random depth values [1, 10]
3. Train encoder + controller to predict saturation parameters
4. Save checkpoints to `outputs/checkpoints/`
5. Log metrics to TensorBoard in `outputs/logs/`

**Configuration** (edit in `scripts/train.py`):
- `audio_dir`: Path to audio dataset (MP3 or WAV files)
- `batch_size`: Batch size (default: 16)
- `num_epochs`: Training epochs (default: 50)
- `learning_rate`: Learning rate (default: 3e-4)
- `num_workers`: DataLoader workers (default: 0)
- `buffer_size_gb`: Audio buffer size in GB (default: 0.5)

**Resume from Checkpoint:**
If training is interrupted, the script automatically detects existing checkpoints and asks if you want to resume:
```
Found checkpoint: outputs/checkpoints/checkpoint_epoch18.pt
Resume from this checkpoint? (y/n):
```

### Monitoring with TensorBoard

```bash
tensorboard --logdir=outputs/logs
```

Then open http://localhost:6006 in your browser.

**Logged Metrics:**
- `train/loss_step`: Training loss per batch
- `train/depth_mean`, `train/depth_std`: Predicted depth statistics
- `val/loss`: Validation loss per epoch
- `loss_comparison`: Train vs validation loss comparison

See [TENSORBOARD.md](TENSORBOARD.md) for detailed documentation.

### Evaluation

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt
```

Evaluates the model on test audio with known saturation depths and reports MAE.

### Testing

```bash
python tests/test_model.py
```

Runs unit tests for all model components.

## Architecture

```
Audio Input → Encoder (MobileNetV2) → Embeddings
                                          ↓
                                      Controller (MLP)
                                          ↓
                                  Parameters [0,1] (normalized)
                                          ↓
Audio Input → Processor (Tape Saturation) → Audio Output
```

### Components

1. **SpectralEncoder**
   - Converts audio to mel-spectrogram
   - Uses custom MobileNetV2 (width_mult=2) for feature extraction
   - Applies specific normalization: mean=0.322970, std=0.278452
   - L2 normalization on embeddings
   - Output: 128-dim embedding

2. **ParameterController**
   - Takes embeddings from input and target audio
   - Concatenates and processes through 3-layer MLP
   - Output: 1 parameter in [0, 1] range

3. **TapeSaturationProcessor**
   - Applies differentiable tape saturation
   - Formula: `y = tanh(x * depth)`
   - Denormalizes parameter: [0,1] → [1,10]

### Loss Function

**Multi-Resolution STFT Loss:**
- Compares spectrograms at multiple resolutions
- FFT sizes: 1024, 2048, 8192
- Combines spectral convergence and log magnitude errors

## Dataset

The `TapeSaturationDataset` generates training pairs on-the-fly:

1. Loads random audio chunks from dataset
2. Normalizes to 0 dBFS (peak = 1.0)
3. Applies tape saturation with random depth ∈ [1, 10]
4. Returns (clean, saturated) audio pairs

**Buffer Management:**
- Preloads audio chunks into RAM for faster iteration
- Configurable buffer size and reload rate
- Dramatically improves training speed vs disk I/O

## Model Summary

```
============================================================
MODEL SUMMARY
============================================================
Encoder:         7.14 M parameters
Controller:      0.13 M parameters
Processor:       0.00 M parameters
------------------------------------------------------------
Total:           7.27 M parameters
============================================================
```

## Project Structure

```
tape-identification/
├── tape_id/                    # Source code
│   ├── models/                # Neural network models
│   │   ├── encoder.py         # SpectralEncoder (MobileNetV2-based)
│   │   ├── controller.py      # ParameterController (MLP)
│   │   ├── tape_processor.py  # TapeSaturationProcessor
│   │   └── mobilenetv2.py     # Custom MobileNetV2 implementation
│   ├── data/                  # Dataset and data loading
│   │   ├── dataset.py         # TapeSaturationDataset
│   │   └── audio.py           # AudioFile helper classes
│   ├── training/              # Training infrastructure
│   │   ├── trainer.py         # Training loop with logging
│   │   └── losses.py          # Multi-resolution STFT loss
│   └── utils.py               # Utilities (model summary, etc.)
├── scripts/
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── tests/
│   └── test_model.py          # Unit tests
├── outputs/
│   ├── checkpoints/           # Model checkpoints
│   └── logs/                  # TensorBoard logs
├── requirements.txt
├── TENSORBOARD.md             # TensorBoard documentation
└── README.md
```

## Differences from DeepAFx-ST

**Code Directly Copied:**
- ✅ `mobilenetv2.py`: Exact copy of custom MobileNetV2 encoder
- ✅ `encoder.py`: SpectralEncoder with specific normalization values
- ✅ Buffer management logic from dataset implementation

**Simplifications:**
- ❌ Removed style transfer (only parameter identification)
- ❌ Removed SPSA optimization
- ❌ Removed proxy network training
- ❌ Removed encoder freezing options
- ❌ Removed EfficientNet option (only MobileNetV2)
- ✅ Native PyTorch training (no PyTorch Lightning)
- ✅ Simplified codebase focused on single effect (tape saturation)
- ✅ Added robust error handling and checkpoint recovery
- ✅ Added comprehensive TensorBoard logging

## Training Tips

1. **Dataset**: Use diverse audio (music, speech, etc.) for better generalization
2. **Buffer Size**: Increase `buffer_size_gb` if you have RAM available for faster training
3. **Workers**: Set `num_workers > 0` only if using Linux (may cause issues on some systems)
4. **Learning Rate**: Default 3e-4 works well, reduce if training is unstable
5. **Epochs**: Model typically converges in 20-30 epochs

## Troubleshooting

**Training crashes intermittently:**
- Enable automatic recovery by setting `num_workers=0`
- Check `outputs/checkpoints/training.log` for error details
- Emergency checkpoints are saved automatically

**High GPU memory usage:**
- Reduce `batch_size`
- Reduce `audio_length`

**Slow training:**
- Increase `buffer_size_gb` to preload more audio
- Enable `num_workers > 0` (if on Linux)
- Use GPU instead of CPU

## License

This project uses code from [DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST) which is licensed under BSD-3-Clause.

See [LICENSE](LICENSE) for details.

## References

- **DeepAFx-ST Paper**: [Differentiable All-pole Filters for Time-varying Audio Systems](https://arxiv.org/abs/2211.00497)
- **DeepAFx-ST Repository**: https://github.com/adobe-research/DeepAFx-ST
- **MobileNetV2**: [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

