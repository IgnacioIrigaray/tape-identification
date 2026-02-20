"""
Dataset for tape saturation parameter identification.

Generates pairs (y, target) where:
- y = degradation(audio, random_param)
- target = the parameter value (normalized to [0,1] for regression, class index for classification)
"""

import gc
import os
import glob
import math
import random
from typing import List, Optional

import numpy as np
import numba
import torch
import torchaudio
from tqdm import tqdm

from .audio import AudioFile
from .. import utils

EPSILON = 1e-8

# ---------------------------------------------------------------------------
# Degradation functions
# ---------------------------------------------------------------------------


def tape_saturation(x: torch.Tensor, gain: float) -> torch.Tensor:
    """Tanh saturation: y = tanh(x * gain).

    Args:
        x: Audio tensor [channels, samples]
        gain: Saturation intensity [1, 10]
    """
    return torch.tanh(x * gain)


def hard_clipping(x: torch.Tensor, gain: float) -> torch.Tensor:
    """Hard clipping: y = clamp(x * gain, -1, 1).

    Args:
        x: Audio tensor [channels, samples]
        gain: Gain before clipping (gain > 1 produces clipping)
    """
    return torch.clamp(x * gain, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Jiles-Atherton hysteresis model
# Ported from AnalogTapeModel (Jatin Chowdhury) — RK4 solver, numba-optimized
# ---------------------------------------------------------------------------

_JA_K = 0.47875          # coercivity
_JA_CLIP_LEVEL = 10.0    # RK4 input clip level


@numba.njit(cache=True)
def _ja_hysteresis_loop(x_np, M_s, a, c, k, makeup, T):
    """RK4 hysteresis loop compiled with numba. Input/output: numpy arrays."""
    alpha = 1.6e-3        # mean-field coupling
    deriv_alpha = 0.75    # alpha-transform for dH/dt
    clip = 10.0
    upper_lim = 20.0      # instability guard
    num_samples = x_np.shape[0]
    y = np.empty(num_samples, dtype=np.float64)

    nc = 1.0 - c
    M_n1 = 0.0
    H_n1 = 0.0
    H_d_n1 = 0.0

    for n in range(num_samples):
        H_n = x_np[n]
        if H_n > clip:
            H_n = clip
        elif H_n < -clip:
            H_n = -clip

        H_d_n = ((1.0 + deriv_alpha) / T) * (H_n - H_n1) - deriv_alpha * H_d_n1

        # RK4: 4 evaluations of hysteresis derivative
        H_mid = (H_n + H_n1) * 0.5
        H_d_mid = (H_d_n + H_d_n1) * 0.5

        M_eval = M_n1
        H_eval = H_n1
        H_d_eval = H_d_n1

        kk = 0.0
        for stage in range(4):
            Q = (H_eval + alpha * M_eval) / a
            # Langevin function and derivative
            if abs(Q) < 0.001:
                L = Q / 3.0
                L_prime = 1.0 / 3.0
            else:
                coth_Q = 1.0 / math.tanh(Q)
                L = coth_Q - 1.0 / Q
                L_prime = 1.0 / (Q * Q) - coth_Q * coth_Q + 1.0

            M_diff = M_s * L - M_eval
            delta = 1.0 if H_d_eval >= 0.0 else -1.0
            d_sign = 1.0 if delta >= 0.0 else -1.0
            md_sign = 1.0 if M_diff >= 0.0 else -1.0
            kap = nc if d_sign == md_sign else 0.0

            f1_den = nc * delta * k - alpha * M_diff
            if abs(f1_den) < 1e-12:
                f1_den = 1e-12 if f1_den >= 0.0 else -1e-12
            f1 = kap * M_diff / f1_den
            f2 = L_prime * c * M_s / a
            f3 = 1.0 - L_prime * alpha * c * M_s / a
            if abs(f3) < 1e-12:
                f3 = 1e-12

            dMdt = H_d_eval * (f1 + f2) / f3
            ki = dMdt * T

            if stage == 0:
                kk = ki / 6.0
                M_eval = M_n1 + ki * 0.5
                H_eval = H_mid
                H_d_eval = H_d_mid
            elif stage == 1:
                kk += ki / 3.0
                M_eval = M_n1 + ki * 0.5
            elif stage == 2:
                kk += ki / 3.0
                M_eval = M_n1 + ki
                H_eval = H_n
                H_d_eval = H_d_n
            else:
                kk += ki / 6.0

        M_n = M_n1 + kk

        if math.isnan(M_n) or M_n > upper_lim or M_n < -upper_lim:
            M_n = 0.0
            H_d_n = 0.0

        y[n] = M_n * makeup
        M_n1 = M_n
        H_n1 = H_n
        H_d_n1 = H_d_n

    return y


def ja_hysteresis(x: torch.Tensor, drive: float, sample_rate: int = 22050,
                  saturation: float = 0.5, width: float = 0.5) -> torch.Tensor:
    """Apply Jiles-Atherton hysteresis saturation (full stateful model with RK4 solver).

    Ported from AnalogTapeModel (Jatin Chowdhury).

    Args:
        x: Audio tensor [1, samples]
        drive: Saturation intensity [0, 1] — the parameter being identified.
        sample_rate: Sample rate in Hz.
        saturation: Material saturation [0, 1] (0=high headroom, 1=low headroom).
        width: Hysteresis width [0, 1] (0=no hysteresis, 1=maximum).

    Returns:
        Audio with hysteresis [1, samples]
    """
    M_s = 0.5 + 1.5 * (1.0 - saturation)
    a = M_s / (0.01 + 6.0 * drive)
    c = math.sqrt(1.0 - width) - 0.01
    c = max(c, 0.001)
    k = _JA_K
    makeup = (1.0 + 0.6 * width) / M_s
    T = 1.0 / sample_rate

    x_np = x.squeeze(0).numpy().astype(np.float64)
    y_np = _ja_hysteresis_loop(x_np, M_s, a, c, k, makeup, T)

    y = torch.from_numpy(y_np).float().unsqueeze(0)
    y = torchaudio.functional.highpass_biquad(y, sample_rate, 35.0)
    return y


# Backward-compatible alias
ja_saturation = ja_hysteresis


# ---------------------------------------------------------------------------
# Time-base distortion: Wow & Flutter
# Adapted from AnalogTapeModel (Jatin Chowdhury)
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def _ou_kernel(noise, alpha, beta, gamma, ema_alpha):
    """Numba kernel: OU process + EMA lowpass in a single pass."""
    n = len(noise)
    y = np.empty(n, dtype=np.float32)

    # OU process
    state = 0.0
    for i in range(n):
        state = alpha * state + beta * noise[i] + gamma
        y[i] = state

    # EMA lowpass
    state = y[0]
    for i in range(n):
        state = state + ema_alpha * (y[i] - state)
        y[i] = state

    return y


def _ornstein_uhlenbeck(num_samples: int, sample_rate: int,
                        amount: float = 0.2, damping: float = 5.0,
                        mean: float = 0.0) -> torch.Tensor:
    """Ornstein-Uhlenbeck process: random walk with mean-reversion + lowpass 10 Hz."""
    T = 1.0 / sample_rate
    alpha = 1.0 - damping * T
    beta = math.sqrt(2.0 * T) * amount
    gamma = damping * mean * T

    cutoff = 10.0
    rc = 1.0 / (2.0 * math.pi * cutoff)
    ema_alpha = T / (rc + T)

    noise = np.random.randn(num_samples).astype(np.float32) / 2.33
    y = _ou_kernel(noise, alpha, beta, gamma, ema_alpha)
    return torch.from_numpy(y)


def _wow_lfo(num_samples: int, sample_rate: int,
             wow_rate: float, wow_depth: float,
             ou_signal: Optional[torch.Tensor] = None,
             max_delay_ms: float = 10.0) -> torch.Tensor:
    """Generate wow LFO: cosine + optional Ornstein-Uhlenbeck modulation.

    Args:
        max_delay_ms: Maximum delay in ms when wow_depth=1.0.
    """
    # Map wow_rate [0,1] to frequency: 4.5^rate - 1 gives ~0-3.5 Hz
    wow_freq = 4.5 ** wow_rate - 1.0
    amplitude = wow_depth * max_delay_ms * sample_rate / 2000.0

    t = torch.arange(num_samples, dtype=torch.float32)
    phase = 2.0 * math.pi * wow_freq * t / sample_rate

    if ou_signal is not None:
        lfo = amplitude * (torch.cos(phase) + ou_signal)
    else:
        lfo = amplitude * torch.cos(phase)

    dc_offset = amplitude
    return lfo + dc_offset


def _flutter_lfo(num_samples: int, sample_rate: int,
                 flutter_rate: float, flutter_depth: float,
                 max_delay_ms: float = 0.5) -> torch.Tensor:
    """Generate flutter LFO: 3 harmonics with fixed amplitude ratios.

    Harmonic amplitudes from AnalogTapeModel: 230, 80, 99 (peak ~409).

    Args:
        max_delay_ms: Maximum delay in ms when flutter_depth=1.0.
    """
    # Map flutter_rate [0,1] to frequency: 0.1 * 1000^rate gives ~0.1-100 Hz
    flutter_freq = 0.1 * (1000.0 ** flutter_rate)
    amplitude = flutter_depth * max_delay_ms * sample_rate / 2000.0

    t = torch.arange(num_samples, dtype=torch.float32)
    phase = 2.0 * math.pi * flutter_freq * t / sample_rate

    # 3 harmonics normalized (original peak amplitudes: 230 + 80 + 99 = 409)
    raw = (-230.0 * torch.cos(phase)
           + -80.0 * torch.cos(2.0 * phase + 13.0 * math.pi / 4.0)
           + -99.0 * torch.cos(3.0 * phase - math.pi / 10.0))
    lfo = amplitude * raw / 409.0

    dc_offset = amplitude
    return lfo + dc_offset


def _variable_delay(x: torch.Tensor, delay_samples: torch.Tensor,
                    interpolation: str = "linear") -> torch.Tensor:
    """Apply per-sample variable delay. x: [1, samples], delay_samples: [samples]."""
    num_samples = x.shape[-1]
    x_flat = x.squeeze(0)

    indices = torch.arange(num_samples, dtype=torch.float32) - delay_samples
    indices = torch.clamp(indices, 0.0, num_samples - 1.0)

    if interpolation == "lagrange3":
        idx_base = indices.long()
        frac = indices - idx_base.float()

        idx_m1 = torch.clamp(idx_base - 1, 0, num_samples - 1)
        idx_0 = torch.clamp(idx_base, 0, num_samples - 1)
        idx_p1 = torch.clamp(idx_base + 1, 0, num_samples - 1)
        idx_p2 = torch.clamp(idx_base + 2, 0, num_samples - 1)

        s_m1 = x_flat[idx_m1]
        s_0 = x_flat[idx_0]
        s_p1 = x_flat[idx_p1]
        s_p2 = x_flat[idx_p2]

        d = frac
        y = (s_m1 * (-d * (d - 1) * (d - 2) / 6)
             + s_0 * ((d + 1) * (d - 1) * (d - 2) / 2)
             + s_p1 * (-(d + 1) * d * (d - 2) / 2)
             + s_p2 * ((d + 1) * d * (d - 1) / 6))
    else:
        idx_floor = indices.long()
        idx_ceil = torch.clamp(idx_floor + 1, max=num_samples - 1)
        frac = indices - idx_floor.float()
        y = torch.lerp(x_flat[idx_floor], x_flat[idx_ceil], frac)

    return y.unsqueeze(0)


def wow_flutter(x: torch.Tensor, depth: float, sample_rate: int = 22050,
                wow_rate: float = 0.4, flutter_rate: float = 0.5,
                enable_ou: bool = True, interpolation: str = "linear") -> torch.Tensor:
    """Apply wow/flutter (time-base distortion) to audio.

    Args:
        x: Audio tensor [1, samples]
        depth: Combined depth [0, 1] — the parameter being identified.
        sample_rate: Sample rate in Hz.
        wow_rate: Wow rate [0,1] -> maps to ~0-3.5 Hz.
        flutter_rate: Flutter rate [0,1] -> maps to ~0.1-100 Hz.
        enable_ou: Add Ornstein-Uhlenbeck random modulation to wow.
        interpolation: "linear" or "lagrange3".

    Returns:
        Audio with wow/flutter [1, samples]
    """
    num_samples = x.shape[-1]

    ou_signal = None
    if enable_ou and depth > 0.01:
        ou_signal = _ornstein_uhlenbeck(num_samples, sample_rate)

    wow = _wow_lfo(num_samples, sample_rate, wow_rate, depth, ou_signal)
    flutter = _flutter_lfo(num_samples, sample_rate, flutter_rate, depth)

    total_delay = torch.clamp(wow + flutter, 0.0)

    y = _variable_delay(x, total_delay, interpolation)
    y = torchaudio.functional.highpass_biquad(y, sample_rate, 15.0)

    return y


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TapeSaturationDataset(torch.utils.data.Dataset):
    """Dataset that applies audio degradation on-the-fly for parameter identification.

    Supports multiple degradation models (tanh, hard_clipping, ja, wow_flutter, ja_wf)
    and both classification and regression modes.

    Args:
        audio_dir: Directory with audio files.
        input_dirs: Subdirectories to search.
        subset: "train", "val", or "test".
        length: Audio length in samples per example.
        min_param: Minimum parameter value.
        max_param: Maximum parameter value.
        num_classes: Number of discrete classes (classification mode).
        degradation_model: One of "tanh", "hard_clipping", "ja", "wow_flutter", "ja_wf".
        regression: If True, continuous parameter prediction in [0, 1].
        buffer_size_gb: GB of audio to keep in RAM.
        buffer_reload_rate: Examples between buffer reloads.
        sample_rate: Target sample rate.
    """

    def __init__(
        self,
        audio_dir: str,
        input_dirs: Optional[List[str]] = None,
        subset: str = "train",
        length: int = 65536,
        train_frac: float = 0.8,
        buffer_size_gb: float = 1.0,
        buffer_reload_rate: int = 1000,
        half: bool = False,
        num_examples_per_epoch: int = 10000,
        min_param: float = 0.2,
        max_param: float = 0.5,
        num_classes: int = 3,
        degradation_model: str = "ja",
        log_scale: bool = False,
        ext: str = "mp3",
        sample_rate: Optional[int] = None,
        regression: bool = False,
        # Wow/flutter parameters
        wow_rate: float = 0.4,
        flutter_rate: float = 0.5,
        enable_ou: bool = True,
        wf_interpolation: str = "linear",
        wf_target_param: str = "depth",
        wf_fixed_depth: float = 0.5,
        # Multi-param / triple-param ranges
        num_classes_depth: Optional[int] = None,
        num_classes_rate: Optional[int] = None,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        min_rate: Optional[float] = None,
        max_rate: Optional[float] = None,
        return_clean: bool = False,
    ):
        super().__init__()
        self.return_clean = return_clean
        self.audio_dir = audio_dir
        self.subset = subset
        self.length = length
        self.train_frac = train_frac
        self.buffer_size_gb = buffer_size_gb
        self.buffer_reload_rate = buffer_reload_rate
        self.half = half
        self.num_examples_per_epoch = num_examples_per_epoch
        self.num_classes = num_classes
        self.degradation_model = degradation_model
        self.log_scale = log_scale
        self.ext = ext
        self.regression = regression
        self.min_param = min_param
        self.max_param = max_param
        self.target_sample_rate = sample_rate

        # Wow/flutter config
        self.wow_rate = wow_rate
        self.flutter_rate = flutter_rate
        self.enable_ou = enable_ou
        self.wf_interpolation = wf_interpolation
        self.wf_target_param = wf_target_param
        self.wf_fixed_depth = wf_fixed_depth

        # Discrete parameter values for classification
        if log_scale:
            self.param_values = list(np.geomspace(min_param, max_param, num_classes))
        else:
            self.param_values = list(np.linspace(min_param, max_param, num_classes))

        # Mode-specific setup
        if degradation_model == "ja_wf":
            self._setup_triple_param(min_param, max_param, min_depth, max_depth, min_rate, max_rate)
        elif wf_target_param == "both":
            self._setup_dual_param(min_param, max_param, num_classes, num_classes_depth,
                                   num_classes_rate, min_depth, max_depth, min_rate, max_rate,
                                   regression)
        else:
            self._print_single_param_info(degradation_model, min_param, max_param,
                                          num_classes, regression)

        # Discover and split audio files
        self.input_filepaths = self._discover_files(audio_dir, input_dirs, ext)
        rng_split = random.Random(42)
        rng_split.shuffle(self.input_filepaths)
        self.input_filepaths = utils.split_dataset(self.input_filepaths, subset, train_frac)

        # Load metadata
        self.input_files = {}
        input_dur_frames = 0
        print(f"\nLoading metadata for {len(self.input_filepaths)} files...")
        for filepath in tqdm(self.input_filepaths, ncols=80):
            file_id = os.path.basename(filepath)
            audio_file = AudioFile(filepath, preload=False, half=half,
                                   target_sample_rate=self.target_sample_rate)
            if audio_file.num_frames < self.length:
                continue
            self.input_files[file_id] = audio_file
            input_dur_frames += audio_file.num_frames

        if len(self.input_files) < 1:
            raise RuntimeError(f"No files found with sufficient length in {audio_dir}")

        self.sample_rate = list(self.input_files.values())[0].sample_rate
        input_dur_hr = (input_dur_frames / self.sample_rate) / 3600
        print(f"Loaded {len(self.input_files)} files for {subset} = {input_dur_hr:.2f} hours")

        # Buffer management
        self.items_since_load = self.buffer_reload_rate
        self.input_files_loaded = {}

    def _setup_triple_param(self, min_param, max_param, min_depth, max_depth, min_rate, max_rate):
        """Configure triple-param mode: JA drive + WF depth + WF rate."""
        self.min_ja = min_param
        self.max_ja = max_param
        self.min_depth = min_depth or 0.1
        self.max_depth = max_depth or 0.8
        self.min_rate = min_rate or 0.1
        self.max_rate = max_rate or 0.8
        print(f"JA + Wow/Flutter TRIPLE param REGRESSION mode:")
        print(f"  JA drive range: [{self.min_ja:.3f}, {self.max_ja:.3f}]")
        print(f"  WF depth range: [{self.min_depth:.3f}, {self.max_depth:.3f}]")
        print(f"  WF rate range:  [{self.min_rate:.3f}, {self.max_rate:.3f}]")

    def _setup_dual_param(self, min_param, max_param, num_classes,
                          num_classes_depth, num_classes_rate,
                          min_depth, max_depth, min_rate, max_rate, regression):
        """Configure dual-param mode: depth + rate independently."""
        self.num_classes_depth = num_classes_depth or num_classes
        self.num_classes_rate = num_classes_rate or num_classes
        self.depth_values = list(np.linspace(
            min_depth or min_param, max_depth or max_param, self.num_classes_depth))
        self.rate_values = list(np.linspace(
            min_rate or min_param, max_rate or max_param, self.num_classes_rate))
        mode = "REGRESSION" if regression else f"{self.num_classes_depth}x{self.num_classes_rate} classes"
        print(f"Wow/Flutter DUAL param mode ({mode}):")
        print(f"  Depth range: [{self.depth_values[0]:.3f}, {self.depth_values[-1]:.3f}]")
        print(f"  Rate range: [{self.rate_values[0]:.3f}, {self.rate_values[-1]:.3f}]")

    def _print_single_param_info(self, degradation_model, min_param, max_param,
                                 num_classes, regression):
        """Print info for single-param mode."""
        model_names = {
            "ja": "Jiles-Atherton", "tanh": "tanh",
            "hard_clipping": "Hard Clipping", "wow_flutter": "Wow/Flutter",
        }
        name = model_names.get(degradation_model, degradation_model)
        if regression:
            print(f"{name} REGRESSION mode: param range [{min_param:.3f}, {max_param:.3f}]")
        else:
            print(f"{name} param values ({num_classes} classes): "
                  f"{[f'{p:.3f}' for p in self.param_values]}")

    @staticmethod
    def _discover_files(audio_dir: str, input_dirs: Optional[List[str]], ext: str) -> list:
        """Find all audio files in the specified directories."""
        filepaths = []
        if input_dirs is None:
            for entry in sorted(os.listdir(audio_dir)):
                subdir = os.path.join(audio_dir, entry)
                if os.path.isdir(subdir):
                    filepaths += glob.glob(os.path.join(subdir, f"*.{ext}"))
            filepaths += glob.glob(os.path.join(audio_dir, f"*.{ext}"))
        else:
            for input_dir in input_dirs:
                filepaths += glob.glob(os.path.join(audio_dir, input_dir, f"*.{ext}"))
            if not filepaths:
                filepaths += glob.glob(os.path.join(audio_dir, f"*.{ext}"))
        return sorted(filepaths)

    def __len__(self):
        return self.num_examples_per_epoch

    def load_audio_buffer(self):
        """Load a subset of audio files into RAM."""
        for file_id in self.input_files_loaded:
            af = self.input_files[file_id]
            af.audio = None
            af.loaded = False

        self.input_files_loaded = {}
        gc.collect()  # Free old tensors before loading new ones to avoid peak memory spike
        self.items_since_load = 0
        nbytes_loaded = 0
        max_bytes = self.buffer_size_gb * 1e9

        filepaths = list(self.input_files.keys())
        random.shuffle(filepaths)

        for file_id in filepaths:
            audio_file = self.input_files[file_id]
            if not audio_file.loaded:
                audio_file.load()
            self.input_files_loaded[file_id] = audio_file
            nbytes_loaded += audio_file.audio.element_size() * audio_file.audio.nelement()
            if nbytes_loaded >= max_bytes:
                break

        print(f"Loaded {len(self.input_files_loaded)} files into buffer "
              f"({nbytes_loaded / 1e9:.2f} GB)")

    def _get_random_audio(self) -> torch.Tensor:
        """Get a random audio patch from the buffer."""
        self.items_since_load += 1
        if self.items_since_load > self.buffer_reload_rate:
            self.load_audio_buffer()

        file_ids = list(self.input_files_loaded.keys())
        while True:
            file_id = random.choice(file_ids)
            audio_file = self.input_files_loaded[file_id]
            if not audio_file.loaded:
                audio_file.load()
            max_start = audio_file.num_frames - self.length
            if max_start > 0:
                break

        start_idx = random.randint(0, max_start)
        x = audio_file.audio[:, start_idx:start_idx + self.length].clone().detach()
        x = x.view(1, -1)

        if self.half:
            x = x.float()

        # Normalize to 0 dBFS
        x = x / (x.abs().max() + EPSILON)
        return x

    def __getitem__(self, idx):
        x = self._get_random_audio()

        if self.degradation_model == "ja_wf":
            return self._apply_triple_param(x)

        if self.degradation_model == "wow_flutter" and self.wf_target_param == "both":
            return self._apply_dual_param(x)

        return self._apply_single_param(x)

    def _apply_triple_param(self, x: torch.Tensor):
        """Apply JA + WF degradation, return (y, ja_target, depth_target, rate_target)."""
        ja_val = random.uniform(self.min_ja, self.max_ja)
        depth_val = random.uniform(self.min_depth, self.max_depth)
        rate_val = random.uniform(self.min_rate, self.max_rate)

        y = ja_hysteresis(x, ja_val, sample_rate=self.sample_rate)
        y = wow_flutter(y, depth_val, sample_rate=self.sample_rate,
                        wow_rate=rate_val, flutter_rate=self.flutter_rate,
                        enable_ou=self.enable_ou, interpolation=self.wf_interpolation)
        y = utils.conform_length(y, self.length)
        y = utils.linear_fade(y, sample_rate=self.sample_rate)

        ja_target = (ja_val - self.min_ja) / (self.max_ja - self.min_ja)
        depth_target = (depth_val - self.min_depth) / (self.max_depth - self.min_depth)
        rate_target = (rate_val - self.min_rate) / (self.max_rate - self.min_rate)
        targets = (torch.tensor(ja_target, dtype=torch.float32),
                   torch.tensor(depth_target, dtype=torch.float32),
                   torch.tensor(rate_target, dtype=torch.float32))
        if self.return_clean:
            x_clean = utils.linear_fade(utils.conform_length(x.clone(), self.length),
                                        sample_rate=self.sample_rate)
            return (x_clean, y) + targets
        return (y,) + targets

    def _apply_dual_param(self, x: torch.Tensor):
        """Apply WF with independent depth+rate, return (y, depth_target, rate_target)."""
        if self.regression:
            min_d, max_d = self.depth_values[0], self.depth_values[-1]
            min_r, max_r = self.rate_values[0], self.rate_values[-1]
            depth_val = random.uniform(min_d, max_d)
            rate_val = random.uniform(min_r, max_r)
            depth_target = (depth_val - min_d) / (max_d - min_d)
            rate_target = (rate_val - min_r) / (max_r - min_r)
        else:
            depth_idx = random.randint(0, self.num_classes_depth - 1)
            rate_idx = random.randint(0, self.num_classes_rate - 1)
            depth_val = self.depth_values[depth_idx]
            rate_val = self.rate_values[rate_idx]

        y = wow_flutter(x, depth_val, sample_rate=self.sample_rate,
                        wow_rate=rate_val, flutter_rate=self.flutter_rate,
                        enable_ou=self.enable_ou, interpolation=self.wf_interpolation)
        y = utils.conform_length(y, self.length)
        y = utils.linear_fade(y, sample_rate=self.sample_rate)

        if self.regression:
            if self.return_clean:
                x_clean = utils.linear_fade(utils.conform_length(x.clone(), self.length),
                                            sample_rate=self.sample_rate)
                return (x_clean, y,
                        torch.tensor(depth_target, dtype=torch.float32),
                        torch.tensor(rate_target, dtype=torch.float32))
            return (y,
                    torch.tensor(depth_target, dtype=torch.float32),
                    torch.tensor(rate_target, dtype=torch.float32))
        return y, depth_idx, rate_idx

    def _apply_single_param(self, x: torch.Tensor):
        """Apply single-param degradation, return (y, target)."""
        if self.regression:
            param = random.uniform(self.min_param, self.max_param)
            target = (param - self.min_param) / (self.max_param - self.min_param)
        else:
            class_idx = random.randint(0, self.num_classes - 1)
            param = self.param_values[class_idx]

        y = self._apply_degradation(x, param)
        y = utils.conform_length(y, self.length)
        y = utils.linear_fade(y, sample_rate=self.sample_rate)

        if self.regression:
            if self.return_clean:
                x_clean = utils.linear_fade(utils.conform_length(x.clone(), self.length),
                                            sample_rate=self.sample_rate)
                return x_clean, y, torch.tensor(target, dtype=torch.float32)
            return y, torch.tensor(target, dtype=torch.float32)
        return y, class_idx

    def _apply_degradation(self, x: torch.Tensor, param: float) -> torch.Tensor:
        """Apply the configured degradation model to audio."""
        if self.degradation_model == "ja":
            return ja_hysteresis(x, param, sample_rate=self.sample_rate)
        elif self.degradation_model == "hard_clipping":
            return hard_clipping(x, param)
        elif self.degradation_model == "wow_flutter":
            if self.wf_target_param == "rate":
                return wow_flutter(x, self.wf_fixed_depth, sample_rate=self.sample_rate,
                                   wow_rate=param, flutter_rate=self.flutter_rate,
                                   enable_ou=self.enable_ou, interpolation=self.wf_interpolation)
            return wow_flutter(x, param, sample_rate=self.sample_rate,
                               wow_rate=self.wow_rate, flutter_rate=self.flutter_rate,
                               enable_ou=self.enable_ou, interpolation=self.wf_interpolation)
        else:
            return tape_saturation(x, param)
