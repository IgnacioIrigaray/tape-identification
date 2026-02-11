"""
Dataset específico para entrenar Tape Saturation.

Genera pares (x, y) donde:
- x = audio original
- y = tape_saturation(x, random_gain)

El modelo aprende a predecir el gain que transforma x en y.
"""

import os
import glob
import math
import torch
import torchaudio
import random
import numpy as np
from tqdm import tqdm
from typing import List

from .audio import AudioFile
from .. import utils


def tape_saturation(x: torch.Tensor, gain: float) -> torch.Tensor:
    """
    Aplica saturación tipo tape usando tanh.

    Args:
        x: Audio tensor [channels, samples]
        gain: Intensidad de saturación [1, 10]

    Returns:
        Audio saturado
    """
    return torch.tanh(x * gain)


# ---------------------------------------------------------------------------
# Jiles-Atherton hysteresis model
# Portado de AnalogTapeModel (Jatin Chowdhury) — RK4 solver, numba-optimizado
# ---------------------------------------------------------------------------

import numba

_JA_ALPHA = 1.6e-3      # mean-field coupling (constante)
_JA_K = 0.47875          # coercivity (constante en modo standard)
_JA_DERIV_ALPHA = 0.75   # alpha-transform para dH/dt
_JA_UPPER_LIM = 20.0     # límite de inestabilidad
_JA_CLIP_LEVEL = 10.0    # clip de entrada (RK4)


@numba.njit(cache=True)
def _ja_hysteresis_loop(x_np, M_s, a, c, k, makeup, T):
    """Loop RK4 compilado con numba. Input/output: numpy arrays."""
    alpha = 1.6e-3
    deriv_alpha = 0.75
    clip = 10.0
    upper_lim = 20.0
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

        # dH/dt alpha-transform
        H_d_n = ((1.0 + deriv_alpha) / T) * (H_n - H_n1) - deriv_alpha * H_d_n1

        # --- RK4 inline (4 evaluations of hysteresis_func) ---
        H_mid = (H_n + H_n1) * 0.5
        H_d_mid = (H_d_n + H_d_n1) * 0.5

        # Evaluate at 4 stages: (M_eval, H_eval, H_d_eval)
        # Stage params: [(M_n1, H_n1, H_d_n1), (M+k1/2, Hmid, Hdmid),
        #                (M+k2/2, Hmid, Hdmid), (M+k3, H_n, H_d_n)]
        M_eval = M_n1
        H_eval = H_n1
        H_d_eval = H_d_n1

        kk = 0.0  # acumulador RK4: M = M_n1 + kk
        for stage in range(4):
            Q = (H_eval + alpha * M_eval) / a
            # Langevin
            if abs(Q) < 0.001:
                L = Q / 3.0
                L_prime = 1.0 / 3.0
            else:
                coth_Q = 1.0 / math.tanh(Q)
                L = coth_Q - 1.0 / Q
                L_prime = 1.0 / (Q * Q) - coth_Q * coth_Q + 1.0

            M_diff = M_s * L - M_eval
            delta = 1.0 if H_d_eval >= 0.0 else -1.0
            # kappa gate
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
                # H_eval, H_d_eval ya son mid
            elif stage == 2:
                kk += ki / 3.0
                M_eval = M_n1 + ki
                H_eval = H_n
                H_d_eval = H_d_n
            else:
                kk += ki / 6.0

        M_n = M_n1 + kk

        # Guard
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
    """
    Aplica saturación con histéresis de Jiles-Atherton (modelo completo con estado).

    Portado de AnalogTapeModel (Jatin Chowdhury), solver RK4.

    Args:
        x: Audio tensor [1, samples]
        drive: Intensidad de saturación [0, 1]. Parámetro de clasificación.
        sample_rate: Tasa de muestreo
        saturation: Sat del material [0, 1] (0=headroom alto, 1=headroom bajo)
        width: Ancho de histéresis [0, 1] (0=sin histéresis, 1=máxima)

    Returns:
        Audio con histéresis [1, samples]
    """
    # Cook: user params -> JA params
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


def ja_saturation(x: torch.Tensor, drive: float, sample_rate: int = 22050,
                  saturation: float = 0.5, width: float = 0.5) -> torch.Tensor:
    """Alias para ja_hysteresis (backward compat)."""
    return ja_hysteresis(x, drive, sample_rate, saturation, width)


def hard_clipping(x: torch.Tensor, gain: float) -> torch.Tensor:
    """
    Aplica hard clipping con ganancia.

    Fórmula: y = clamp(x * gain, -1, 1)

    Args:
        x: Audio tensor [channels, samples]
        gain: Ganancia antes del clipping (gain > 1 produce clipping)

    Returns:
        Audio con hard clipping
    """
    return torch.clamp(x * gain, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Time-base distortion: Wow & Flutter
# Adaptado de AnalogTapeModel (Jatin Chowdhury)
# ---------------------------------------------------------------------------

def _ornstein_uhlenbeck(num_samples: int, sample_rate: int,
                        amount: float = 0.2, damping: float = 5.0,
                        mean: float = 0.0) -> torch.Tensor:
    """Proceso Ornstein-Uhlenbeck: random walk con mean-reversion + lowpass 10 Hz."""
    T = 1.0 / sample_rate
    sqrt_delta = math.sqrt(2.0 * T)
    alpha = 1.0 - damping * T
    beta = sqrt_delta * amount
    gamma = damping * mean * T

    noise = torch.randn(num_samples) / 2.33
    y = torch.zeros(num_samples)
    state = 0.0
    for n in range(num_samples):
        state = alpha * state + beta * noise[n].item() + gamma
        y[n] = state

    # Lowpass EMA a 10 Hz
    cutoff = 10.0
    rc = 1.0 / (2.0 * math.pi * cutoff)
    ema_alpha = T / (rc + T)
    state = y[0].item()
    for n in range(num_samples):
        state = state + ema_alpha * (y[n].item() - state)
        y[n] = state

    return y


def _wow_lfo(num_samples: int, sample_rate: int,
             wow_rate: float, wow_depth: float,
             ou_signal: torch.Tensor = None,
             max_delay_ms: float = 10.0) -> torch.Tensor:
    """Genera LFO de wow: coseno + opcional OU.

    Args:
        max_delay_ms: Delay máximo en ms cuando wow_depth=1.0.
    """
    wow_freq = 4.5 ** wow_rate - 1.0
    # Delay amplitude en samples: lineal con depth
    amplitude = wow_depth * max_delay_ms * sample_rate / 2000.0

    t = torch.arange(num_samples, dtype=torch.float32)
    phase = 2.0 * math.pi * wow_freq * t / sample_rate

    if ou_signal is not None:
        lfo = amplitude * (torch.cos(phase) + ou_signal)
    else:
        lfo = amplitude * torch.cos(phase)

    # DC offset para que delay >= 0
    dc_offset = amplitude
    return lfo + dc_offset


def _flutter_lfo(num_samples: int, sample_rate: int,
                 flutter_rate: float, flutter_depth: float,
                 max_delay_ms: float = 0.5) -> torch.Tensor:
    """Genera LFO de flutter: 3 armónicos con amplitudes fijas.

    Args:
        max_delay_ms: Delay máximo en ms cuando flutter_depth=1.0.
    """
    flutter_freq = 0.1 * (1000.0 ** flutter_rate)
    amplitude = flutter_depth * max_delay_ms * sample_rate / 2000.0

    t = torch.arange(num_samples, dtype=torch.float32)
    phase = 2.0 * math.pi * flutter_freq * t / sample_rate

    # 3 armónicos normalizados (amplitudes originales: 230, 80, 99 → peak ~409)
    raw = (-230.0 * torch.cos(phase)
           + -80.0 * torch.cos(2.0 * phase + 13.0 * math.pi / 4.0)
           + -99.0 * torch.cos(3.0 * phase - math.pi / 10.0))
    lfo = amplitude * raw / 409.0

    dc_offset = amplitude
    return lfo + dc_offset


def _variable_delay(x: torch.Tensor, delay_samples: torch.Tensor,
                    interpolation: str = "linear") -> torch.Tensor:
    """Aplica delay variable por muestra. x: [1, samples], delay_samples: [samples]."""
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
    """
    Aplica wow/flutter (time-base distortion) al audio.

    Args:
        x: Audio tensor [1, samples]
        depth: Profundidad combinada [0, 1]. Parámetro de clasificación.
        sample_rate: Tasa de muestreo
        wow_rate: Tasa de wow [0,1] → 0-3.5 Hz
        flutter_rate: Tasa de flutter [0,1] → 0.1-100 Hz
        enable_ou: Agrega proceso Ornstein-Uhlenbeck al wow
        interpolation: "linear" o "lagrange3"

    Returns:
        Audio con wow/flutter aplicado [1, samples]
    """
    num_samples = x.shape[-1]

    ou_signal = None
    if enable_ou and depth > 0.01:
        ou_signal = _ornstein_uhlenbeck(num_samples, sample_rate)

    wow = _wow_lfo(num_samples, sample_rate, wow_rate, depth, ou_signal)
    flutter = _flutter_lfo(num_samples, sample_rate, flutter_rate, depth)

    total_delay = wow + flutter
    total_delay = torch.clamp(total_delay, 0.0)

    y = _variable_delay(x, total_delay, interpolation)

    # DC blocker: highpass a 15 Hz
    y = torchaudio.functional.highpass_biquad(y, sample_rate, 15.0)

    return y


class TapeSaturationDataset(torch.utils.data.Dataset):
    """
    Dataset para entrenar modelos de tape saturation.

    Aplica tape saturation con parámetro aleatorio a cada audio.

    Args:
        audio_dir: Directorio con archivos de audio
        input_dirs: Subdirectorios a buscar
        subset: "train" o "val"
        length: Longitud en samples de cada ejemplo
        train_frac: Fracción para training
        buffer_size_gb: GB de audio a mantener en RAM
        buffer_reload_rate: Ejemplos entre recargas de buffer
        num_examples_per_epoch: Ejemplos por época
        min_param: Parámetro mínimo para saturación
        max_param: Parámetro máximo para saturación
        num_classes: Número de clases discretas
        degradation_model: "tanh", "ja" (Jiles-Atherton) o "hard_clipping"
        log_scale: Si True, usa escala logarítmica (solo para JA)
        ext: Extensión de archivos de audio
    """

    def __init__(
        self,
        audio_dir: str,
        input_dirs: List[str] = ["00", "01", "02"],
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
        log_scale: bool = True,
        ext: str = "mp3",
        sample_rate: int = None,
        # Aliases para compatibilidad
        min_gain: float = None,
        max_gain: float = None,
        **kwargs,
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.input_dirs = input_dirs
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

        # Compatibilidad con parámetros antiguos
        if min_gain is not None:
            min_param = min_gain
        if max_gain is not None:
            max_param = max_gain

        self.min_param = min_param
        self.max_param = max_param
        self.target_sample_rate = sample_rate

        # Generar valores discretos de parámetro para las N clases
        import numpy as np
        if log_scale:
            self.param_values = list(np.geomspace(min_param, max_param, num_classes))
        else:
            self.param_values = list(np.linspace(min_param, max_param, num_classes))

        # Wow/flutter config
        self.wow_rate = kwargs.get("wow_rate", 0.4)
        self.flutter_rate = kwargs.get("flutter_rate", 0.5)
        self.enable_ou = kwargs.get("enable_ou", True)
        self.wf_interpolation = kwargs.get("wf_interpolation", "linear")
        self.wf_target_param = kwargs.get("wf_target_param", "depth")  # "depth", "rate" o "both"
        self.wf_fixed_depth = kwargs.get("wf_fixed_depth", 0.5)

        # Modo dual-param: depth + rate independientes
        if self.wf_target_param == "both":
            self.num_classes_depth = kwargs.get("num_classes_depth", num_classes)
            self.num_classes_rate = kwargs.get("num_classes_rate", num_classes)
            self.depth_values = list(np.linspace(
                kwargs.get("min_depth", min_param),
                kwargs.get("max_depth", max_param),
                self.num_classes_depth,
            ))
            self.rate_values = list(np.linspace(
                kwargs.get("min_rate", min_param),
                kwargs.get("max_rate", max_param),
                self.num_classes_rate,
            ))
            print(f"Wow/Flutter DUAL param mode:")
            print(f"  Depth values ({self.num_classes_depth} classes): {[f'{p:.3f}' for p in self.depth_values]}")
            print(f"  Rate values ({self.num_classes_rate} classes): {[f'{p:.3f}' for p in self.rate_values]}")
        else:
            model_names = {
                "ja": "Jiles-Atherton", "tanh": "tanh",
                "hard_clipping": "Hard Clipping", "wow_flutter": "Wow/Flutter",
            }
            model_name = model_names.get(degradation_model, degradation_model)
            print(f"{model_name} param values ({num_classes} classes): {[f'{p:.3f}' for p in self.param_values]}")

        # Buscar archivos de audio
        self.input_filepaths = []
        for input_dir in input_dirs:
            search_path = os.path.join(audio_dir, input_dir, f"*.{ext}")
            self.input_filepaths += glob.glob(search_path)
        self.input_filepaths = sorted(self.input_filepaths)

        if len(self.input_filepaths) == 0:
            # Intentar buscar directamente en audio_dir
            search_path = os.path.join(audio_dir, f"*.{ext}")
            self.input_filepaths = glob.glob(search_path)
            self.input_filepaths = sorted(self.input_filepaths)

        # Shuffle determinista antes de split para garantizar separación reproducible
        rng_split = random.Random(42)
        rng_split.shuffle(self.input_filepaths)

        # Split train/val
        self.input_filepaths = utils.split_dataset(
            self.input_filepaths,
            subset,
            train_frac,
        )

        # Cargar metadata de archivos
        self.input_files = {}
        input_dur_frames = 0

        print(f"\nCargando metadata de {len(self.input_filepaths)} archivos...")
        for input_filepath in tqdm(self.input_filepaths, ncols=80):
            file_id = os.path.basename(input_filepath)
            audio_file = AudioFile(
                input_filepath,
                preload=False,
                half=half,
                target_sample_rate=self.target_sample_rate,
            )
            # Necesitamos al menos length samples
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

    def __len__(self):
        return self.num_examples_per_epoch

    def load_audio_buffer(self):
        """Carga un subconjunto de archivos en RAM."""
        # Descargar archivos del buffer anterior para liberar RAM
        for file_id in self.input_files_loaded:
            af = self.input_files[file_id]
            af.audio = None
            af.loaded = False

        self.input_files_loaded = {}
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

        print(f"Loaded {len(self.input_files_loaded)} files into buffer ({nbytes_loaded/1e9:.2f} GB)")

    def get_random_file_id(self):
        """Retorna un file_id aleatorio del buffer."""
        return random.choice(list(self.input_files_loaded.keys()))

    def get_random_patch(self, audio_file, length, rng=None):
        """Obtiene índices aleatorios para un patch de audio."""
        if rng is None:
            rng = random
        max_start = audio_file.num_frames - length
        if max_start <= 0:
            return -1, -1
        start_idx = rng.randint(0, max_start)
        stop_idx = start_idx + length
        return start_idx, stop_idx

    def __getitem__(self, idx):
        """
        Genera (y, class_idx) donde y = saturación(audio, param[class_idx]).
        Para validación, usa seed determinista basado en idx.
        """
        # Recargar buffer si es necesario
        self.items_since_load += 1
        if self.items_since_load > self.buffer_reload_rate:
            self.load_audio_buffer()

        rng = random

        # Obtener audio aleatorio
        file_ids = list(self.input_files_loaded.keys())
        while True:
            file_id = rng.choice(file_ids)
            audio_file = self.input_files_loaded[file_id]

            if not audio_file.loaded:
                audio_file.load()

            start_idx, stop_idx = self.get_random_patch(audio_file, self.length, rng)
            if start_idx >= 0:
                break

        # Extraer patch
        x = audio_file.audio[:, start_idx:stop_idx].clone().detach()
        x = x.view(1, -1)  # [1, samples]

        if self.half:
            x = x.float()

        # Normalizar a 0 dBFS (amplitud máxima = 1.0)
        x = x / (x.abs().max() + 1e-8)

        # Seleccionar clase(s) y aplicar degradación
        if self.degradation_model == "wow_flutter" and self.wf_target_param == "both":
            depth_idx = rng.randint(0, self.num_classes_depth - 1)
            rate_idx = rng.randint(0, self.num_classes_rate - 1)
            depth_val = self.depth_values[depth_idx]
            rate_val = self.rate_values[rate_idx]
            y = wow_flutter(x, depth_val, sample_rate=self.sample_rate,
                            wow_rate=rate_val, flutter_rate=self.flutter_rate,
                            enable_ou=self.enable_ou, interpolation=self.wf_interpolation)
            y = utils.conform_length(y, self.length)
            y = utils.linear_fade(y, sample_rate=self.sample_rate)
            return y, depth_idx, rate_idx

        class_idx = rng.randint(0, self.num_classes - 1)
        param = self.param_values[class_idx]
        if self.degradation_model == "ja":
            y = ja_saturation(x, param, sample_rate=self.sample_rate)
        elif self.degradation_model == "hard_clipping":
            y = hard_clipping(x, param)
        elif self.degradation_model == "wow_flutter":
            if self.wf_target_param == "rate":
                y = wow_flutter(x, self.wf_fixed_depth, sample_rate=self.sample_rate,
                                wow_rate=param, flutter_rate=self.flutter_rate,
                                enable_ou=self.enable_ou, interpolation=self.wf_interpolation)
            else:
                y = wow_flutter(x, param, sample_rate=self.sample_rate,
                                wow_rate=self.wow_rate, flutter_rate=self.flutter_rate,
                                enable_ou=self.enable_ou, interpolation=self.wf_interpolation)
        else:
            y = tape_saturation(x, param)

        y = utils.conform_length(y, self.length)
        y = utils.linear_fade(y, sample_rate=self.sample_rate)
        return y, class_idx
