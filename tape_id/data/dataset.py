"""
Dataset específico para entrenar Tape Saturation.

Genera pares (x, y) donde:
- x = audio original
- y = tape_saturation(x, random_depth)

El modelo aprende a predecir el depth que transforma x en y.
"""

import os
import glob
import torch
import random
from tqdm import tqdm
from typing import List

from .audio import AudioFile
from .. import utils


def tape_saturation(x: torch.Tensor, depth: float) -> torch.Tensor:
    """
    Aplica saturación tipo tape.

    Args:
        x: Audio tensor [channels, samples]
        depth: Intensidad de saturación [1, 10]

    Returns:
        Audio saturado
    """
    saturated = torch.tanh(x * depth)
    #wet = depth / 2.0
    output =  saturated
    return output


class TapeSaturationDataset(torch.utils.data.Dataset):
    """
    Dataset para entrenar modelos de tape saturation.

    Aplica tape saturation con depth aleatorio a cada audio.

    Args:
        audio_dir: Directorio con archivos de audio
        input_dirs: Subdirectorios a buscar
        subset: "train" o "val"
        length: Longitud en samples de cada ejemplo
        train_frac: Fracción para training
        buffer_size_gb: GB de audio a mantener en RAM
        buffer_reload_rate: Ejemplos entre recargas de buffer
        num_examples_per_epoch: Ejemplos por época
        min_depth: Depth mínimo para saturación
        max_depth: Depth máximo para saturación
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
        min_depth: float = 1.0,
        max_depth: float = 10.0,
        ext: str = "mp3",
        **kwargs,  # Ignorar argumentos extra
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
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.ext = ext

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
        self.input_files_loaded = {}
        self.items_since_load = 0
        nbytes_loaded = 0
        max_bytes = self.buffer_size_gb * 1e9

        # Shuffle files
        filepaths = list(self.input_files.keys())
        random.shuffle(filepaths)

        for file_id in filepaths:
            audio_file = self.input_files[file_id]

            # Cargar si no está cargado
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

    def get_random_patch(self, audio_file, length):
        """Obtiene índices aleatorios para un patch de audio."""
        max_start = audio_file.num_frames - length
        if max_start <= 0:
            return -1, -1
        start_idx = random.randint(0, max_start)
        stop_idx = start_idx + length
        return start_idx, stop_idx

    def __getitem__(self, _):
        """
        Genera un par (x, y) donde y = tape_saturation(x, random_depth).
        """
        # Recargar buffer si es necesario
        self.items_since_load += 1
        if self.items_since_load > self.buffer_reload_rate:
            self.load_audio_buffer()

        # Obtener audio aleatorio
        while True:
            file_id = self.get_random_file_id()
            audio_file = self.input_files_loaded[file_id]

            if not audio_file.loaded:
                audio_file.load()

            start_idx, stop_idx = self.get_random_patch(audio_file, self.length)
            if start_idx >= 0:
                break

        # Extraer patch
        x = audio_file.audio[:, start_idx:stop_idx].clone().detach()
        x = x.view(1, -1)  # [1, samples]

        if self.half:
            x = x.float()

        # Normalizar a 0 dBFS (amplitud máxima = 1.0)
        x = x / (x.abs().max() + 1e-8)

        # Aplicar tape saturation con depth aleatorio
        depth = random.uniform(self.min_depth, self.max_depth)
        y = tape_saturation(x, depth)

        # Conformar longitud
        x = utils.conform_length(x, self.length)
        y = utils.conform_length(y, self.length)

        # Aplicar fade
        x = utils.linear_fade(x, sample_rate=self.sample_rate)
        y = utils.linear_fade(y, sample_rate=self.sample_rate)

        return x, y
