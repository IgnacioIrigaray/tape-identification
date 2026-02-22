"""
Apply a tape degradation model to audio files with fixed parameters.

Usage:
    python scripts/process_audio.py --input /path/to/audio --output /path/to/out \
        --model ja --drive 5.0

    python scripts/process_audio.py --input /path/to/audio --output /path/to/out \
        --model tanh --gain 3.0

    python scripts/process_audio.py --input /path/to/audio --output /path/to/out \
        --model ja_wf --drive 4.0 --depth 0.4 --rate 0.3

    python scripts/process_audio.py --input /path/to/audio --output /path/to/out \
        --model wow_flutter --depth 0.5 --wow_rate 0.4

    python scripts/process_audio.py --input /path/to/audio --output /path/to/out \
        --model tape_noise --snr 20.0 --noise_dir /path/to/MagTapeDB

Supported models:
    tanh          Tanh saturation             --gain  [1, 10]
    hard_clipping Hard clipping               --gain  [1, 4]
    ja            Jiles-Atherton hysteresis   --drive [1, 10]
    wow_flutter   Wow + flutter               --depth [0.1, 0.8]  --wow_rate [0.1, 0.8]
    ja_wf         JA + wow/flutter            --drive [1, 10]  --depth [0.1, 0.8]  --rate [0.1, 0.8]
    tape_noise    Additive tape noise         --snr   [5, 40] dB  --noise_dir <MagTapeDB path>
                                              or --noise_file <single noise file>
"""

import argparse
import random
import sys
from pathlib import Path

import torch
import torchaudio

# Add repo root to path so tape_id is importable without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tape_id.data.dataset import (
    tape_saturation, hard_clipping, ja_saturation, wow_flutter, tape_noise,
)
from tape_id import utils

EPSILON = 1e-8


SAMPLE_RATE = 22050


def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    """Load audio, convert to mono, resample if needed. Returns [1, samples]."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    # Peak-normalize to 0 dBFS
    peak = wav.abs().max()
    if peak > 1e-8:
        wav = wav / peak
    return wav


def apply_degradation(audio: torch.Tensor, args) -> torch.Tensor:
    model = args.model
    sr = SAMPLE_RATE

    if model == "tanh":
        return tape_saturation(audio, gain=args.gain)

    elif model == "hard_clipping":
        return hard_clipping(audio, gain=args.gain)

    elif model == "ja":
        return ja_saturation(audio, drive=args.drive, sample_rate=sr)

    elif model == "wow_flutter":
        return wow_flutter(
            audio,
            depth=args.depth,
            sample_rate=sr,
            wow_rate=args.wow_rate,
            flutter_rate=args.flutter_rate,
            enable_ou=not args.no_ou,
            interpolation=args.interpolation,
        )

    elif model == "ja_wf":
        y = ja_saturation(audio, drive=args.drive, sample_rate=sr)
        y = wow_flutter(
            y,
            depth=args.depth,
            sample_rate=sr,
            wow_rate=args.rate,
            flutter_rate=args.flutter_rate,
            enable_ou=not args.no_ou,
            interpolation=args.interpolation,
        )
        return y

    elif model == "tape_noise":
        noise = args._noise_clip
        if noise is None:
            raise ValueError("tape_noise requires --noise_dir or --noise_file")
        # Match noise length to input
        length = audio.shape[-1]
        if noise.shape[-1] < length:
            repeats = (length // noise.shape[-1]) + 1
            noise = noise.repeat(1, repeats)
        max_start = noise.shape[-1] - length
        start = random.randint(0, max_start) if max_start > 0 else 0
        noise = noise[:, start:start + length]
        noise = noise / (noise.abs().max().clamp(min=EPSILON))
        return tape_noise(audio, snr_db=args.snr, noise=noise)

    else:
        raise ValueError(f"Unknown model: {model}")


def _load_noise_files(noise_dir: str, noise_ext: str) -> list:
    """Return list of noise file paths from a directory (recursive)."""
    return sorted(Path(noise_dir).rglob(f"*.{noise_ext}"))


def main():
    parser = argparse.ArgumentParser(description="Apply tape degradation to audio files")
    parser.add_argument("--input", required=True,
                        help="Input file or directory")
    parser.add_argument("--output", required=True,
                        help="Output directory")
    parser.add_argument("--model", required=True,
                        choices=["tanh", "hard_clipping", "ja", "wow_flutter", "ja_wf",
                                 "tape_noise"],
                        help="Degradation model")
    parser.add_argument("--ext", default=None,
                        help="Extension filter when input is a directory (e.g. mp3, wav)")

    # Model parameters
    parser.add_argument("--gain", type=float, default=3.0,
                        help="Gain (tanh / hard_clipping). Default: 3.0")
    parser.add_argument("--drive", type=float, default=5.0,
                        help="JA drive [1, 10]. Default: 5.0")
    parser.add_argument("--depth", type=float, default=0.4,
                        help="Wow/flutter depth [0.1, 0.8]. Default: 0.4")
    parser.add_argument("--wow_rate", type=float, default=0.4,
                        help="Wow rate [0.1, 0.8] (wow_flutter model). Default: 0.4")
    parser.add_argument("--rate", type=float, default=0.3,
                        help="Wow rate [0.1, 0.8] (ja_wf model). Default: 0.3")
    parser.add_argument("--flutter_rate", type=float, default=0.5,
                        help="Flutter rate [0, 1]. Default: 0.5")
    parser.add_argument("--no_ou", action="store_true",
                        help="Disable Ornstein-Uhlenbeck modulation for wow")
    parser.add_argument("--interpolation", default="linear",
                        choices=["linear", "lagrange3"],
                        help="Delay interpolation method. Default: linear")
    # tape_noise parameters
    parser.add_argument("--snr", type=float, default=20.0,
                        help="Target SNR in dB (tape_noise model). Default: 20.0")
    parser.add_argument("--noise_dir", default=None,
                        help="Directory of MagTapeDB noise files (tape_noise model)")
    parser.add_argument("--noise_file", default=None,
                        help="Single noise file to use (tape_noise model)")
    parser.add_argument("--noise_ext", default="wav",
                        help="Noise file extension when using --noise_dir. Default: wav")

    # Output
    parser.add_argument("--fade", action="store_true",
                        help="Apply 50 ms fade-in/out to output")
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE,
                        help=f"Sample rate. Default: {SAMPLE_RATE}")
    parser.add_argument("--format", default="wav",
                        choices=["wav", "flac", "mp3"],
                        help="Output format. Default: wav")

    args = parser.parse_args()

    global SAMPLE_RATE
    SAMPLE_RATE = args.sample_rate

    # Pre-load noise files for tape_noise model
    args._noise_clip = None
    noise_files = []
    if args.model == "tape_noise":
        if args.noise_file:
            noise_files = [Path(args.noise_file)]
        elif args.noise_dir:
            noise_files = _load_noise_files(args.noise_dir, args.noise_ext)
        if not noise_files:
            print("Error: --noise_dir or --noise_file required for tape_noise model.")
            sys.exit(1)
        print(f"Noise files : {len(noise_files)}")
        # Load all noise into a list of tensors (they're small)
        noise_clips = []
        for nf in noise_files:
            n, sr = torchaudio.load(str(nf))
            if n.shape[0] > 1:
                n = n.mean(dim=0, keepdim=True)
            if sr != args.sample_rate:
                n = torchaudio.functional.resample(n, sr, args.sample_rate)
            noise_clips.append(n)
        # Concatenate all noise into one long tensor for easy random seeking
        args._noise_clip = torch.cat(noise_clips, dim=-1)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        exts = [args.ext] if args.ext else ["wav", "mp3", "flac", "ogg", "aiff"]
        files = []
        for ext in exts:
            files.extend(sorted(input_path.rglob(f"*.{ext}")))
    else:
        print(f"Error: {input_path} does not exist.")
        sys.exit(1)

    if not files:
        print(f"No audio files found in {input_path}")
        sys.exit(1)

    print(f"Model : {args.model}")
    if args.model in ("tanh", "hard_clipping"):
        print(f"Gain  : {args.gain}")
    if args.model in ("ja", "ja_wf"):
        print(f"Drive : {args.drive}")
    if args.model in ("wow_flutter", "ja_wf"):
        print(f"Depth : {args.depth}")
        rate_val = args.wow_rate if args.model == "wow_flutter" else args.rate
        print(f"Rate  : {rate_val}")
    if args.model == "tape_noise":
        print(f"SNR   : {args.snr} dB")
    print(f"Files : {len(files)}")
    print()

    for fpath in files:
        try:
            audio = load_audio(fpath, args.sample_rate)
            degraded = apply_degradation(audio, args)

            if args.fade:
                degraded = utils.linear_fade(degraded, sample_rate=args.sample_rate)

            out_name = fpath.stem + f"_{args.model}." + args.format
            out_path = output_dir / out_name
            torchaudio.save(str(out_path), degraded, args.sample_rate)
            print(f"  {fpath.name}  →  {out_path.name}")

        except Exception as e:
            print(f"  ERROR {fpath.name}: {e}")

    print(f"\nDone. Output in: {output_dir}")


if __name__ == "__main__":
    main()
