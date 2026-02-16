import torchaudio


class AudioFile(object):
    """Audio file wrapper with lazy loading and optional resampling.

    Args:
        filepath (str): Path to audio file.
        preload (bool): If set, load audio data into RAM immediately.
        half (bool): If set, store audio as float16 to save space.
        target_sample_rate (int, optional): Resample to this rate on load.
    """

    def __init__(self, filepath, preload=False, half=False, target_sample_rate=None):
        super().__init__()

        self.filepath = filepath
        self.half = half
        self.target_sample_rate = target_sample_rate
        self.loaded = False

        if preload:
            self.load()
            num_frames = self.audio.shape[-1]
            num_channels = self.audio.shape[0]
        else:
            metadata = torchaudio.info(filepath)
            self.sample_rate = metadata.sample_rate
            num_frames = metadata.num_frames
            num_channels = metadata.num_channels

            if target_sample_rate and self.sample_rate != target_sample_rate:
                num_frames = int(num_frames * target_sample_rate / self.sample_rate)
                self.sample_rate = target_sample_rate

        self.num_frames = num_frames
        self.num_channels = num_channels

    def load(self):
        audio, sr = torchaudio.load(self.filepath, normalize=True)

        if self.target_sample_rate is not None and sr != self.target_sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.target_sample_rate)(audio)
            sr = self.target_sample_rate

        self.audio = audio
        self.sample_rate = sr
        self.num_frames = audio.shape[-1]

        if self.half:
            self.audio = audio.half()

        self.loaded = True
