from whisperx import load_audio
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch as torch
from pyannote.audio import Pipeline
from whisperx import load_audio
from whisperx.audio import SAMPLE_RATE


class DiarizationWithEmbeddingsPipeline:
    def __init__(
            self,
            model_name="pyannote/speaker-diarization-3.1",
            use_auth_token=None,
            device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.model = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token).to(device)

    def __call__(self, audio: Union[str, np.ndarray], min_speakers=None, max_speakers=None, return_embeddings=False):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }

        if not return_embeddings:
            segments = self.model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers, return_embeddings=False)
            embeddings = None
        else:
            segments, embeddings = self.model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers, return_embeddings=True)
            embeddings = embeddings.tolist()
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        return diarize_df, embeddings
