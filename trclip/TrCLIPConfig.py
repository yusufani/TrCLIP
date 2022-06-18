from transformers import PretrainedConfig
from typing import List
import torch


class TrCLIPConfig(PretrainedConfig):
    model_type = "trclip"

    def __init__(
        self,
        text_encoder_path: str = None,
        clip_model: str = None,
        text_encoder_base: str = 'dbmdz/bert-base-turkish-cased',
        device: str = None,
        delete_original_text_encoder: bool = True,

        **kwargs,
    ):
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"

        self.text_encoder_path = text_encoder_path
        self.clip_model = clip_model
        self.text_encoder_base = text_encoder_base
        self.delete_original_text_encoder = delete_original_text_encoder
        super().__init__(**kwargs)
