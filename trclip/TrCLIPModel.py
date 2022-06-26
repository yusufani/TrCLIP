from transformers import PreTrainedModel

from trclip.TrCLIPConfig import TrCLIPConfig

import torch.nn as nn
from transformers import BertModel
import torch
import numpy as np
import pickle
import os
import clip
from transformers import AutoTokenizer , AutoModel
from tqdm import tqdm

class TrCLIPModel(PreTrainedModel):
    config_class = TrCLIPConfig

    def __init__(self, config):
        super().__init__(config)
        #self.device = config.device
        #print(f'Using device {self.device}')
        self.clip_model, self.compose = clip.load(config.clip_model, jit=False, device=config.device)
        #print(f'Clip Model loaded')
        if config.delete_original_text_encoder:
            del self.clip_model.transformer  # delete original text encoder
        text_encoder_model = ModifiedTextEncoder(config.text_encoder_base, self.get_embedding_size(config.clip_model))
        text_encoder_model.load_state_dict(
            torch.load(config.text_encoder_path, map_location=config.device))
        text_encoder_model.eval()
        text_encoder_model.to(config.device)
        self.text_encoder = text_encoder_model
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_base)

    def forward(self, tensor):
        return self.model.forward_features(tensor)
    def get_image_features(self, images, return_numpy=False, verbose=False):
        images = tqdm(images) if verbose else images
        with torch.no_grad():
            img_inputs = torch.stack([self.compose(image).to(self.device) for image in images])
            image_embs = self.clip_model.encode_image(img_inputs).float()
            image_features = image_embs / image_embs.norm(dim=-1, keepdim=True)
        return image_features if not return_numpy else image_features.detach().cpu().numpy()

    def get_text_features(self, texts, use_original_text_encoder=False, return_numpy=False, verbose=False):
        if not use_original_text_encoder:
            encoding = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True,
                                                        add_special_tokens=True, verbose=verbose)
            with torch.no_grad():
                text_embs = self.text_encoder(encoding["input_ids"].to(self.device),
                                              encoding['attention_mask'].to(self.device))
                text_features = text_embs / text_embs.norm(dim=-1, keepdim=True)
            return text_features if not return_numpy else text_features.detach().cpu().numpy()
        else:
            with torch.no_grad():
                text = clip.tokenize(texts).to(self.device)
                text_features = self.clip_model.encode_text(text).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
                return text_features if not return_numpy else text_features.detach().cpu().numpy()

    def get_results(self, images=None, texts=None, mode='per_image', text_features=None, image_features=None,
                    use_original_text_encoder=False):
        if mode not in ['per_image', 'per_text']:
            raise ValueError('Mode must be either per_image or per_text')
        with torch.no_grad():
            image_features = self.get_image_features(images) if image_features is None else image_features
            text_features = self.get_text_features(texts,
                                                   use_original_text_encoder=use_original_text_encoder) if text_features is None else text_features
            logit_scale = self.clip_model.logit_scale.exp().float().to(self.device)
            # cosine similarity as logits
            if mode == 'per_image':
                per_mode_probs = logit_scale * image_features @ text_features.t()
            elif mode == 'per_text':
                per_mode_probs = logit_scale * text_features @ image_features.t()

            per_mode_probs = per_mode_probs.softmax(dim=-1).cpu().detach().numpy()
        # per_mode_probs = np.around(per_mode_probs, decimals=2)
        per_mode_indices = [np.argsort(prob)[::-1] for prob in per_mode_probs]
        return per_mode_indices, per_mode_probs

    def get_embedding_size(self, clip_type):
        data = {'ViT-B/32': 512,
                'ViT-B/16': 512,
                'RN101': 512,
                'ViT-L/14': 768,
                'RN50x16': 768,
                'RN50x4': 640,
                'RN50': 1024}
        return data[clip_type]

class ModifiedTextEncoder(nn.Module):
    def __init__(self, model_base, embeddingSize=640):
        super(ModifiedTextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_base)
        self.dropout = nn.Dropout()
        ### New layers:
        self.linear1 = nn.Linear(768, embeddingSize)
        self.linear2 = nn.Linear(embeddingSize, embeddingSize)

    def forward(self, ids, mask):
        outputs = self.bert(
            ids,
            attention_mask=mask)
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        # hidden_states = outputs[1]
        sequence_output = outputs[0]
        bert_output = sequence_output[:, 0, :].view(-1, 768)
        x = self.dropout(bert_output)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
