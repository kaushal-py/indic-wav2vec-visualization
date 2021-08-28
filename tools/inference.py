from typing import Dict, Iterable, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
from loguru import logger



def module_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        logger.debug("Module hook running..")
        print(input[0].shape, output[0].shape, output[1].shape)
        logger.debug("Completed model hook")

class Wav2VecInferenceModule:

    def __init__(self, checkpoint_path: str, arch_type: str ='large',
                    gpu_device: str = 'cuda', pretrained=True):
        
        self.device = gpu_device
        self.model = self._load_model(checkpoint_path, arch_type, pretrained)
    
    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            squeezed = output[0].squeeze()
            self._features[layer_id] = squeezed
        return fn
        
    def _load_model(self, checkpoint_path: str, arch_type: str ='large',
                        pretrained=True):
        # Argument Valdation
        assert arch_type in ['large', 'small'], \
            "Architecture type can only be 'small' or 'large'."
        # Load pytorch checkpoint
        ckpt = torch.load(checkpoint_path)
        # Set configuration based on architecture
        if arch_type == 'large':
            conf = Wav2Vec2Config(
                quantize_targets=True, 
                extractor_mode='layer_norm',
                layer_norm_first=True,
                final_dim=768, encoder_embed_dim=1024,
                latent_temp=[2.0,0.1,0.999995],
                dropout=0.0,
                attention_dropout=0.0,
                conv_bias=True,
                encoder_layerdrop= 0.00,
                dropout_input= 0.0,
                dropout_features= 0.0,
                encoder_layers=24, 
                encoder_ffn_embed_dim=4096,
                encoder_attention_heads=16,
                feature_grad_mult= 1.0)
        elif arch_type == 'small':
            conf = Wav2Vec2Config(
                quantize_targets = True,
                final_dim= 256,
                encoder_layerdrop= 0.05,
                dropout_input= 0.1,
                dropout_features= 0.1,
                feature_grad_mult= 0.1,
                encoder_embed_dim= 768)
        # Load the model in GPU
        logger.info("Loading model..")
        model = Wav2Vec2Model.build_model(conf, task=None)
        if pretrained:
            model.load_state_dict(ckpt['model'])
        model.to(device=self.device)
        model.eval()
        logger.info("Model loaded")
        return model
    
    def _postprocess(self, feats, normalize=True):
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        if normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats
    
    def get_codebook_sequence(self, wav: np.ndarray):
        feats = torch.from_numpy(wav).float()
        out = self._postprocess(feats).unsqueeze(0)    
        out = out.to(device=self.device)
        with torch.no_grad():
            z = self.model.quantize(out)
        codebook_seq = (z[1][0,:,0]*320 + z[1][0,:,1]).cpu().numpy()
        return codebook_seq
    
    def get_transformer_outputs(self, wav: np.ndarray):
        for layer_id in range(12):
            self.model.encoder.layers[layer_id].register_forward_hook(self.save_outputs_hook(layer_id))
        self._features = {}
        
        feats = torch.from_numpy(wav).float()
        out = self._postprocess(feats).unsqueeze(0)    
        out = out.to(device=self.device)
        with torch.no_grad():
            c = self.model(out)
        transformer_outs = list(self._features.values())
        return transformer_outs


if __name__ == '__main__':

    logger.debug("Running debug tests..")
    model = Wav2VecInferenceModule('data/models/CLSRIL-23.pt', 'small')
    random_wav = np.random.rand(320*100)
    logger.debug("Running test 1..")
    codebook_seq = model.get_codebook_sequence(random_wav)
    transformer_outs = model.get_transformer_outputs(random_wav)
    assert codebook_seq.shape[0] == 99, "Codebook sequence shape did not match"
    assert transformer_outs[0].shape == (99, 768), "Transformer output shape did not match"
    assert len(transformer_outs) == 12, "Number of transformer layers did not match"
    logger.debug("Running test 2..")
    random_wav = np.random.rand(320*100+80)
    codebook_seq = model.get_codebook_sequence(random_wav)
    transformer_outs = model.get_transformer_outputs(random_wav)
    assert codebook_seq.shape[0] == 100, "Codebook sequence shape did not match"
    assert transformer_outs[0].shape == (100, 768), "Transformer output shape did not match"
    assert len(transformer_outs) == 12, "Number of transformer layers did not match"
    logger.debug("Successfully completed Debug tests")
