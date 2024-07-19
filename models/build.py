# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

#from .swin_transformer import SwinTransformer
#from .swin_transformer_v2 import SwinTransformerV2
#from .swin_transformer_moe import SwinTransformerMoE
from .swin_mlp import SwinMLP
from .resnet_bu import resnet50# 先简单写一个版本，跑起来看看
from .vit_bu import ViT
from .simmim import build_simmim

from .discrete_model import DiscreteModel

def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if is_pretrain:
        model = build_simmim(config)
        return model

        
    if model_type == "resnet50":
        model = resnet50(version='v1')
            
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

def build_dicrete_model(config):    
    if config.DISCRETE_MODEL.TYPE == 'resnet':
        model = DiscreteModel(config,
                              input_resolution = config.DATA.IMG_SIZE,
                              embed_dim = config.DISCRETE_MODEL.EMBED_DIM,
                              n_embed = config.DISCRETE_MODEL.N_EMBED,
                              z_channels = config.DISCRETE_MODEL.RESNET.Z_CHANNELS,
                              output_dim = config.CLIP_EMBED)
    elif config.DISCRETE_MODEL.TYPE == 'vit':
        model = DiscreteModel(config, 
                              embed_dim = config.DISCRETE_MODEL.EMBED_DIM,
                              n_embed = config.DISCRETE_MODEL.N_EMBED,
                              z_channels = config.DISCRETE_MODEL.VIT.Z_CHANNELS)
    elif config.DISCRETE_MODEL.TYPE == 'continuous_vit':
        model = ViT(config)

    
    else:
        raise NotImplementedError
    
    return model
