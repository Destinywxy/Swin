import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .vecquantize.vit_quantizers import VectorQuantizer
from .vecquantize.taming_quantizers import VectorQuantizer2 as TamingQuantizer
from .resnet_layers import BasicBlock, Bottleneck, ResEncoder, ResDecoder
from .vit_layers import ViTEncoder, ViTDecoderForCls

class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        type = config.DISCRETE_MODEL.TYPE
        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config)

    def forward(self, input):
        h = self.encode(input)
        logits = self.decode(h)
        return logits

    def encode(self, x):
        h = self.encoder(x)
        return h
    
    def decode(self, h):
        logits = self.decoder(h)
        return logits# 没有经过softmax
    

def build_encoder(config, is_pretrain=False):
    # 暂时只考虑encoder decoder 同构
    type = config.DISCRETE_MODEL.TYPE
    if type == 'resnet':
        encoder = ResEncoder(Bottleneck,
                             enc_layers = config.DISCRETE_MODEL.RESNET.ENCODER.LAYERS)
    elif type == "vit" or "continuous_vit":
        encoder = ViTEncoder(image_size = config.DATA.IMG_SIZE, 
                             patch_size = config.DISCRETE_MODEL.VIT.PATCH_SIZE,
                             dim = config.DISCRETE_MODEL.VIT.ENCODER.DIM,
                             depth = config.DISCRETE_MODEL.VIT.ENCODER.DEPTHS, 
                             heads =config.DISCRETE_MODEL.VIT.ENCODER.HEADS, 
                             mlp_dim = config.DISCRETE_MODEL.VIT.ENCODER.MLP_DIM, 
                             channels = 3, 
                             dim_head = 64)

    else:
        raise NotImplementedError
    return encoder
    
def build_decoder(config, is_pretrain=False):
    type = config.DISCRETE_MODEL.TYPE
    if type == 'resnet':
        decoder = ResDecoder(Bottleneck,
                             dec_layers = config.DISCRETE_MODEL.RESNET.DECODER.LAYERS)
    elif type == "vit" or "continuous_vit":
        decoder = ViTDecoderForCls(image_size = config.DATA.IMG_SIZE, 
                                   patch_size = config.DISCRETE_MODEL.VIT.PATCH_SIZE,
                                   dim = config.DISCRETE_MODEL.VIT.DECODER.DIM,
                                   depth = config.DISCRETE_MODEL.VIT.DECODER.DEPTHS, 
                                   heads =config.DISCRETE_MODEL.VIT.DECODER.HEADS, 
                                   mlp_dim = config.DISCRETE_MODEL.VIT.DECODER.MLP_DIM, 
                                   channels = 3, 
                                   dim_head = 64,
                                   num_classes=config.DISCRETE_MODEL.VIT.DECODER.NUM_CLASSES)
    else:
        raise NotImplementedError
    return decoder