import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .vecquantize.vit_quantizers import VectorQuantizer
from .vecquantize.taming_quantizers import VectorQuantizer2 as TamingQuantizer
from .resnet_layers import BasicBlock, Bottleneck, ResEncoder, ResDecoder, AttentionPool2d
from .vit_layers import ViTEncoder, ViTDecoderForCls

class DiscreteModel(nn.Module):
    def __init__(self, config, input_resolution, embed_dim, n_embed, z_channels, output_dim = 1024):
        super().__init__()
        type = config.DISCRETE_MODEL.TYPE
        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config)
        self.output_dim = output_dim
        self.spacial_dim = input_resolution // 8
        self.heads = 8
        if type == "resnet":
            self.quantizer = TamingQuantizer(n_embed, embed_dim, beta=0.25,
                                         remap=None, sane_index_shape=False)
            self.pre_quant = torch.nn.Conv2d(z_channels, embed_dim, 1)# Encoder最后会映射到z_channels
            self.post_quant = torch.nn.Conv2d(embed_dim, z_channels, 1)
            self.attnpool = AttentionPool2d(self.spacial_dim, z_channels, self.heads, self.output_dim)
        elif type == "vit":
            self.quantizer = VectorQuantizer(embed_dim, n_embed)
            self.pre_quant = nn.Linear(z_channels, embed_dim)
            self.post_quant = nn.Linear(embed_dim, z_channels)

    def forward(self, input):   
        quant, diff, _ = self.encode(input)
        logits = self.decode(quant)
        return logits, diff

    def encode(self, x):
        h = self.encoder(x)
        h = self.pre_quant(h)
        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info
    
    def decode(self, quant):
        quant = self.post_quant(quant)
        logits = self.decoder(quant)
        return logits# 没有经过softmax

    def kd(self, x):
        h = self.encoder(x)
        h = self.pre_quant(h)
        quant, emb_loss, info = self.quantizer(h)
        quant = self.post_quant(quant)
        feature = self.attnpool(quant)
        return feature, emb_loss
    
def build_encoder(config, is_pretrain=False):
    # 暂时只考虑encoder decoder 同构
    type = config.DISCRETE_MODEL.TYPE
    if type == 'resnet':
        encoder = ResEncoder(Bottleneck,
                             enc_layers = config.DISCRETE_MODEL.RESNET.ENCODER.LAYERS)
    elif type == "vit":
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
    elif type == "vit":
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
