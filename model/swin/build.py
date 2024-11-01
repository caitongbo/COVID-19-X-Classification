# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import nn
from .swin_transformer2 import SwinTransformer
from .swin_mlp import SwinMLP
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(pretrained=False, out_features=None, path=None):
    model_type = 'swin'
    if model_type == 'swin':
        model = SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=3,
                                embed_dim=128,
                                depths=[ 2, 2, 18, 2 ],
                                num_heads= [4, 8, 16, 32 ],
                                window_size=7,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.5,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
                                # cls_num_heads=1,cls_num_cls=3, lam=0.3)
        # model.load_state_dict(torch.load("/home/ubuntu/caitongbo/COVID-Efficientnet-Pytorch/swin_large_patch4_window7_224_22k.pth", map_location=device))
    # elif model_type == 'swin_mlp':
    #     model = SwinMLP(img_size=config.DATA.IMG_SIZE,
    #                     patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
    #                     in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
    #                     num_classes=config.MODEL.NUM_CLASSES,
    #                     embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
    #                     depths=config.MODEL.SWIN_MLP.DEPTHS,
    #                     num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
    #                     window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
    #                     mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
    #                     drop_rate=config.MODEL.DROP_RATE,
    #                     drop_path_rate=config.MODEL.DROP_PATH_RATE,
    #                     ape=config.MODEL.SWIN_MLP.APE,
    #                     patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
    #                     use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    
    if pretrained:
        print('Load Swin pretrained model')
        state_dict = torch.load('/home/ubuntu/caitongbo/COVID-Efficientnet-Pytorch/ckpt/swin_base_patch4_window7_224_22k.pth')
        model.load_state_dict(state_dict['model'], strict=False)
        model.head = nn.Linear(21841, 3) 

    return model
