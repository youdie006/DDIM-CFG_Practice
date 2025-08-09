from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import DownSample, ResBlock, Swish, TimeEmbedding, UpSample
from torch.nn import init


class UNet(nn.Module):
    def __init__(self, T=1000, image_resolution=64, ch=128, ch_mult=[1,2,2,2], attn=[1], num_res_blocks=4, dropout=0.1, use_cfg=False, cfg_dropout=0.1, num_classes=None):
        super().__init__()
        self.image_resolution = image_resolution
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        # self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.time_embedding = TimeEmbedding(tdim)

        # classifier-free guidance
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        if use_cfg:
            assert num_classes is not None
            cdim = tdim
            self.class_embedding = nn.Embedding(num_classes+1, cdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, timestep, class_label=None):
        # Timestep embedding
        temb = self.time_embedding(timestep)

        if self.use_cfg and class_label is not None:
            if self.training:
                assert not torch.any(class_label == 0) # 0 for null.
                
                ######## TODO ########
                # DO NOT change the code outside this part.
                # Assignment 2-2. Implement random null conditioning in CFG training.
                
                # CFG training: 일정 확률로 클래스 라벨을 null(0)로 교체
                mask = torch.rand(class_label.shape[0], device=class_label.device) < self.cfg_dropout
                class_label = torch.where(mask, torch.zeros_like(class_label), class_label)
                
                # cfg_dropout 확률로 class_label을 0으로 변경
                # 이를 통해 모델이 conditional과 unconditional 둘 다 학습
                
                #######################
            
            ######## TODO ########
            # DO NOT change the code outside this part.
            # Assignment 2-1. Implement class conditioning
            
            # 1. 클래스 라벨을 embedding으로 변환
            cemb = self.class_embedding(class_label)  # [B, cdim]
            
            # 2. timestep embedding과 class embedding 결합
            temb = temb + cemb  # [B, tdim] + [B, cdim], cdim=tdim이므로 가능
            
            # CFG의 핵심: 
            # - timestep 정보와 class 정보를 함께 사용
            # - ResBlock에서 이 combined embedding을 조건으로 사용
            # - Training 시 null condition도 학습하여 inference 시 guidance 가능
            
            #######################

        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
