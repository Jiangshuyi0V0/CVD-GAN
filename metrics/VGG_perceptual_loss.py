import torch
import torchvision
import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib
from CVD_Lens import convert, simulate, generate
from training.loss import StyleGAN2Loss

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, opts, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).to(opts.device).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).to(opts.device).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).to(opts.device).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).to(opts.device).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).to(opts.device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).to(opts.device).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            #print('input shape:', input.shape)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

def compute_vgg(opts, num_gen, batch_size=64, batch_gen=None):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

    z = torch.randn([num_gen, G.z_dim], device=opts.device)
    c = None
    test = G(z, c, **opts.G_kwargs)

    if opts.fix:
        test_sim = StyleGAN2Loss.run_Sim(test, fix=opts.fix, cvd_type=opts.cvd_type, degree=opts.degree, Simulator=opts.sim, device=opts.device)
    else:
        test_sim = StyleGAN2Loss.run_Sim(test, opts.device, opts.delta, opts.mul)

    VGG_loss = VGGPerceptualLoss(opts)
    Loss = VGG_loss((test+1)/2, (test_sim+1)/2)

    return float(Loss.mean())
