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
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


def compute_ssim(opts, num_gen, batch_size=64, batch_gen=None):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

    def run_generator(z, c):
        img = G(z=z, c=c, **opts.G_kwargs)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    z = torch.randn([num_gen, G.z_dim], device=opts.device)
    c = None
    test = G(z, c, **opts.G_kwargs)

    # print('test.shape', test.shape)
    if opts.fix:
        test_sim = StyleGAN2Loss.run_Sim(test, fix=opts.fix, cvd_type=opts.cvd_type, degree=opts.degree, Simulator=opts.sim, device=opts.device)
    else:
        test_sim = StyleGAN2Loss.run_Sim(test, opts.device, opts.delta, opts.mul)

    res = ssim(((test + 1) / 2).clamp(0, 1), ((test_sim + 1) / 2).clamp(0, 1), data_range=1, size_average=False,
               nonnegative_ssim=True)  # return (N,)
    # print('res.shape',res.shape)
    # print('res:',res)
    return float(res.mean())
