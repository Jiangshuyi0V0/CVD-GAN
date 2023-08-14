import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
from piqa.fsim import FSIM
import dnnlib
from CVD_Lens import convert, simulate, generate
from training.loss import StyleGAN2Loss


def compute_fsimc(opts, num_gen, batch_size=64, batch_gen=None):
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
    test = G(z, c)

    # print('test.shape', test.shape)
    if opts.fix:
        test_sim = StyleGAN2Loss.run_Sim(test, fix=opts.fix, cvd_type=opts.cvd_type, degree=opts.degree,
                                         Simulator=opts.sim, device=opts.device)
    else:
        test_sim = StyleGAN2Loss.run_Sim(test, opts.device, opts.delta, opts.mul)
    test = ((test + 1) / 2).clamp(0, 1)
    test_sim = ((test_sim + 1) / 2).clamp(0, 1)
    criterion = FSIM().to(opts.device)
    # print('FSIMc:', criterion(test, test_sim))
    return float(criterion(test, test_sim))
