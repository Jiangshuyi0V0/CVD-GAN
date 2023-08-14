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


def compute_cpr(opts, num_gen, batch_size=64, batch_gen=None):
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
    res = StyleGAN2Loss.contrastLoss(test, test_sim).to(opts.device)

    return float(res.mean())
