import torch
import os
import re
from typing import List, Optional

import dnnlib
import click
import pickle
import numpy as np
import PIL.Image


# ----------------------------------------------------------------------------
@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# @click.option('--seeds', type=num_range, help='List of random seeds')
# @click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
# @click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--n_imgs', type=int, help='Number of image pairs to generate', required=True)
@click.option('--n_continuous', type=int, help='Number of continuous latents', required=True)
@click.option('--batch_size', type=int, help='Batch size for generation', default=10)
def main(ctx: click.Context,
         network_pkl: str,
         #    seeds: Optional[List[int]],
         #    truncation_psi: float,
         noise_mode: str,
         outdir: str,
         class_idx: Optional[int],
         #    projected_w: Optional[str]
         n_imgs: int,
         n_continuous: int,
         batch_size: int
         ):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with open(network_pkl, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)  # torch.nn.Module
        assert G.z_dim == n_continuous
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    n_batches = n_imgs // batch_size
    labels = None
    for i in range(n_batches):
        print('Generating image pairs %d/%d ...' % (i, n_batches))
        # Labels.
        grid_labels = torch.zeros([batch_size, G.c_dim], device=device)
        if G.c_dim != 0:
            if class_idx is None:
                ctx.fail('Must specify class label with --class when using a conditional network')
            grid_labels[:, class_idx] = 1
        else:
            if class_idx is not None:
                print('warn: --class=lbl ignored when running on an unconditional network')

        z1 = torch.FloatTensor(batch_size, n_continuous).uniform_(-2, 2).to(device)
        z2 = torch.FloatTensor(batch_size, n_continuous).uniform_(-2, 2).to(device)
        delta_dim = torch.randint(0, G.z_dim, size=(batch_size,))
        delta_onehot = torch.zeros((batch_size, n_continuous)).to(device)
        for row, idx in enumerate(delta_dim):
            delta_onehot[row][idx] = 1
        z2 = torch.where(delta_onehot > 0, z2, z1)
        delta_z = z1 - z2

        if i == 0:
            labels = delta_z
        else:
            labels = torch.cat([labels, delta_z], 0)

        fakes1 = G(z1, grid_labels, noise_mode=noise_mode)
        fakes2 = G(z2, grid_labels, noise_mode=noise_mode)
        # print('fakes1.shape', fakes1.shape)
        # print('fakes2.shape', fakes2.shape)

        for j in range(fakes1.shape[0]):
            pair_img = torch.cat([fakes1[j], fakes2[j]], 2)
            # print('pair_img.shape', pair_img.shape)
            img = (pair_img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # print('img.shape', img.shape)
            PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(
                os.path.join(outdir, 'pair_%06d.jpg' % (i * batch_size + j)))
    np.save(os.path.join(outdir, 'labels.npy'), labels.cpu())


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
