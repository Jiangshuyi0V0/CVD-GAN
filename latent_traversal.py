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
@click.option('--seeds', type=int, help='random seeds')
# @click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
# @click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
# @click.option('--n_imgs', type=int, help='Number of image pairs to generate', required=True)
@click.option('--n_continuous', type=int, help='Number of continuous latents', required=True)
# @click.option('--batch_size', type=int, help='Batch size for generation', default=10)
@click.option('--n_dim_loop', type=int, help='traversal times of each dimensions', default=20)
def main(ctx: click.Context,
         network_pkl: str,
         seeds: int,
         #    truncation_psi: float,
         noise_mode: str,
         outdir: str,
         class_idx: Optional[int],
         #    projected_w: Optional[str]
         # n_imgs: int,
         n_continuous: int,
         # batch_size: int
         n_dim_loop: int
         ):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with open(network_pkl, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)  # torch.nn.Module
        assert G.z_dim == n_continuous
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    # labels = None
    for dim_idx in range(G.z_dim):
        print('latent traversal of %d dimension' % (dim_idx+1))
        # Labels.
        grid_labels = torch.zeros([1, G.c_dim], device=device)
        if G.c_dim != 0:
            if class_idx is None:
                ctx.fail('Must specify class label with --class when using a conditional network')
            grid_labels[:, class_idx] = 1
        else:
            if class_idx is not None:
                print('warn: --class=lbl ignored when running on an unconditional network')

        #z = torch.randn([1, n_continuous]).to(device)
        z = torch.from_numpy(np.random.RandomState(seeds).randn(1, G.z_dim)).to(device)
        fake = G(z, grid_labels, noise_mode=noise_mode)
        delta_z = 0.4 * torch.FloatTensor(n_dim_loop, 1).uniform_(0, 1).to(device)
        delta_z, _indice = delta_z.sort(dim=0)
        if dim_idx == G.z_dim-1:
          print('delta_z is', delta_z)
          severity = ((delta_z / 0.4) + 1) / 2.0
          print('degree is ', severity)
        for i in range(n_dim_loop):
            z[0][dim_idx] = z[0][dim_idx] + float(delta_z[i,:])
            fake1 = G(z, grid_labels, noise_mode=noise_mode)
            #print('fake1__*.shape', fake1.shape)
            fake = torch.cat([fake, fake1], 3)
            #print('fake.shape', fake.shape)
        if dim_idx == 0:
            img_sum = fake
        img = (fake.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        #print('fake.shape', fake.shape)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(
            os.path.join(outdir, 'traversal_%06d.jpg' % (dim_idx+1)))
        if dim_idx > 0:
            img_sum = torch.cat([img_sum, fake], 2)
            #print('sum.shape', img_sum.shape)
    img_sum = (img_sum.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img_sum[0].cpu().numpy(), 'RGB').save(
            os.path.join(outdir, 'summary.jpg'))

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
