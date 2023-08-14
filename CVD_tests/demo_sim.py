import torch
import torch.nn as nn
from CVD_Lens import simulate
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pylab
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# Generate a test image that spans the RGB range
im = np.asarray(Image.open("/Users/jiangshuyi/USYD/Research/Dataset/Art mini/Ishihara-Plate-30-38.jpeg").convert('RGB'))
#im = np.asarray(Image.open("../metrics/2.png").convert('RGB'))
im = torch.from_numpy(im)
#im = im / 127.5 - 1
im = im / 255.0

def bi_mask(im):
    im_bi = torch.zeros_like(im)
    white_mask = im > 0.5
    black_mask = np.logical_not(white_mask)
    im_bi[white_mask] = 1
    im_bi[black_mask] = 0
    return im_bi

def bi_mask_1(im):
    im_bi = torch.zeros(im.shape[0], im.shape[1])
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    white_mask = gray > 0.5
    black_mask = torch.logical_not(white_mask)
    im_bi[white_mask] = 1
    im_bi[black_mask] = 0
    return im_bi

#im = torch.from_numpy(np.random.RandomState(50).randn(256, 256, 3))
#im = torch.randn(256,256,3)
#im = im * 0.5 + 0.5
im.requires_grad = True
#Lin = nn.Linear(256, 3)

# Create a simulator using the Machado 2009 algorithm.
simulator = simulate.Simulator_Machado2009()
# Apply the simulator to the input image to get a simulation of protanomaly
protan_im = simulator.simulate_cvd(im, simulate.Deficiency.DEUTAN, severity=0.8, device='cpu')

protan_im_all = simulator.simulate_cvd(im, simulate.Deficiency.DEUTAN, severity=1, device='cpu')
#im = im.clamp(-1, 1) * 0.5 + 0.5
print('im', im)
print('protan_im', protan_im)
print('protan_im_all', protan_im_all)

im_bi = bi_mask(im)
protan_im_bi = bi_mask(protan_im)
protan_im_all_bi = bi_mask(protan_im_all)

im_bi_1 = bi_mask_1(im)
protan_im_bi_1 = bi_mask_1(protan_im)
protan_im_all_bi_1 = bi_mask_1(protan_im_all)

#ssim_val = ssim(im, protan_im, data_range=1, size_average=False, nonnegative_ssim=True)  # return (N,)

plt.figure()
plt.subplot(337)
plt.imshow(im_bi_1, cmap='gray')
plt.subplot(338)
plt.imshow(protan_im_bi_1, cmap='gray')
plt.subplot(339)
plt.imshow(protan_im_all_bi_1, cmap='gray')

plt.subplot(331)
plt.imshow(im.detach().numpy())
plt.subplot(332)
plt.imshow(protan_im.detach().numpy())
plt.subplot(333)
plt.imshow(protan_im_all.detach().numpy())

plt.subplot(334)
plt.imshow(im_bi)
plt.subplot(335)
plt.imshow(protan_im_bi)
plt.subplot(336)
plt.imshow(protan_im_all_bi)

pylab.show()
print('*'*100)