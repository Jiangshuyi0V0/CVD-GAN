﻿# CVD-GAN Implementation

## Paper
#### Personalized Image Generation for Color Vision Deficiency Population
Shuyi Jiang, Daochang Liu, Dingquan Li, Chang Xu

[Click here to access the original paper.](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Personalized_Image_Generation_for_Color_Vision_Deficiency_Population_ICCV_2023_paper.pdf)

## Abstract
*Approximately, 350 million people, a proportion of 8%, suffer from color vision deficiency (CVD). While image
generation algorithms have been highly successful in synthesizing high-quality images, CVD populations are unintentionally 
excluded from target users and have difficulties understanding the generated images as normal viewers do. Although a 
straightforward baseline can be formed by combining generation models and recolor compensation methods as the post-processing, 
the CVD friendliness of the result images is still limited since the input image content of recolor methods is not 
CVD-oriented and will be fixed during the recolor compensation process. Besides, the CVD populations can not be fully 
served since the varying degrees of CVD are often neglected in recoloring methods. Instead, we propose a personalized 
CVD friendly image generation algorithm with two key characteristics: (i) generating CVD-oriented images aligned with
the needs of CVD populations; (ii) generating continuous personalized images for people with various CVD degrees through 
disentangling the color representation based on a triple-latent structure. Quantitative and qualitative experiments 
indicate our proposed image generation model can generate practical and compelling results compared to the normal generation 
model and combination baselines on several datasets.*


## Runtime Requirements
The runtime requirements are same as stylegan2-ada-pytorch, please refer to [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch#requirements).

Additional package requirements:
* pytorch_msssim
* kornia
* piqa
* ninja

To configure the conda environment (recommended command - uses CUDA 11.1):
```.bash
conda create -n cvdgan python=3.8
conda activate cvdgan
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install click requests psutil scipy matplotlib ninja pytorch_msssim piqa kornia==0.6.5
```

## Code Structure
The code structure builds up on [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) framework.

Some important files:
* [train.py](./train.py): The main training script.
* [training/training_loop.py](./training/training_loop.py): Defines the entire training loop, including initialization, 
loss accumulation and optimization, image snapshot saving, and metrics evaluation.
* [training/loss.py](./training/loss.py): Defines the loss functions, including the ones described in paper.
* [metrics/metric_test.py](./metrics/metric_test.py): Script for evaluating the generated images according to the metrics.
* [latent_traversal.py](./latent_traversal.py): Script for latent traversal generation.

## Training
The training command inherits from [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch#stylegan2-ada--official-pytorch-implementation), 
with some additional CVD-GAN-specific arguments:
* latent_size: the dimension size of the latent code.
* loss_mode: legacy name, it now specifies weights for each loss component. - Format: ``all_<Color Information Loss Weight>_<Local Contrast Loss Weight>_<Color Histogram Disentanglement Loss Weight>``
* dis_dim: the disentangled dimension for the CVD degree. - Options: ```['first', 'last']```

Some other important arguments:
* --outdir: specifies the output directory where the training results are saved.
* --data: specifies the path to training dataset.
* --gpus: specifies the number of GPUs to use for training.

For more information regarding the comprehensive set of arguments, please refer to the [Click command options in train.py](./train.py).

To give an example:
```.bash
python train.py --outdir=CVD-GAN/training-runs --data=datasets/flowers102.zip --gpus=8 --cfg=auto --load=True --latent_size=16 --dis_dim='last' --loss_mode='all_5.0_50.0_10.0'
```

There is no strict stopping criterion for training, but we recommend to closely monitor the FID score during training, as well as the quality of generated image samples. Typically, training may be considered sufficient once the FID score has stabilized.

## Testing
[metric_test.py](./metrics/metric_test.py) offers a way to measure the generated images according to the following metrics:
* SSIM
* MS_SSIM
* Contrast Loss
* Color Histogram Disentanglement Loss
* VGG Perceptual Loss

[latent_traversal.py](./latent_traversal.py) can be used to visualize the changes in the generated images when traversing the latent z dimensions with incremental add-on values/degrees.

## Acknowledgement
This code was built based on the [Pytorch Implementation](https://github.com/NVlabs/stylegan2-ada-pytorch) of [StyleGAN-ADA](https://arxiv.org/abs/2006.06676).

## Citation - Bibtex:
```
@InProceedings{Jiang_2023_ICCV,
    author    = {Jiang, Shuyi and Liu, Daochang and Li, Dingquan and Xu, Chang},
    title     = {Personalized Image Generation for Color Vision Deficiency Population},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22571-22580}
}
```

