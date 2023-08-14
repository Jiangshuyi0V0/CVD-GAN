# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from CVD_Lens import convert, simulate
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from piqa.fsim import FSIM
import warnings
import torch.nn.functional as F
from kornia.color import rgb_to_lab
from training.loss_utils import _colorLoss, _contrast, _fspecial_gauss_1d, RGBuvHistBlock


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_z1, gen_cvd, delta, epsilon, mul, gen_c, sync,
                             gain, **kwargs):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    Simulator = simulate.Simulator_Machado2009()

    def __init__(self, device, G_mapping, G_synthesis, Q, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10,
                 pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.Q = Q
        # self.Sim = simulate.Simulator_Machado2009()
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                         torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def run_Q(self, img, img1, sync):
        with misc.ddp_sync(self.Q, sync):
            img_in = torch.cat([img, img1], 1)
            regress_out = self.Q(img_in)
        return regress_out

    @staticmethod
    def run_Sim(img, device, delta=0.0, latent_mul=1.0, fix=False, cvd_type=None, degree=None, Simulator=Simulator):
        """
        Input: img.shape - ([batchsize, channel, height, width])
               img - range[0.0, 1.0] for SSIM_Loss calculation and simulator translation
               delta - the delta of dim0 in the latent code - range [-1, 1] * latent_mul
               latent_mul - multiple of delta processed in the latent
               severity - range[0.0,1.0]
        Output: content_loss.shape - ([batchsize])
        """
        """
        For CVD simulator:
        input: img.shape(h,w,3) in the range of [0.0, 1.0]
        severity: float - - range[0.0,1.0]
        """
        img = (img + 1) / 2.0  # map img[-1, 1] into the range of [0.0, 1.0]
        img_Sim = torch.zeros_like(img)
        idx = 0
        # print('img:', img.shape)
        # print('severity:', severity.shape, '\n', severity)
        if fix:
            for I in img:
                I = I.permute(1, 2, 0)
                if cvd_type == 'DEUTAN':
                    Sim = Simulator.simulate_cvd(I, simulate.Deficiency.DEUTAN, severity=degree,
                                                 device=device)  # img_Sim should in the range of [0.0, 1.0]
                else:
                    Sim = Simulator.simulate_cvd(I, simulate.Deficiency.PROTAN, severity=degree,
                                                 device=device)  # img_Sim should in the range of [0.0, 1.0]
                img_Sim[idx] = Sim.permute(2, 0, 1)
                idx += 1
            img_Sim = img_Sim * 2 - 1  # to the range of [-1.0, 1.0]
            return img_Sim
        severity = torch.abs(delta / latent_mul)
        for I, S in zip(img, severity):
            I = I.permute(1, 2, 0)
            # print('I.shape',I.shape)
            S = S.item()
            # print('severity', S)
            Sim = Simulator.simulate_cvd(I, simulate.Deficiency.PROTAN, severity=S,
                                         device=device)  # img_Sim should in the range of [0.0, 1.0]
            # img_Sim[idx] = Sim
            # print('Sim.shape', Sim.shape)
            img_Sim[idx] = Sim.permute(2, 0, 1)
            idx += 1
        # print('img_Sim.shape', img_Sim.shape)
        img_Sim = img_Sim * 2 - 1  # to the range of [-1.0, 1.0]
        return img_Sim

    @staticmethod
    def SSIM_loss(img, img1):
        img = ((img + 1) / 2.0)  # .clamp(0, 1)
        img1 = ((img1 + 1) / 2.0)  # .clamp(0, 1)
        ssim_val = ssim(img, img1, data_range=1, size_average=False, nonnegative_ssim=True)  # return (N,)
        return 1 - ssim_val

    @staticmethod
    def MS_SSIM_loss( img, img1):
        img = (img + 1) / 2.0  # .clamp(0, 1)
        img1 = (img1 + 1) / 2.0  # .clamp(0, 1)
        ms_ssim_val = ms_ssim(img, img1, data_range=1, size_average=False)  # return (N,)
        return 1 - ms_ssim_val

    def FSIMc_loss(self, img, img1):
        criterion = FSIM().to(self.device)
        return 1 - criterion(img, img1)

    @staticmethod
    def contrastLoss(X, Y, size_average=True, win_size=11, win_sigma=1.5, win=None):
        X = rgb_to_lab(((X + 1) / 2.0).clamp(0, 1))
        Y = rgb_to_lab(((Y + 1) / 2.0).clamp(0, 1))
        if not X.shape == Y.shape:
            raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

        for d in range(len(X.shape) - 1, 1, -1):
            X = X.squeeze(dim=d)
            Y = Y.squeeze(dim=d)

        if len(X.shape) not in (4, 5):
            raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

        if not X.type() == Y.type():
            raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

        if win is not None:  # set win_size
            win_size = win.shape[-1]

        if not (win_size % 2 == 1):
            raise ValueError("Window size should be odd.")

        if win is None:
            win = _fspecial_gauss_1d(win_size, win_sigma)
            win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
            # win = win.repeat([X.shape[1] - 1] + [1] * (len(X.shape) - 1))

        cpr = _contrast(X, Y, win=win)

        if size_average:
            return cpr.mean()
        else:
            return cpr.mean(1)

    @staticmethod
    def colorInfoLoss(X, Y, size_average=True, win_size=11, win_sigma=5, win=None):
        # X = rgb_to_lab(((X + 1) / 2.0).clamp(0, 1))
        # Y = rgb_to_lab(((Y + 1) / 2.0).clamp(0, 1))
        if not X.shape == Y.shape:
            raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

        if not X.shape[1] == Y.shape[1]:
            raise ValueError(f"Input images should be color images.")

        for d in range(len(X.shape) - 1, 1, -1):
            X = X.squeeze(dim=d)
            Y = Y.squeeze(dim=d)

        if len(X.shape) not in (4, 5):
            raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

        if not X.type() == Y.type():
            raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

        if win is not None:  # set win_size
            win_size = win.shape[-1]

        if not (win_size % 2 == 1):
            raise ValueError("Window size should be odd.")

        if win is None:
            win = _fspecial_gauss_1d(win_size, win_sigma)
            win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
        loss = _colorLoss(X, Y, win=win)
        return loss

    def Content_loss(self, img1, img2):
        """
        Input: img.shape - ([batchsize, channel, height, width])
        Output: content_loss.shape - ([batchsize])
        """
        loss = torch.zeros(img1.shape[0])
        idx = 0
        # print('loss.shape', loss.shape)
        for I1, I2 in zip(img1, img2):
            I1 = convert.bin_mask(I1)
            I2 = convert.bin_mask(I2)
            # print('I1.shape', I1.shape)
            loss[idx] = torch.abs(I1 - I2).mean()
            idx += 1
        return loss.to(self.device)

    def VP_loss(self, delta, regress_out, C_lambda=1):
        prob_C = torch.nn.functional.softmax(regress_out, 1)
        loss = delta * torch.log(prob_C + 1e-12)
        loss = C_lambda * torch.sum(loss, dim=1)
        loss = -loss
        return loss

    @staticmethod
    def histloss(device, img1, img2):
        img1 = (img1 + 1) / 2.
        img2 = (img2 + 1) / 2.
        # create a histogram block
        histogram_block = RGBuvHistBlock(device=device)

        input_hist = histogram_block(img1)
        target_hist = histogram_block(img2)

        histogram_loss = (1 / np.sqrt(2.0) * (torch.sqrt(torch.sum(
            torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)))) /
                          input_hist.shape[0])
        return histogram_loss

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_z1, gen_cvd, delta, epsilon, mul, gen_c, sync,
                             gain,
                             **kwargs):
        # for cifar:
        # real_img: torch.Size([32, 3, 32, 32])
        # gen_z: torch.Size([32, 512])
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']  # when cfg=cifar10, Gmain,Greg,Dmain,Dreg
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # loss paras:
        loss_mode = kwargs['loss_mode']
        loss_para = kwargs['loss_para']
        is_single = kwargs['single']
        if is_single:
            degree = kwargs['degree']
            cvd_type = kwargs['cvd_type']
        else:
            degree = None
        dim_idx = kwargs['dim']

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl))  # May get synced by Gpl.
                if is_single:  # 为了方便，暂时保留变量名，img_Sim1用的原始img生成
                    img_Sim1 = self.run_Sim(gen_img, self.device, fix=is_single, degree=degree,
                                            cvd_type=cvd_type)  # in the range of [0.0, 1.0]

                else:
                    gen_img1, _gen_ws1 = self.run_G(gen_z1, gen_c, sync=(sync and not do_Gpl))  # May get synced by Gpl.
                    gen_cvd, _gen_ws2 = self.run_G(gen_cvd, gen_c, sync=(sync and not do_Gpl))  # May get synced by Gpl.
                    img_Sim1 = self.run_Sim(gen_cvd, self.device, epsilon, mul)  # in the range of [-1.0, 1.0]
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)

                assert loss_mode in ['contentLoss', 'ssimLoss', 'both', 'msssimLoss', 'fsimLoss', 'no',
                                     'contrastLoss', 'colorLoss', '', 'hisLoss', 'all']

                if loss_mode == 'contentLoss':
                    content_loss = self.Content_loss(gen_img, img_Sim1)
                    training_stats.report('Loss/G/content_loss', content_loss)
                    loss_Gmain = loss_Gmain + loss_para * content_loss
                    training_stats.report('Loss/G/Gmain_with_content', loss_Gmain)
                elif loss_mode == 'ssimLoss':
                    ssim_loss = self.SSIM_loss(gen_img, img_Sim1)
                    training_stats.report('Loss/G/ssim_loss', ssim_loss)
                    loss_Gmain = loss_Gmain + loss_para * ssim_loss
                    training_stats.report('Loss/G/Gmain_with_ssim', loss_Gmain)
                elif loss_mode == 'both':
                    CL_para = loss_para['CL_para']
                    SL_para = loss_para['SL_para']
                    color_loss = self.histloss(self.device, gen_img, img_Sim1)
                    training_stats.report('Loss/G/color_loss', color_loss)
                    training_stats.report('Loss/G/Gmain_with_colorInfo', loss_Gmain + CL_para * color_loss)
                    ms_ssim_loss = self.MS_SSIM_loss(gen_img, img_Sim1)
                    training_stats.report('Loss/G/MSssim_loss', ms_ssim_loss)
                    training_stats.report('Loss/G/Gmain_with_msssim', loss_Gmain + SL_para * ms_ssim_loss)
                    loss_Gmain = loss_Gmain + CL_para * color_loss + SL_para * ms_ssim_loss
                    training_stats.report('Loss/G/Gmain_with_both', loss_Gmain)
                elif loss_mode == 'all':
                    CL_para = loss_para['CL_para']
                    SL_para = loss_para['SL_para']
                    hL_para = loss_para['hL_para']
                    color_loss = self.colorInfoLoss(gen_cvd, img_Sim1)
                    training_stats.report('Loss/G/color_loss', color_loss)
                    ms_ssim_loss = self.MS_SSIM_loss(gen_cvd, img_Sim1)
                    training_stats.report('Loss/G/MSssim_loss', ms_ssim_loss)
                    hist_loss = self.histloss(self.device, gen_img, gen_img1)
                    training_stats.report('Loss/G/hist_loss', hist_loss)
                    loss_Gmain = loss_Gmain + CL_para * color_loss + SL_para * ms_ssim_loss + hL_para * hist_loss
                    training_stats.report('Loss/G/Gmain_with_both', loss_Gmain)
                elif loss_mode == 'msssimLoss':
                    ms_ssim_loss = self.MS_SSIM_loss(gen_img, img_Sim1)
                    training_stats.report('Loss/G/MSssim_loss', ms_ssim_loss)
                    loss_Gmain = loss_Gmain + loss_para * ms_ssim_loss
                    training_stats.report('Loss/G/Gmain_with_ms-ssim', loss_Gmain)
                elif loss_mode == 'fsimLoss':
                    fsim_loss = self.FSIMc_loss(gen_img, img_Sim1)
                    training_stats.report('Loss/G/fsim_loss', fsim_loss)
                    loss_Gmain = loss_Gmain + loss_para * fsim_loss
                    training_stats.report('Loss/G/Gmain_with_fsim', loss_Gmain)
                elif loss_mode == 'contrastLoss':
                    contrast_loss = self.contrastLoss(gen_img, img_Sim1)
                    training_stats.report('Loss/G/contrast_loss', contrast_loss)
                    loss_Gmain = loss_Gmain + loss_para * contrast_loss
                    training_stats.report('Loss/G/Gmain_with_contrast', loss_Gmain)
                elif loss_mode == 'colorLoss':
                    color_loss = self.colorInfoLoss(gen_img, img_Sim1)
                    training_stats.report('Loss/G/color_loss', color_loss)
                    loss_Gmain = loss_Gmain + loss_para * color_loss
                    training_stats.report('Loss/G/Gmain_with_colorInfo', loss_Gmain)

                if loss_mode == 'hisLoss':
                    hist_loss = self.histloss(self.device, gen_img, gen_img1)
                    training_stats.report('Loss/G/hist_loss', hist_loss)
                    loss_Gmain = loss_Gmain + loss_para * hist_loss
                    training_stats.report('Loss/G/Gmain_with_hist', loss_Gmain)
                # regress_out = self.run_Q(img=gen_img, img1=gen_img1, sync=sync)
                # vp_loss = self.VP_loss(delta, regress_out)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = \
                        torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True,
                                            only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                if loss_mode == 'no':
                    img_Sim1 = self.run_Sim(gen_img, self.device, epsilon, mul,
                                            fix=is_single)  # in the range of [0.0, 1.0]
                    img_Sim1 = img_Sim1 * 2 - 1  # to the range of [-1.0, 1.0]
                    gen_logits = self.run_D(img_Sim1, gen_c, sync=False)  # Gets synced by loss_Dreal.
                else:
                    gen_logits = self.run_D(gen_img, gen_c, sync=False)  # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = \
                            torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
                                                only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

# ----------------------------------------------------------------------------
