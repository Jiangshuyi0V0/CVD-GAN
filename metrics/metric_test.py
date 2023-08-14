import numpy as np
from CVD_Lens import convert, simulate, generate
import PIL
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms
from pytorch_msssim import ms_ssim, SSIM, MS_SSIM
import torch
from training.loss import StyleGAN2Loss
from skimage.color import rgb2lab, lab2rgb
from kornia.color import rgb_to_lab
from cpr_utils import ssim
import pytorch_msssim
from color_utils import colorInfoLoss
import os

transform1 = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize([224,224])
    # print('*'*100)
]
)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            # print('input shape:', input.shape)
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


def get_tensor(img):
    img = transform1(img)
    res = torch.zeros([480, 3, 256, 256])
    idx = 0
    for h_idx in range(16):
        for w_idx in range(30):
            res[idx] = img[:, h_idx * 256:(h_idx + 1) * 256, w_idx * 256:(w_idx + 1) * 256]
            # img_save = img[:,h_idx*256:(h_idx+1)*256,w_idx*256:(w_idx+1)*256]
            # img_save = (np.asarray(res[idx], dtype=np.float32)) * 255
            # img_save = np.rint(img_save).astype(np.uint8)
            # print('res[idx].shape',res[idx].shape)
            # img_save = img_save.transpose(1, 2, 0)
            idx += 1
            # PIL.Image.fromarray(img_save, 'RGB').save(f'./data/base_{idx}.png')
    print('res.shape', res.shape)
    return res


def norm_lab(img):
    assert (img.shape[1] == 3)
    img[:, 0, :, :] /= 100.
    img[:, 1, :, :] = (img[:, 1, :, :] + 127) / 255
    img[:, 2, :, :] = (img[:, 2, :, :] + 127) / 255
    return img


def compute_VGG(img1, img2, num_gen=480, batch_size=48):
    VGG_loss = VGGPerceptualLoss()
    idx = 0
    Loss = torch.zeros_like(torch.zeros(int(num_gen / batch_size), batch_size))

    while idx < int(num_gen / batch_size):
        Loss[idx] = VGG_loss(img1, img2)
        idx += 1
    return Loss


def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-4:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)
    return Filelist


def all_tensor(imglist):
    res = torch.zeros([480, 3, 256, 256])
    for idx, imgpath in enumerate(imglist):
        # print('image:', imgpath)
        im = np.asarray(PIL.Image.open(imgpath).convert('RGB'))
        im = transform1(im).unsqueeze(0)
        res[idx] = im
    # print(res)
    return res


def run_folder(folder):
    org_img_folder = '/Volumes/SUYEE/result_image/' + folder

    # 检索文件
    imglist = getFileList(org_img_folder, [], 'png')
    res = all_tensor(imglist)
    res_sim = StyleGAN2Loss.run_Sim(res * 2 - 1, device='cpu', fix=True, cvd_type='DEUTAN', degree=1.0)
    ssim_val = pytorch_msssim.ssim(res, (res_sim + 1) / 2,
                                   data_range=1)  # , size_average=False, nonnegative_ssim=True)
    print(f'SSIM of {folder}.png', ssim_val)
    ms_ssim_val = ms_ssim(res, (res_sim + 1) / 2, data_range=1)  # , size_average=False, nonnegative_ssim=True)
    print(f'MS_SSIM of {folder}.png', ms_ssim_val)
    res = res * 2 - 1
    cpr = StyleGAN2Loss.contrastLoss(res, res_sim)
    print(f'cpr of {folder}.png', cpr)
    hist = StyleGAN2Loss.histloss('cpu', res, res_sim)
    print(f'hist of {folder}.png', hist)
    VGG_loss = VGGPerceptualLoss()
    Loss = VGG_loss((res + 1) / 2, (res_sim + 1) / 2)
    print(f'VGG of {folder}.png', Loss)
    print('*' * 50)


def run_res(imgList):
    for img in imgList:
        im = np.asarray(
            PIL.Image.open(f'/Users/jiangshuyi/USYD/Research/hfai/testdata/{img}.png').convert('RGB'))
        res = get_tensor(im)
        res_sim = StyleGAN2Loss.run_Sim(res * 2 - 1, device='cpu', fix=True, cvd_type='PROTAN', degree=1.0)
        ssim_val = pytorch_msssim.ssim(res, (res_sim + 1) / 2,
                                       data_range=1)  # , size_average=False, nonnegative_ssim=True)
        print(f'SSIM of {img}.png', ssim_val)
        ms_ssim_val = ms_ssim(res, (res_sim + 1) / 2, data_range=1)  # , size_average=False, nonnegative_ssim=True)
        print(f'MS_SSIM of {img}.png', ms_ssim_val)
        res = res * 2 - 1
        cpr = StyleGAN2Loss.contrastLoss(res, res_sim)
        print(f'cpr of {img}.png', cpr)
        hist = StyleGAN2Loss.histloss('cpu', res, res_sim)
        print(f'hist of {img}.png', hist)
        VGG_loss = VGGPerceptualLoss()
        Loss = VGG_loss((res + 1) / 2, (res_sim + 1) / 2)
        print(f'VGG of {img}.png', Loss)
        print('*' * 50)


def run_single(img):
    im = np.asarray(PIL.Image.open(f'{img}.png').convert('RGB'))
    res = transform1(im).unsqueeze(0)
    res_sim = StyleGAN2Loss.run_Sim(res * 2 - 1, device='cpu', fix=True, cvd_type='PROTAN', degree=1.0)
    ssim_val = pytorch_msssim.ssim(res, (res_sim + 1) / 2,
                                   data_range=1)  # , size_average=False, nonnegative_ssim=True)
    print(f'SSIM of {img}.png', ssim_val)
    ms_ssim_val = ms_ssim(res, (res_sim + 1) / 2, data_range=1)  # , size_average=False, nonnegative_ssim=True)
    print(f'MS_SSIM of {img}.png', ms_ssim_val)
    res = res * 2 - 1
    cpr = StyleGAN2Loss.contrastLoss(res, res_sim)
    print(f'cpr of {img}.png', cpr)
    hist = StyleGAN2Loss.histloss('cpu', res, res_sim)
    print(f'hist of {img}.png', hist)
    VGG_loss = VGGPerceptualLoss()
    Loss = VGG_loss((res + 1) / 2, (res_sim + 1) / 2)
    print(f'VGG of {img}.png', Loss)
    print('*' * 50)


# run_single('2')
# run_single('3')
# run_res(['abstract_20', 'abstract_80'])
run_folder('deutan/cvd_lvl_100')
