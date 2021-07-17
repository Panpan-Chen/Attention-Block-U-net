import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from utilis.DataHelper import *

# Chose models
from networks.U_Net import UNet  # U-Net
from networks.Attention_UNet import Att_UNet  # Attention U-Net
from networks.AB_UNet import AB_UNet  # AB-UNet
from utilis.utilis import ImageEval_

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show(img):
    """ Prediction visualization """

    plt.figure(figsize=(12, 8))
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


def test():
    # Normalization
    x_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    y_transform = transforms.ToTensor()

    batch_size = 1

    PATH = r'E:\CPP\AutoEncoder\model\threshold\UNetRev-0.20-4.pth'

    # Load Test dataset
    test_root = r'D:\CPP\ThresholdDataset3.0\TR\test\0.20'
    label_root = r'D:\CPP\ThresholdDataset3.0\Gray\test\0.20'
    test_set = EvalDataset(img_root=test_root,
                           label_root=label_root,
                           transform=x_transform,
                           target_transform=y_transform)
    test_dataloader = DataLoader(test_set,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4)

    # Load mdoel
    model = AB_UNet(1, 1).to(device)

    # Load dict
    model.load_state_dict(torch.load(PATH, map_location=lambda storage, loc: storage))

    model.eval()

    psnr_ = []
    ssim_ = []
    with torch.no_grad():
        for index, batches in enumerate(test_dataloader):
            images, labels = batches

            # CUDA
            images = images.to(device)
            labels = labels.to(device)

            predicts = model(images)

            # Visualize
            save_image(predicts, r"D:\Threshold_Test\recon\0.20\{}.png".format(index + 1), normalize=True,
                       range=(-1, 1), scale_each=False, pad_value=0)
            show(make_grid(predicts, normalize=True, range=(-1, 1), scale_each=False, pad_value=0))

            # Calculate SSIM & PSNR
            ImageEval1 = ImageEval_(predicts.clamp(0.1, 1.0), labels, channel=1)
            ImageEval2 = ImageEval_(images.clamp(0.1, 1.0), labels, channel=1)
            psnr_score1 = ImageEval1.cal_psnr()  # predict
            psnr_score2 = ImageEval2.cal_psnr()  # img
            psnr_.append((psnr_score1, psnr_score2))

            ssim_score1 = ImageEval1.cal_ssim()
            ssim_score2 = ImageEval2.cal_ssim()
            ssim_.append((ssim_score1, ssim_score2))

    data_psnr = pd.DataFrame(psnr_)
    data_psnr.to_csv('csvs/psnr-thresh.csv')

    data_ssim = pd.DataFrame(ssim_)
    data_ssim.to_csv('csvs/ssim-thresh.csv')


if __name__ == '__main__':
    print("Start Test >>>>>>>>>>>>>>>>")
    test()
    print('######### Finished Test #########')
