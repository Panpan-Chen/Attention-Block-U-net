
import torch
import numpy as np
import cv2
from skimage.measure import compare_ssim, compare_psnr
import matplotlib.pyplot as plt


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def torch2numpy(img, channel):
    """ Transform from torch to numpy  """
    img_numpy = torch.squeeze(img).cpu().numpy()
    if channel == 3:
        img_numpy = np.transpose(img_numpy, (1, 2, 0))
    return img_numpy


class ImageEval_(object):
    """
    cal_ssim: calculate ssim value of model predicted images
    cal_psnr: calculate psnr value of model predicted images
    """

    def __init__(self, predicts, labels, channel=3):
        self.predicts = torch2numpy(predicts, channel=1)
        self.labels = torch2numpy(labels, channel=1)
        self.channel = channel

    def cal_ssim(self):
        if self.channel == 3:
            grayA = cv2.cvtColor(self.predicts, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(self.labels, cv2.COLOR_BGR2GRAY)
        elif self.channel == 1:
            grayA = self.predicts
            grayB = self.labels
        else:
            print('Channel is 1 for Gray Image or 3 for RGB Image')
        (score_ssim, diff) = compare_ssim(grayA, grayB, full=True)

        return score_ssim

    def cal_psnr(self):
        score_psnr = compare_psnr(self.predicts, self.labels, 1)
        return score_psnr


def adjust_learning_rate(starting_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = starting_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
