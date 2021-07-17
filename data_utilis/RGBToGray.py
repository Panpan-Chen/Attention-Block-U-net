"""
The photoacoustic image generated by the time reversal algorithm is cropped and processed to gray pics
"""

import os
from PIL import Image
import numpy as np

img_origin_dir = r'D:\CPP\ThresholdDataset1.0\TR\train\0.32'
img_gray_dir = r'D:\CPP\ThresholdDataset3.0\TR\train\0.32'

label_origin_dir = r'D:\CPP\ThresholdDataset1.0\Gray\test\OSTU'
label_gray_dir = r'D:\CPP\ThresholdDataset3.0\Gray\test\OSTU'


def Input_Image_Process(input, output):
    img_num = 1
    for parent, dirnames, filenames in os.walk(input):
        filenames.sort()
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.png':
                currentPath = os.path.join(parent, filename)
                img = Image.open(currentPath)
                box = (187, 50, 721, 585)
                image = img.crop(box)
                image = image.resize((512, 512), Image.BICUBIC)  # resize
                new_png = image.convert('L')
                matrix = np.asarray(new_png)
                new_png = Image.fromarray(matrix)
                new_png.save(output + '/' + 'img' + filename)
                print(output + '/' + 'img' + filename)
            img_num += 1


def Label_Image_Process(input, output):
    img_num = 1
    for parent, dirnames, filenames in os.walk(input):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.png':
                currentPath = os.path.join(parent, filename)
                img = Image.open(currentPath)
                new_png = img.convert('L')
                new_png = new_png.resize((512, 512), Image.BICUBIC)  # resize
                # matrix = 255 - np.asarray(new_png)
                # new_png = Image.fromarray(matrix)
                new_png.save(output + '/' + 'label' + filename)
                print(output + '/' + 'label' + filename)
            img_num += 1


Input_Image_Process(img_origin_dir, img_gray_dir)