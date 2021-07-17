"""
Transform RGB images to gray image and then reverse pics
"""
import PIL.Image as Image
import numpy as np
import os

RGB_dir = r'D:\CPP\DATASATS_Version4.0\Test-02\label'
GRAY_dir = r'D:\CPP\DATASATS_Version4.0\Test-02\label_reversed'
files = os.listdir(RGB_dir)
for file in files:
    fileType = os.path.splitext(file)
    if fileType[1] == '.png':
        new_png = Image.open(RGB_dir + '/' + file).convert('L')
        # new_png = new_png.resize((512, 512), Image.BICUBIC)  # resize
        # matrix = np.asarray(new_png)
        matrix = 255 - np.asarray(new_png)
        new_png = Image.fromarray(matrix)
        new_png.save(GRAY_dir + '/' + file)
        print(GRAY_dir + '/' + file)
