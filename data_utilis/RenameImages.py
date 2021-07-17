"""
Name and sort the pictures uniformly

Example:
rename(Image_dir, Rename_dir, 'img')

Notes:
image_dir: Path to unnamed picture folder
rename_dir: Path to renamed picture folder
parent path:
dirnames: List of folders in this directory
filenames: List of files in this directory
filename[0]:  Image_DataLoader'
filename[1]:  '.py'
"""

import os
import cv2
from PIL import Image


def RenameImage(image_dir, rename_dir, datasets):
    img_num = 1
    for parent, dirnames, filenames in os.walk(image_dir):
        filenames.sort()
        print(filenames)
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.png':
                currentPath = os.path.join(parent, filename)
                img = Image.open(currentPath)
                if datasets == 'img':
                    img.save(rename_dir + '/' + 'img' + str(img_num) + '.png')
                    print(rename_dir + '/' + 'img' + str(img_num) + '.png')
                elif datasets == 'label':
                    img.save(rename_dir + '/' + 'label' + str(img_num) + '.png')
                    print(rename_dir + '/' + 'label' + str(img_num) + '.png')
                else:
                    print(' ERROR: datasets=img or label')
                img_num += 1
            elif os.path.splitext(filename)[1] == '.TIF':
                currentPath = os.path.join(parent, filename)
                img = cv2.imread(currentPath)
                if datasets == 'img':
                    cv2.imwrite(rename_dir + '/' + 'img' + str(img_num) + '.png', img,
                                [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
                    print(rename_dir + '/' + 'img' + str(img_num) + '.png')
                elif datasets == 'label':
                    cv2.imwrite(rename_dir + '/' + 'label' + str(img_num) + '.png', img,
                                [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
                    print(rename_dir + '/' + 'label' + str(img_num) + '.png')
                else:
                    print(' ERROR: datasets=img or label')
                img_num += 1
            else:
                print('ERROR')

        print('Rename Finished')


Image_dir = r'D:\CPP\ThresholdDataset1.0\Gray\train\0.25'
Rename_dir = r'D:\CPP\ThresholdDataset2.0\Gray\train\0.25'

RenameImage(Image_dir, Rename_dir, datasets='label')
