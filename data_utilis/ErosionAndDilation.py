
import os
import cv2

input_dir = r'D:\CPP\DATASATS_Version4.0\Test-02\label_reversed'
output_erode = r'D:\CPP\ErodeAndDilate1.0\erode\test'
output_dilate = r'D:\CPP\ErodeAndDilate1.0\dilate\test'

def morphology(input, output_erode, output_dilate):
    for parent, dirnames, filenames in os.walk(input):
        filenames.sort()
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.png':
                currentPath = os.path.join(parent, filename)
                img = cv2.imread(currentPath)
                ret, thresholdImg = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
                cv2.imshow("thresholdImg ", thresholdImg)

                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))

                erodImg = cv2.erode(thresholdImg, kernel)

                cv2.imwrite((output_erode + '/' + filename), erodImg)
                cv2.imshow("erodImg", erodImg)

                dilateImg = cv2.dilate(thresholdImg, kernel)
                cv2.imwrite((output_dilate + '/' + filename), dilateImg)
                cv2.imshow("dilateImg", dilateImg)


morphology(input_dir, output_erode,output_dilate)

