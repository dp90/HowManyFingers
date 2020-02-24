# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:26:07 2020

@author: David
"""

import os
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def create_labels_file(srcDir):
    with open(srcDir + "labels.txt", "w") as file:
        for fileName in sorted(os.listdir(srcDir)):
            if fileName == "Thumbs.db":
                continue
            file.write(fileName+":-\n")
        file.close()
    return

def labels_into_dict(srcDir):
    d = {}
    with open(srcDir + "labels.txt", "r") as labelFile:
        for line in labelFile:
            key, value = line.strip().split(':-')
            d[key] = value
        labelFile.close()
    return d

def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = dict(image._getexif().items())
    
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass
    
    return image

def prepare_shape(image,imSize):
    image = correct_orientation(image)
    iW, iH = image.size
    if iW > iH:
        image = image.crop((iW/8,0,7*iW/8,iH))
    elif iH > iW:
        image = image.crop((0,iH/8,iW,7*iH/8))
    image = image.resize(imSize)
    
    return image

def prepare_image_dataset(srcDir,finDir=None,imSize=(64,64),save=False,flips=False):
    nImgs = 0
    print("Check ", finDir, " to see progress.")
    labelDict = labels_into_dict(srcDir) # Dict of img filenames to labels
    
    with open(finDir + "labels.txt", "w") as file:
        for fileName in sorted(os.listdir(srcDir)):
            if (fileName == "Thumbs.db" or fileName == "labels.txt"):
                continue
            img = Image.open(srcDir + fileName)
            img = prepare_shape(img,imSize)
            img.save(finDir + "img" + str(nImgs) + ".jpg")
            file.write("img" + str(nImgs) + ".jpg:-" + labelDict[fileName] + "\n")
            nImgs += 1
            if not flips:
                continue
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(finDir + "img" + str(nImgs) + ".jpg")
            file.write("img" + str(nImgs) + ".jpg:-" + labelDict[fileName] + "\n")
            nImgs += 1
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img.save(finDir + "img" + str(nImgs) + ".jpg")
            file.write("img" + str(nImgs) + ".jpg:-" + labelDict[fileName] + "\n")
            nImgs += 1
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(finDir + "img" + str(nImgs) + ".jpg")
            file.write("img" + str(nImgs) + ".jpg:-" + labelDict[fileName] + "\n")
            nImgs += 1
            img.close()
        file.close()
        
    print("Images prepared.")
    return 

def add_images_to_dataset(srcDir,finDir=None,imSize=(64,64),save=False,flips=False):
    nImgs = len(os.listdir(finDir)) if len(os.listdir(finDir)) == 0 else len(os.listdir(finDir)) - 1 # consider labels.txt
    
    with open(finDir + "labels.txt", "a") as file:
        for fileName in sorted(os.listdir(srcDir)):
            imgPlot = mpimg.imread(srcDir + fileName)
            plt.imshow(imgPlot)
            plt.show()
            label = input("Label: ")
            
            img = Image.open(srcDir + fileName)
            img = prepare_shape(img,imSize)
            img.save(finDir + "img" + str(nImgs) + ".jpg")
            file.write("img" + str(nImgs) + ".jpg:-" + label + "\n")
            nImgs += 1
            if not flips:
                continue
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(finDir + "img" + str(nImgs) + ".jpg")
            file.write("img" + str(nImgs) + ".jpg:-" + label + "\n")
            nImgs += 1
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img.save(finDir + "img" + str(nImgs) + ".jpg")
            file.write("img" + str(nImgs) + ".jpg:-" + label + "\n")
            nImgs += 1
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(finDir + "img" + str(nImgs) + ".jpg")
            file.write("img" + str(nImgs) + ".jpg:-" + label + "\n")
            nImgs += 1
            img.close()
        file.close()
    print("New images added to dataset. Please remove from folder.")
    return