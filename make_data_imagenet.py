import numpy as np
import glob
import random
import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tqdm import tqdm

####################
N_images = 100000
N_split = 100
random.seed(111)
####################

os.makedirs('./data/ImageNet/color/', exist_ok=True)
os.makedirs('./data/ImageNet/mono/', exist_ok=True)

# randomly selected N_images images
image_list = []
dir_list = glob.glob('./datasets/ILSVRC2012/train/n*/', recursive=True)
for dir_path in dir_list:
        image_list.extend(glob.glob(dir_path +'/'+ '*.JPEG'))

# randomize the file names
image_list = random.sample(image_list, N_images)
image_list = np.array(image_list) 

# RGB-color
counter = 0
for i in tqdm(range(N_split)):
        data = []
        for j in range(int(N_images/N_split)):
                img = img_to_array(load_img(image_list[counter], target_size=(299,299)))
                data.append(img)
                counter += 1
        data = np.asarray(data)
        data = data.astype('float32')
        np.save('./data/ImageNet/color/ImageNet_{}'.format(i),data)

# Monochrome
counter = 0
for i in tqdm(range(N_split)):
        data = []
        for j in range(int(N_images/N_split)):
                img = img_to_array(load_img(image_list[counter], grayscale=True, target_size=(299,299)))
                data.append(img)
                counter += 1
        data = np.asarray(data)
        data = data.astype('float32')
        np.save('./data/ImageNet/mono/ImageNet_mono_{}'.format(i),data)
