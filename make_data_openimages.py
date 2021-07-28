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

os.makedirs('./data/openimages/color/', exist_ok=True)
os.makedirs('./data/openimages/mono/', exist_ok=True)

# randomly selected N_images images
image_list = glob.glob('./datasets/open-images-dataset/test/' + '*.jpg')

# randomize the file names
image_list = random.sample(image_list, N_images)
image_list = np.array(image_list) 

# RGB-color
counter = 0
for i in tqdm(range(N_split)):
        data = []
        for j in range(int(N_images/N_split)):
                img = img_to_array(load_img(image_list[counter], target_size=(299,299)))
                img = (img/img.max())*255.0
                data.append(img)
                counter += 1
        data = np.asarray(data)
        data = data.astype('float32')
        np.save("./data/openimages/color/open-images-dataset_{}".format(i),data)

# Monochrome
counter = 0
for i in tqdm(range(N_split)):
        data = []
        for j in range(int(N_images/N_split)):
                img = img_to_array(load_img(image_list[counter], grayscale=True, target_size=(299,299)))
                img = (img/img.max())*255.0
                data.append(img)
                counter += 1
        data = np.asarray(data)
        data = data.astype('float32')
        np.save("./data/openimages/mono/open-images-dataset_{}".format(i),data)
