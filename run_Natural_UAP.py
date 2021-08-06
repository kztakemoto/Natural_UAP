#################################################################################
# argument parser
#################################################################################
#    --X_train_path: str, path to training images (in npy format) '*/*.npy'
#    --y_train_path: str, path to the labels of training images '*/*.npy'
#    --X_test_path: str, path to test images '*/*.npy'
#    --y_test_path: str, Upath to the labels of test images '*/*.npy'
#    --X_materials_dir: str, path to the directory storing natural images
#    --model_path: str, path to model weight '*/*.h5'
#    --model_type: 'InceptionV3', 'VGG16', 'ResNet50'
#    --norm_type: str, '2' or 'inf', norm type of UAPs
#    --norm_rate: float, noise strength (zeta)
#    --fgsm_eps: float, attack step size of FGSM
#    --uap_iter: int, maximum number of iterations for computing UAP.
#    --targeted: int, target class (negative value indicates non-targeted attacks)
#    --save_path: str, path to output files 
#################################################################################

import warnings
warnings.filterwarnings('ignore')
import os, sys, glob, argparse
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
import numpy as np

import keras
import tensorflow as tf
from keras import backend as K
from keras import utils

# for preventing tensorflow from allocating the totality of a GPU memory.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import Lambda, Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD

from art.classifiers import KerasClassifier
from art.attacks import UniversalPerturbation
from art.utils import random_sphere
from art.utils import projection

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# check the starting time for computing total processing time
import time
start_time = time.time()

### UAP class ###
# classifier: classifier
# X_train: ndarray, training images
# y_train: ndarray, the labels of the training images
# X_test: ndarray, test images
# y_test: ndarray, the labels of the test images
# X_materials_paths: array, path to the directory storing natural images
# norm_type: 2 or np.inf, norm type of UAPs
# norm_size: float, noise size (xi)
# fgsm_eps: float, Fattack step size of FGSM
# uap_iter: int, maximum number of iterations for computing UAP.
# targeted: int, target class (negative value indicates non-targeted attacks)
# save_path: str, path to output files 
class my_UAP:
    def __init__(
                self, 
                classifier, 
                X_train, y_train,
                X_test, y_test,
                X_materials_paths,
                norm_type, 
                norm_size, 
                fgsm_eps,
                uap_iter, 
                targeted,
                save_path
                ):
        self.classifier = classifier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_materials_paths = X_materials_paths
        self.norm_type = norm_type
        self.norm_size = norm_size
        self.fgsm_eps = fgsm_eps
        self.uap_iter = uap_iter
        self.targeted = targeted
        self.save_path = save_path

    ### compute the attack success rate
    # images: ndarray, target image set
    # noise: ndarray, UAP
    def my_calc_fooling_ratio(self, images=0, noise=0):
        adv_images = images + noise
        if self.targeted < 0:
            preds = np.argmax(self.classifier.predict(images), axis=1)
            preds_adv = np.argmax(self.classifier.predict(adv_images), axis=1)
            fooling_ratio = np.sum(preds_adv != preds) / images.shape[0]
            return fooling_ratio
        else:
            preds_adv = np.argmax(self.classifier.predict(adv_images), axis=1)
            fooling_ratio_targeted = np.sum(preds_adv == self.targeted) / adv_images.shape[0]
            return fooling_ratio_targeted

    ### generate the labels (in one-hot vector representation) for targeted attacks
    # length: int, number of target images
    def my_target_labels(self, length=0):
        classes = self.y_train.shape[1]
        return utils.to_categorical([self.targeted] * length, classes)

    ### generate UAP
    def my_gen_UAP(self):
        num_m = len(self.X_materials_paths)
        imshape = self.X_train[0].shape
        
        if self.targeted >= 0:
            print(" *** targeted attack *** \n")
            adv_crafter = UniversalPerturbation(
                self.classifier,
                attacker='fgsm',
                delta=0.000001,
                attacker_params={"targeted":True, "eps":self.fgsm_eps},
                max_iter=self.uap_iter,
                eps=self.norm_size,
                norm=self.norm_type)
        else:
            print(" *** non-targeted attack *** \n")
            adv_crafter = UniversalPerturbation(
                self.classifier,
                attacker='fgsm',
                delta=0.000001,
                attacker_params={"eps":self.fgsm_eps},
                max_iter=self.uap_iter,
                eps=self.norm_size,
                norm=self.norm_type)

        # initialization
        LOG = []
        X_materials_cnt = 0
        noise = np.zeros(imshape)
        noise = noise.astype('float32')
        for i, path in enumerate(self.X_materials_paths):
            X_materials = np.load(path)
            X_materials_cnt += X_materials.shape[0]
            # normalization
            X_materials -= 128.0
            X_materials /= 128.0 

            # craft UAP
            if self.targeted >= 0:
                # generate the labels for targeted attacks
                Y_materials_tar = self.my_target_labels(length=X_materials.shape[0])
                noise = adv_crafter.generate(X_materials, noise=noise,  y=Y_materials_tar, targeted=True)
            else:
                noise = adv_crafter.generate(X_materials, noise=noise)
            
            # handling for no noise
            if type(adv_crafter.noise[0,:]) == int:
                noise = np.zeros(imshape)
            else:
                noise = np.copy(adv_crafter.noise)
                noise = np.reshape(noise, imshape)

            # generate random UAP whose size equals to the size of the UAP
            noise_size = float(np.linalg.norm(noise.reshape(-1), ord=self.norm_type))
            noise_random = random_sphere(
                nb_points=1,
                nb_dims=np.prod(X_materials[0].shape),
                radius=noise_size,
                norm=self.norm_type
            ).reshape(imshape)

            # compute attack success rate of UAP
            # for X_train
            fr_train = self.my_calc_fooling_ratio(images=self.X_train, noise=noise)
            # for X_test
            fr_test = self.my_calc_fooling_ratio(images=self.X_test, noise=noise)
            # for X_materials
            fr_m = self.my_calc_fooling_ratio(images=X_materials, noise=noise)
            # compute attack success rate of random UAP (random control)
            # for X_train
            fr_train_r = self.my_calc_fooling_ratio(images=self.X_train, noise=noise_random)
            # for X_test
            fr_test_r = self.my_calc_fooling_ratio(images=self.X_test, noise=noise_random)
            # for X_materials
            fr_m_r = self.my_calc_fooling_ratio(images=X_materials, noise=noise_random)

            # compute UAP size
            norm_2 = np.linalg.norm(noise)
            norm_inf = abs(noise).max()

            LOG.append([X_materials_cnt, norm_2, norm_inf, fr_train, fr_test, fr_m, fr_train_r, fr_test_r, fr_m_r])
            print("LOG: {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(X_materials_cnt, norm_2, norm_inf, fr_train, fr_test, fr_m, fr_train_r, fr_test_r, fr_m_r))
            del(X_materials) # for saving memory
        np.save(self.save_path+'_noise', noise)
        np.save(self.save_path+'_LOG', np.array(LOG))
        return noise, np.array(LOG)


### cofiguration of classifier
# model_type: 'InceptionV3', 'VGG16', 'ResNet50'
# model_path: str, path to model weight
# output_class: int, number of classes
# mono: int, monochrome images if mono = 1, RGB images otherwise
# silence: int, prevent to output model summary if silence = 1, not otherwise
class my_DNN:
    def __init__(
                self, 
                model_type, 
                model_path,
                output_class, 
                mono, 
                silence
                ):
        self.model_type = model_type
        self.model_path = model_path
        self.output_class = output_class
        self.mono = mono
        self.silence = silence

    def my_classifier(self):
        if self.mono==1:
            if self.model_type == 'InceptionV3':
                print(" MODEL: InceptionV3")
                base_model = InceptionV3(weights='imagenet', include_top=False)
            elif self.model_type == 'VGG16':
                print(" MODEL: VGG16")
                base_model = VGG16(weights='imagenet', include_top=False)
            elif self.model_type == "ResNet50":
                print(" MODEL: ResNet50")
                base_model = ResNet50(weights='imagenet', include_top=False)
            else:
                print(" --- ERROR : UNKNOWN MODEL TYPE --- ")
            base_model.layers.pop(0)
            newInput = Input(batch_shape=(None, 299,299,1))
            x = Lambda(lambda image: tf.image.grayscale_to_rgb(image))(newInput)
            tmp_out = base_model(x)
            tmpModel = Model(newInput, tmp_out)
            x = tmpModel.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(self.output_class, activation='softmax')(x)
            model = Model(tmpModel.input, predictions)
        else:
            input_shape = (299, 299, 3)
            if self.model_type == 'InceptionV3':
                print(" MODEL: InceptionV3")
                base_model = InceptionV3(weights='imagenet', input_shape=input_shape, include_top=False)
            elif self.model_type == 'VGG16':
                print(" MODEL: VGG16")
                base_model = VGG16(weights='imagenet', input_shape=input_shape, include_top=False)
            elif self.model_type == "ResNet50":
                print(" MODEL: ResNet50")
                base_model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)
            else:
                print(" --- ERROR: UNKNOWN MODEL TYPE --- ")
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(self.output_class, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)

        for layer in model.layers:
            layer.trainable = True
        
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(self.model_path)
        if self.silence != 1:
            model.summary() 
        classifier = KerasClassifier(model=model)
        return classifier

### Main ###
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--X_train_path', type=str)
    parser.add_argument('--y_train_path', type=str)
    parser.add_argument('--X_test_path', type=str)
    parser.add_argument('--y_test_path', type=str)
    parser.add_argument('--X_materials_dir', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--norm_type', type=str)
    parser.add_argument('--norm_rate', type=float)
    parser.add_argument('--fgsm_eps', type=float)
    parser.add_argument('--uap_iter', type=int)
    parser.add_argument('--targeted', type=int)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()
    if args.norm_type == '2':
        norm_type = 2
    elif args.norm_type == 'inf':
        norm_type = np.inf
    norm_rate = args.norm_rate

    # load data
    X_train = np.load(args.X_train_path)
    y_train = np.load(args.y_train_path)
    X_test = np.load(args.X_test_path)
    y_test = np.load(args.y_test_path)

    # obtain the file names of X_materials
    X_materials_paths = glob.glob(args.X_materials_dir + '/*.npy')

    # check color type (mono or RGB)
    if X_train.shape[-1] != 3:
        mono = 1
    else:
        mono = 0

    # compute the actual norm size from the ratio `norm_rate` of the Lp of the UAP to the average Lp norm of an image in the dataset (training images)
    if norm_type == np.inf:
        norm_mean = 0
        for img in X_train:
            norm_mean += abs(img).max()
        norm_mean = norm_mean/X_train.shape[0]
        norm_size = float(norm_rate*norm_mean/128.0)
        print("\n ------------------------------------")
        print(" Linf norm: {:.2f} ".format(norm_size))   
    else:
        norm_mean = 0
        for img in X_train:
            norm_mean += np.linalg.norm(img)
        norm_mean = norm_mean/X_train.shape[0]
        norm_size = float(norm_rate*norm_mean/128.0)
        print(" L2 norm: {:.2f} ".format(norm_size))   

    # normalization
    X_train -= 128.0
    X_train /= 128.0
    X_test -= 128.0
    X_test /= 128.0

    dnn = my_DNN(
                model_type=args.model_type, 
                model_path=args.model_path,
                output_class=y_train.shape[1], 
                mono=mono, 
                silence=1
                )
    classifier = dnn.my_classifier()

    # compute the accuracies for clean images
    preds = np.argmax(classifier.predict(X_train), axis=1)
    acc = np.sum(preds == np.argmax(y_train, axis=1)) / y_train.shape[0]
    print(" Accuracy [train]: {:.2f}".format(acc))
    preds = np.argmax(classifier.predict(X_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print(" Accuracy [test]: {:.2f}".format(acc))  
    print(" ------------------------------------\n")

    # generate UAP
    uap = my_UAP(
                classifier=classifier, 
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test, 
                X_materials_paths=X_materials_paths,
                norm_type=norm_type, 
                norm_size=norm_size, 
                fgsm_eps=args.fgsm_eps, 
                uap_iter=args.uap_iter, 
                targeted=args.targeted,
                save_path=args.save_path,
                )
    
    noise, LOG = uap.my_gen_UAP()

    # output log
    # X_materials_cnt, norm_2, norm_inf, fr_train, fr_test, fr_m, fr_train_r, fr_test_r, fr_m_r
    plt.figure()
    plt.ylim(0, LOG[:,0][-1])
    plt.ylim(0, 1)
    p1 = plt.plot(LOG[:,0], LOG[:,3], linewidth=3, color="darkred", linestyle="solid", label="fr_train")
    p2 = plt.plot(LOG[:,0], LOG[:,4], linewidth=3, color="darkblue", linestyle="solid", label="fr_test")
    p3 = plt.plot(LOG[:,0], LOG[:,5], linewidth=3, color="dimgray", linestyle="solid", label="fr_matel")
    p4 = plt.plot(LOG[:,0], LOG[:,6], linewidth=3, color="lightcoral", linestyle="dashed", label="fr_train_r")
    p5 = plt.plot(LOG[:,0], LOG[:,7], linewidth=3, color="lightblue", linestyle="dashed", label="fr_test_r") 
    p6 = plt.plot(LOG[:,0], LOG[:,8], linewidth=3, color="lightgray", linestyle="dashed", label="fr_matel_r") 
    plt.xlabel("# of iterations (natural images)")
    plt.ylabel("Attack success rate")
    plt.legend(loc='lower right')  
    plt.grid(True)
    plt.savefig(args.save_path+'_fig.png')

    # output processing time
    processing_time = time.time() - start_time
    print("\n\t ------------------------------------")
    print("\t   total processing time : {:.2f} h.".format(processing_time / 3600.0))
    print("\t ------------------------------------\n")

