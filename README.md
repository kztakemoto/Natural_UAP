# Natural UAP

This repository contains the codes used in our study on [*Natural images allow universal adversarial attacks on medical image classification using deep neural networks with transfer learning*](https://www.mdpi.com/2313-433X/8/2/38).

# Terms of use

MIT licensed. Happy if you cite our preprint when using the codes:

Minagi A, Hirano H & Takemoto K (2022) **Natural images allow universal adversarial attacks on medical image classification using deep neural networks with transfer learning.** J. Imaging 8, 32. doi:10.3390/jimaging8020038

## Usage

### 1. Medical images and DNN models
See [hkthirano/MedicalAI-UAP](https://github.com/hkthirano/MedicalAI-UAP) for details. The images and DNNs (model weights) will be stored in the `data` directory.

### 2. Requirements
* Python 3.7.0
```
pip install -r requirements.txt
```

### 3. Natural Images
```
# Directories
.
└── Natural_UAP
    └── datasets
        ├── ILSVRC2012
        └── open-images-dataset
```

#### ImageNet Dataset
* download [the training images of the ImageNet dataset](https://www.image-net.org/download.php).
* convert the images to npy files.
```
python make_data_imagenet.py 
```

#### Open Images Dataset
* download [the test images of the Open Images Dataset (V6)](https://storage.googleapis.com/openimages/web/download.html).
* convert the images to npy files.
```
python make_data_openimages.py 
```

### 4. Generating UAP using Natural Images
Given the directory structure in [hkthirano/MedicalAI-UAP](https://github.com/hkthirano/MedicalAI-UAP), a use example is as follows:

```
python run_Natural_UAP.py \
--X_train_path './data/melanoma/X_train.npy' \
--y_train_path './data/melanoma/y_train.npy' \
--X_test_path './data/melanoma/X_test.npy' \
--y_test_path './data/melanoma/y_test.npy' \
--X_materials_dir './data/ImageNet/color' \
--model_path './data/melanoma/model/inceptionv3.h5' \
--model_type 'InceptionV3' \
--norm_type '2' \
--norm_rate 0.04 \
--fgsm_eps 0.0005 \
--uap_iter 1 \
--targeted -1 \
--save_path './results/melanoma_imagenet_inceptionv3_L2norm_zeta4_fgsmeps00005'
```
