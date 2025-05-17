import cv2
import os
import numpy as np
import shutil

"""

Multiple solutions were tested:
- different implementations of StarGan (both PyTorch and Tensorflow ones) and StyleGAN were tested, for example:
https://github.com/yunjey/stargan
https://github.com/clovaai/stargan-v2-tensorflow
https://github.com/NVlabs/stylegan

Intergan and Stylegan implementations also were tested - each of them failed because of internal libraries errors and libraries incompatibilities.
Found solutions on Hugging Face like: 
https://huggingface.co/spaces/fffiloni/expression-editor

"""

BASE_DATA_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\dataset_PCoIA_model_based_augmented"






