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

Stylegan implementations also were tested - each of them failed because of internal libraries errors and libraries incompatibilities.
There was a tes of running Startagn on a computer with GPU - unfortunately it ended in unsuccessful augmentation.
Solutions from the Hugging Face like https://huggingface.co/spaces/fffiloni/expression-editor were also tested. 
This solution would need processing manually thousands of images. It was tried to automate the data ingestion pipeline but was not completed due to memory resources lacking.


"""

BASE_DATA_PATH = rf"C:\Users\wdomc\Documents\personal_color_analysis\dataset_PCoIA_model_based_augmented"






