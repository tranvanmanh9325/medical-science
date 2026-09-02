import os
import imageio
import numpy as np

def read_images_from_path(directory):
    images = []
    for filename in os.listdir(directory):
        if '_left.jpg' in filename:
            filename = os.path.join(directory, filename)
            image = imageio.imread(filename)

            image_array = np.array(image)
            images.append(image_array)
        
    return np.stack(images)

# def read_images_from_path(directory):
#     for filename in os.listdir(directory):
#         print(filename)

imgs = read_images_from_path('/research/d1/rshr/jwfu/CSR/data_jiawei_new/move/20220826-160914-SNXXXXXX.mp4_move_Psa1_4')
print(imgs.shape)

