import numpy as np
import pickle
import os
import cv2

train_file='train_list.npy'
test_file='test_list.npy'

data_list=np.load(test_file)

print("data_list: ", type(data_list))
print("data_list: ", len(data_list))
print("data_list: ", np.max(data_list))
print("data_list: ", np.min(data_list))

for i, data in enumerate(data_list):
    data_list[i] = data+1

print("data_list: ", type(data_list))
print("data_list: ", len(data_list))
print("data_list: ", np.max(data_list))
print("data_list: ", np.min(data_list))

np.save(test_file, data_list)

