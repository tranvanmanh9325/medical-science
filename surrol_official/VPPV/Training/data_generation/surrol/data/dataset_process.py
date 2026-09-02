import numpy as np
import pickle
import os
import cv2

train_file='train_list.npy'
test_file='test_list.npy'

# change here
depth_dir='/home/yhlong/project/VPPV_checking/data/vessel/depth/'
seg_dir = '/home/yhlong/project/VPPV_checking/data/vessel/seg_npy/'
outdir='/home/yhlong/project/VPPV_checking/data/vessel/'


def get_pickle(file_list, img_dir, status, filetype, outdir):
    data_list=np.load(file_list)
    
    collect_data={}
    for cid in data_list:
        filepath=os.path.join(img_dir, '{}_{}.npy'.format(filetype, cid))
        
        #filepath=os.path.join(img_dir, cid)
        item=np.load(filepath)

        #cid=cid.split('_')[1].split('.')[0]
        collect_data[cid]=item
        
    with open(os.path.join(outdir,'{}_{}.pkl'.format(status,filetype)),"wb") as f:
        pickle.dump(collect_data, f)

# for depth testset
get_pickle(test_file, depth_dir, status='test', filetype='depth',outdir=outdir)

# for depth trainset
get_pickle(train_file, depth_dir, status='train', filetype='depth',outdir=outdir)

# for seg testset
get_pickle(test_file, seg_dir, status='test', filetype='seg',outdir=outdir)

# for seg trainset
get_pickle(train_file, seg_dir, status='train', filetype='seg',outdir=outdir)

