from easydict import EasyDict as edict 
import os

opts=edict()
opts.data_dir='/research/d1/rshr/arlin/regress_data6'
opts.obs_list=os.path.join(opts.data_dir,'img_obs.pkl')
#opts.seg_file=os.path.join(opts.data_dir,"seg_npy")
opts.img_dir=os.path.join(opts.data_dir,"img")
opts.vis_lr=1e-5
opts.batch_size=32
opts.workers=2
opts.test_batch_size=1
opts.test_workers=2

opts.base_dir='/research/d1/rshr/arlin/StateRegress'
opts.img_size=(240,320)
opts.seed=16
opts.postfix='again_close_scaling1_d_matrix_exist_depth'
opts.work_dir=os.path.join(opts.base_dir,"exp_{}{}".format(opts.seed,opts.postfix))
opts.use_wb=True
opts.project_name='regress'
opts.entity_name='picksh'
opts.max_steps=100000000
opts.log_interval=200
opts.eval_interval=1000
opts.save_interval=1000
opts.ckpt_dir='./pretrained_models/vein_sr_best_model.pt'
opts.use_exist_depth=True
opts.continue_training=False
