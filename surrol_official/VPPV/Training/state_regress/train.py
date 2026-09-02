import torch
import torch.nn
import os
from vmodel import vismodel
from VisDataset import ObsDataset
from torch.utils.data import DataLoader
from utils import Logger, WandBLogger, logger
import numpy as np
from config import opts

class Coach:
	def __init__(self, opts):
		self.opts=opts
		self.global_step=0
		self.device = 'cuda:0'
		self.opts.device=self.device
		self.net=vismodel(self.opts).to(self.device)
		self.vis_optimizer=torch.optim.Adam(
			self.net.parameters(), lr=self.opts.vis_lr
		)

		if self.opts.continue_training:
			ckpt=torch.load(opts.ckpt_dir, map_location=self.device)['state_dict']
			self.net.load_state_dict(ckpt)
			print("--continue training from checkpoint--")

		ecm_view_matrix =[2.7644696427853166e-12, -0.8253368139266968, 0.5646408796310425, 0.0, 1.0, 2.76391192918779e-12, -8.559629784479772e-13, 0.0, -8.541598418149166e-13, 0.5646408796310425, 0.8253368139266968, 0.0, -1.582376590869572e-11, 0.4536721706390381, -5.886332988739014,1.0]
        #ecm_view_matrix[14]=-5.25 #-5.0#-5.25
		ecm_view_matrix[14]=-0.97 #-4.7 #-5.25#-5.0#-5.25
		ecm_view_matrix[13]=0.07 #0.3-0.5
		# ecm_view_matrix[12]=0.004 #0.3-0.5
		shape_view_matrix=np.array(ecm_view_matrix).reshape(4,4)
		Tc = np.array([[1,  0,  0,  0],
						[0, -1,  0,  0],
						[0,  0, -1,  0],
						[0,  0,  0,  1]])
		self.view_matrix=Tc@(shape_view_matrix.T)

		#self.view_matrix=np.array(ecm_view_matrix).reshape(4,4)
		self.train_dataset=ObsDataset(self.opts, self.view_matrix, add_noise=True)
		self.test_dataset =ObsDataset(self.opts, self.view_matrix, status="test")


		self.train_dataloader = DataLoader(self.train_dataset,
											batch_size=self.opts.batch_size,
											shuffle=True,
											num_workers=int(self.opts.workers),
											drop_last=True)

		self.test_dataloader = DataLoader(self.test_dataset,
											batch_size=self.opts.test_batch_size,
											shuffle=False,
											num_workers=int(self.opts.test_workers),
											drop_last=True)
		print("----all data loaded----")
		if not os.path.exists(self.opts.work_dir):
			os.mkdir(self.opts.work_dir)

		self.checkpoint_dir = os.path.join(opts.work_dir, 'checkpoints')
		if not os.path.exists(self.checkpoint_dir):
			os.mkdir(self.checkpoint_dir)
		self._setup_logger()
		self.best_val_loss=None

	def _setup_logger(self):
		#update_mpi_config(self.cfg)

		exp_name = f"seed{self.opts.seed}"
		if self.opts.postfix is not None:
			exp_name =  exp_name + '_' + str(self.opts.postfix) 
		if self.opts.use_wb:
			self.wb = WandBLogger(exp_name=exp_name, project_name=self.opts.project_name, entity=self.opts.entity_name, \
					path=self.opts.work_dir, conf=self.opts)
		self.logger = Logger(self.opts.work_dir)
		self.termlog = logger

	def train(self):
		"----start train----"
		self.net.train()
		while self.global_step<self.opts.max_steps:
			print("global step: ",self.global_step)
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.vis_optimizer.zero_grad()
				rgb=batch["img"].to(self.device).float()
				seg=batch["seg"].to(self.device).float()
				state=batch["state"].to(self.device).float()

				metrics,v_loss=self.net(seg, rgb, state)
				
				v_loss.backward()
				self.vis_optimizer.step() 

				self.global_step+=1

				if metrics is not None and (self.global_step%self.opts.log_interval==0):
					#print("global step: ",self.global_step)
					print("log model")
					self.logger.log_metrics(metrics, self.global_step, ty='train')
					if self.opts.use_wb:
						self.wb.log_outputs(metrics, None, log_images=False, step=self.global_step, is_train=True)
					

				if self.global_step%self.opts.eval_interval==0:
					print("eval model")
					metrics=self.eval()
					if (self.best_val_loss is None or metrics['v_loss'] < self.best_val_loss):
						self.best_val_loss = metrics['v_loss']
						self.checkpoint_me(metrics, is_best=True)

				if metrics is not None and self.global_step%self.opts.save_interval==0:
					self.checkpoint_me(metrics, is_best=False)
					print("save model")

	def eval(self):
		self.net.eval()
		for batch_idx, batch in enumerate(self.train_dataloader):
			
			rgb=batch["img"].to(self.device)
			seg=batch["seg"].to(self.device)
			state=batch["state"].to(self.device)
			with torch.no_grad():
				metrics,v_loss=self.net(seg, rgb, state)
			
			self.logger.log_metrics(metrics, self.global_step, ty='eval')
			self.wb.log_outputs(metrics, None, log_images=False, step=self.global_step, is_train=False)

		self.net.train()
		return metrics

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		return save_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))


if __name__=="__main__":

	trainer=Coach(opts)
	trainer.train()