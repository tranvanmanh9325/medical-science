
## Installation

The project is built on Ubuntu 20.04 with Python 3.7.

### 1. Install prerequisites

Run following commands in the terminal to install build-essential and cmake:

 ```shell
sudo apt-get install build-essential
sudo apt-get install cmake
 ```

Install Anaconda following the [Official Guideline](https://www.anaconda.com/).


### 2. Prepare environment

Create conda virtual environments for 'data_generation' or 'policy_learning' and activate it:

 ```shell
conda env create -f environment.yaml
conda activate ${env_name}
 ```

### 3. Install SurRoL

Install SurRoL in the created conda environment:

   ```shell
   cd ${SurRoL_path}
   pip install -e .
   pip install -r MPM/requirements.txt
   ```

## Training Policy in Simulator

### 1. Data Generation

* Go to directory 'data_generation'

* Change directory path for data saving in the task python file (such as tasks/needle_grasping.py) and dataset_process.py

* Then run the following command:
   ```
   python data_generation.py --env ${task_name}
   python dataset_process.py
   ```

### 2. Train Perceptual Regressor
* Edit the stateregress/config.py with your own path (data_dir, seed, wandb...)

* Then run the following command:
   ```
   python train.py
   ```

### 3. Policy Learning

* Go to directory 'surrol'

* Edit the "ckpt_dir" in stateregress/config.py with the best model path of perceptual regressor

* Edit rl/configs/train.yaml. Basically only "seed", wandb related items and the following items should be changed:
   ```
   # train.yaml
   resume_training: False
   reload_vis: False
   restart_vis: False
   ckpt_dir: ''  
   vis_ckpt_dir: '' 
   ckpt_episode: best
   buffer_dir: None
   test: False
   ```
* Edit rl/configs/agent/ddpg.yaml: n_seed_steps=1000

* Setup the executable path of ffmpeg for wandb:
   ```
   export IMAGEIO_FFMPEG_EXE=/home/yhlong/anaconda3/envs/surrol/lib/python3.7/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2
   ```

* Add one line of code at the top of gym/envs/__init__.py to register SurRoL tasks:
   ```
   # path is located at somewhere similar to: anaconda3/envs/surrol/lib/python3.7/site-packages/
   import surrol.gym
   ```

* Then run the following command:
   ```
   python3 rl/train.py task=${task_name} agent=ddpg use_wb=True
   ```

### 4. Policy Testing
* Edit rl/configs/train.yaml.
   ```
   # train.yaml
   resume_training: True
   reload_vis: False
   restart_vis: False
   ckpt_dir: # please find the checkpoint dir you want to test
   vis_ckpt_dir: '' 
   ckpt_episode: best
   buffer_dir: None
   test: True
   ```
* Edit "seed", and wandb setting as needed.
* Edit rl/configs/agent/ddpg.yaml: n_seed_steps=0