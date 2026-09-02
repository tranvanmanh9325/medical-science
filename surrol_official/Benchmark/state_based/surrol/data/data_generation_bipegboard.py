"""
Data generation for the case of Psm Envs and demonstrations.
Refer to
https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py
"""
import os
import argparse
import gym
import time
import numpy as np
import imageio
from surrol.const import ROOT_DIR_PATH
from rl.components.envrionment import BiPegBoardSLWrapper

parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
parser.add_argument('--env', type=str, required=True,
                    help='the environment to generate demonstrations')
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')
parser.add_argument('--steps', type=int,
                    help='how many steps allowed to run')
parser.add_argument('--subtask', type=str,
                    help='how many steps allowed to run', default='grasp')
args = parser.parse_args()

actions = []
observations = []
infos = []
terminals = []
images = []  # record video
masks = []
gt_actions = []
global success_counter 


SUBTASK_START = {
    'grasp': 0,
    'handover': 4,
}

SUBTASK_END = {
    'grasp': 5,
    'handover': 9,
    'release': 12
}


def main():
    env = gym.make(args.env, render_mode='rgb_array')  # 'human'
    env = BiPegBoardSLWrapper(env, output_raw_obs=True, subtask=args.subtask)
    num_itr = 100 if not args.video else 5
    cnt = 0
    success_counter = 0
    init_state_space = 'random'
    env.reset()
    print("Reset!")
    init_time = time.time()

    if args.steps is None:
        args.steps = env.max_episode_steps

    print()
    while len(infos) < num_itr:
        obs_, obs = env.reset()
        print("ITERATION NUMBER ", len(infos))
        goToGoal(env, obs_, obs)
        cnt += 1

    file_name = "data_"
    file_name += args.env
    file_name += "_" + init_state_space
    file_name += "_" + str(num_itr)
    file_name += "_primitive_new" + args.subtask + ".npz"

    folder = 'demo' if not args.video else 'video'
    folder = os.path.join(ROOT_DIR_PATH, 'data', folder)

    np.savez_compressed(os.path.join(folder, file_name),
                        actions=actions, observations=observations, terminals=terminals, gt_actions=gt_actions)  # save the file

    if args.video:
        video_name = "video"
        video_name += args.env + ".mp4"
        writer = imageio.get_writer(os.path.join(folder, video_name), fps=20)
        for img in images:
            writer.append_data(img)
        writer.close()

        if len(masks) > 0:
            mask_name = "mask_"
            mask_name += args.env + ".npz"
            np.savez_compressed(os.path.join(folder, mask_name),
                                masks=masks)  # save the file

    used_time = time.time() - init_time
    print("Saved data at:", folder)
    print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
    print(f"Trials: {num_itr}/{cnt}")
    env.close()


def goToGoal(env, last_obs_, last_obs):
    episode_acs = []
    episode_obs = []
    episode_info = []
    episode_terminals = []
    episode_gt_acs = []

    time_step = 0  # count the total number of time steps
    episode_init_time = time.time()
    
    episode_obs.append(last_obs_)

    obs_, obs, success = last_obs_, last_obs, False
    while time_step < min(env.max_episode_steps, args.steps):
        action, i = env.get_oracle_action(obs)
        print(time_step)

        if i == SUBTASK_END[args.subtask]:
            info['is_success'] = 1
            action = np.zeros_like(action)
            if args.subtask == 'grasp':
                action[-1] = -0.5
                action [4] = 0.5
            elif args.subtask == 'handover':
                action[-1] = 0.5
                action[4] = -0.5
            elif args.subtask == 'release':
                action[-1] = -0.5
                action[4] = -0.5

        if args.video:
            # img, mask = env.render('img_array')
            img = env.render('rgb_array')
            images.append(img)
            # masks.append(mask)

        obs_, reward, done, info, obs = env.step(action)
        # print(f" -> obs: {obs}, reward: {reward}, done: {done}, info: {info}.")
        time_step += 1
        print(reward, i)

        if isinstance(obs, dict) and info['is_success'] > 0 and not success:
            print("Timesteps to finish:", time_step)
            success = True

        # # if i >= 4 and i < 9:
        # if i >= SUBTASK_START[args.subtask] and i < SUBTASK_END[args.subtask]:
        episode_acs.append(action)
        episode_info.append(info)
        episode_obs.append(obs_)
        episode_terminals.append(done)
        episode_gt_acs.append(info['gt_goal'])
        last_obs_ = obs_

    print("Episode time used: {:.2f}s\n".format(time.time() - episode_init_time))

    if success:
        actions.append(episode_acs)
        observations.append(episode_obs)
        infos.append(episode_info)
        terminals.append(episode_terminals)
        gt_actions.append(episode_gt_acs)


if __name__ == "__main__":
    main()
