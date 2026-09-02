import os
import copy

import numpy as np


from torch.utils.data import Dataset

# Copy from SurRoL simulator
GOAL_DISTANCE = 0.005
SCALE = 1.

class SurRoLDataset(Dataset):
    '''Maximum sequence length is set with 50, 
       demonstrations from scipt will follow this automatically,
       human demonstrations will be downsampled to follow this rule.
       The main concern locates overlong sequence may cause negative influence to performance.'''

    def __init__(self, file_root, seq_max_length, task, with_relabel=True):
        '''Preprocess data from offline dataset'''

        self.T = seq_max_length
        # assert self.T == 100 # Debug
        self.returns_to_goal_scale = self.T
        self.times_to_goal_scale = self.T

        self.data = []
        observations = []
        goals = []

        # Get all files in the file folder
        g = os.walk(file_root)
        for path, _, file_list in g:  
            for file_name in file_list:  
                demo_path = os.path.join(path, file_name)
                if '_' + task + '_' not in demo_path:
                    continue

                print('[SurRoLDataset] valid demo_path: {}'.format(demo_path))
                demo = np.load(demo_path, allow_pickle=True)
                demo_obs, demo_acs = demo['obs'], demo['acs']


                for idx in range(len(demo_obs)):

                    # DEBUG
                    # print('[SurRoLDataset] demo observation length: {}'.format(len(demo_obs[idx])))
                    # print('[SurRoLDataset] demo action length: {}'.format(len(demo_acs[idx])))
                    # END DEBUG

                    seq = {
                        'observations': [],
                        'achieved_goals': [],
                        'goals': [],
                        'actions': [],
                        'rewards': [],
                        'returns_to_goal': [],
                        'times_to_goal': [],
                        'timesteps': [],
                        'success': False,
                        'success_t': -1,
                    }
                    for t in range(len(demo_acs[idx])):
                        # Last observation is remove to match decision transformer's input formulation
                        seq['observations'].append(demo_obs[idx][t]['observation'])
                        observations.append(demo_obs[idx][t]['observation'])
                        seq['achieved_goals'].append(demo_obs[idx][t]['achieved_goal'])
                        seq['goals'].append(demo_obs[idx][t]['desired_goal'])
                        goals.append(demo_obs[idx][t]['desired_goal'])
                        seq['actions'].append(demo_acs[idx][t])
                        seq['rewards'].append(0.0 if np.linalg.norm(demo_obs[idx][t]['achieved_goal'] - demo_obs[idx][t]['desired_goal'], axis=-1) <= GOAL_DISTANCE * SCALE else -1.)
                        if seq['rewards'][-1] == 0.0:
                            seq['success'] = True
                            seq['success_t'] = t
                        seq['timesteps'].append(t)

                    seq['times_to_goal'] = SurRoLDataUtils.calculate_times_to_goal(seq, len(seq['observations']))
                    seq['returns_to_goal'] = SurRoLDataUtils.calculate_returns_to_goal(seq, len(seq['observations']))

                    seq['observations'] = np.array(seq['observations'])
                    seq['achieved_goals'] = np.array(seq['achieved_goals'])
                    seq['goals'] = np.array(seq['goals'])
                    seq['actions'] = np.array(seq['actions'])
                    seq['rewards'] = np.array(seq['rewards'])
                    seq['returns_to_goal'] = np.array(seq['returns_to_goal'])
                    seq['times_to_goal'] = np.array(seq['times_to_goal'])
                    seq['timesteps'] = np.array(seq['timesteps'])

                    self.data.append(seq)
        
        # Hindsight propagate data
        if with_relabel:
            relabeller = HindsightRelabeller(50)
            prop_data = relabeller.step_batch(self.data)
        
            print('[SurRoLDataset] original data sequence number: {}'.format(len(self.data)))
            print('[SurRoLDataset] propagated data sequence number: {}'.format(len(prop_data)))

            self.data += prop_data

        # Calculate mean and std for observation and goal
        observations = np.array(observations)
        goals = np.array(goals)
        
        self.observations_mean = np.mean(observations, axis=0)
        self.observations_std = np.std(observations, axis=0) + 1e-6
        self.goals_mean = np.mean(goals, axis=0)
        self.goals_std = np.std(goals, axis=0) + 1e-6
    
    def pop_normalization_parameters(self):
        return self.observations_mean, self.observations_std, self.goals_mean, self.goals_std


    def __getitem__(self, index):
        observations = self.data[index]['observations']
        goals = self.data[index]['goals']
        actions = self.data[index]['actions']
        rewards = self.data[index]['rewards']
        returns_to_goal = self.data[index]['returns_to_goal']
        times_to_goal = self.data[index]['times_to_goal']
        timesteps = self.data[index]['timesteps']

        # Padding to maximum sequence length
        seq_length = observations.shape[0]
        observations = np.concatenate([np.zeros((self.T - seq_length, observations.shape[-1])), observations], axis=0)
        goals = np.concatenate([np.zeros((self.T - seq_length, goals.shape[-1])), goals], axis=0)
        actions = np.concatenate([np.ones((self.T - seq_length, actions.shape[-1])) * -10., actions], axis=0)
        rewards = np.concatenate([np.ones((self.T - seq_length, 1)) * -1., np.expand_dims(rewards, axis=1)], axis=0)
        returns_to_goal = np.concatenate([np.ones((self.T - seq_length, 1)) * -self.T, np.expand_dims(returns_to_goal, axis=1)], axis=0)
        times_to_goal = np.concatenate([np.ones((self.T - seq_length, 1)) * self.T, np.expand_dims(times_to_goal, axis=1)], axis=0)
        timesteps = np.concatenate([np.zeros((self.T - seq_length)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.T - seq_length)), np.ones((seq_length))], axis=0)

        # Normalization for observations and goals
        observations = (observations - self.observations_mean) / self.observations_std
        goals = (goals - self.goals_mean) / self.goals_std
        returns_to_goal = returns_to_goal / self.returns_to_goal_scale
        times_to_goal = times_to_goal / self.times_to_goal_scale

        # print('[DEBUG] observation shape: {}'.format(observations.shape))
        # print('[DEBUG] goals shape: {}'.format(goals.shape))
        # print('[DEBUG] actions shape: {}'.format(actions.shape))
        # print('[DEBUG] rewards shape: {}'.format(rewards.shape))
        # print('[DEBUG] returns_to_goal shape: {}'.format(returns_to_goal.shape))
        # print('[DEBUG] times_to_goal shape: {}'.format(times_to_goal.shape))
        # print('[DEBUG] timesteps shape: {}'.format(timesteps.shape))
        # print('[DEBUG] mask shape: {}'.format(mask.shape))

        return observations, goals, actions, rewards, returns_to_goal, times_to_goal, timesteps, mask

    def __len__(self):
        return len(self.data)
    

class SurRoLDataUtils:
    @staticmethod
    def calculate_times_to_goal(seq, sequence_length):
        if seq['success']:
            times_to_goal = [seq['success_t'] - t for t in range(0, seq['success_t'] + 1)] + \
                            [0 for _ in range(sequence_length - seq['success_t'] - 1)]
        else:
            times_to_goal = [sequence_length for _ in range(sequence_length)]
        
        return times_to_goal

    @staticmethod
    def calculate_returns_to_goal(seq, sequence_length):
        if seq['success']:
            returns_to_goal = [-1.0 * (seq['success_t'] - t) for t in range(0, seq['success_t'] + 1)] + \
                              [0. for _ in range(sequence_length - seq['success_t'] - 1)]
        else:
            returns_to_goal = [-1.0 * sequence_length for _ in range(sequence_length)]
        
        return returns_to_goal

class HindsightRelabeller:
    '''Truncate sequence length to match succes index.'''

    def __init__(self, propagate_num):
        self.propagate_num = propagate_num

    def step(self, sequence):
        seq_length = len(sequence['observations'])
        hindsight_indices = np.random.randint(0, seq_length, size=(self.propagate_num))

        prop_sequences = []
        for idx in hindsight_indices:
            prop_seq = copy.deepcopy(sequence)

            # Change desired goal with achieved goal
            prop_seq['goals'][idx:] = prop_seq['achieved_goals'][idx:]
            prop_seq['success_t'] = idx
            prop_seq['success'] = True

            # Truncate sequence length to remove redundant items
            prop_seq['observations'] = prop_seq['observations'][:idx+1]
            prop_seq['achieved_goals'] = prop_seq['achieved_goals'][:idx+1]
            prop_seq['goals'] = prop_seq['goals'][:idx+1]
            prop_seq['actions'] = prop_seq['actions'][:idx+1]
            prop_seq['rewards'] = prop_seq['rewards'][:idx+1]
            prop_seq['timesteps'] = prop_seq['timesteps'][:idx+1]

            # Recalculate times to goal and returns to goal
            prop_seq['times_to_goal'] = np.array(SurRoLDataUtils.calculate_times_to_goal(prop_seq, idx + 1))
            prop_seq['returns_to_goal'] = np.array(SurRoLDataUtils.calculate_returns_to_goal(prop_seq, idx + 1))

            prop_sequences.append(prop_seq)

            # print('[HindsightRelabeller] truncated sequence length: {}'.format(len(prop_seq['observations'])))

            # print('[HindsightRelabeller] observations shape: {}'.format(prop_seq['observations'].shape))
            # print('[HindsightRelabeller] achieved_goals shape: {}'.format(prop_seq['achieved_goals'].shape))
            # print('[HindsightRelabeller] goals shape: {}'.format(prop_seq['goals'].shape))
            # print('[HindsightRelabeller] actions shape: {}'.format(prop_seq['actions'].shape))
            # print('[HindsightRelabeller] rewards shape: {}'.format(prop_seq['rewards'].shape))
            # print('[HindsightRelabeller] timesteps shape: {}'.format(prop_seq['timesteps'].shape))
            # print('[HindsightRelabeller] times_to_goal shape: {}'.format(prop_seq['times_to_goal'].shape))
            # print('[HindsightRelabeller] returns_to_goal shape: {}'.format(prop_seq['returns_to_goal'].shape))


        return prop_sequences

    def step_batch(self, sequences):
        prop_sequence_batch = []
        for seq in sequences:
            prop_sequence_batch += self.step(seq)
        return prop_sequence_batch
    



if __name__ == '__main__':
    root = '/home/jwfu/goal_conditioned_dt/data/single_demo/'

    surrol_dataset = SurRoLDataset(root, 50, with_relabel=True)
    print(len(surrol_dataset))

    from torch.utils.data import DataLoader

    surrol_dataloader = DataLoader(dataset=surrol_dataset, batch_size=128, shuffle=True)


    for observations, goals, actions, rewards, \
        returns_to_goal, times_to_goal, timesteps, mask in surrol_dataloader:
        print('observations shape: {}'.format(observations.shape))
        print('goals shape: {}'.format(goals.shape))
        print('actions shape: {}'.format(actions.shape))
        print('rewards shape: {}'.format(rewards.shape))
        print('returns_to_goal shape: {}'.format(returns_to_goal.shape))
        print('times_to_goal shape: {}'.format(times_to_goal.shape))
        print('timesteps shape: {}'.format(timesteps.shape))
        print('mask shape: {}'.format(mask.shape))
