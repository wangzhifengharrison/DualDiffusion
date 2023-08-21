from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils import data_utils_for_bighand
from matplotlib import pyplot as plt
import torch

import os 

'''
adapted from
https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/h36motion3d.py
'''


class BIGHAND(Dataset):

    def __init__(self, data_dir,input_n,output_n,skip_rate, actions=None, split=0, miss_rate=0.2,
                 miss_type='no_miss', all_data=False, joints=21):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir,'datasets/bighand2.2m')
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.miss_rate = miss_rate
        self.miss_type = miss_type
        self.sample_rate = 2
        self.p3d = {}
        self.params = {}
        self.masks = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n
        subs = np.array([['Subject_1', 'Subject_4','Subject_5', 'Subject_6', 'Subject_7','Subject_8', 'Subject_9', 'Subject_10'], [ 'Subject_11'], [ 'Subject_2']])
        # acts = data_utils.define_actions(actions)
        if actions is None:
            acts = ['combined']
        else:
            acts = actions
        # subs = np.array([[1], [11], [5]])
        # acts = ['walking']
        # 32 human3.6 joint name:
        # joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg",
        #               "LeftFoot",
        #               "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
        #               "LeftForeArm",
        #               "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm",
        #               "RightForeArm",
        #               "RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]

        subs = subs[split]
        key = 0
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                if self.split <= 1:
                    print("Reading subject {0}, action {1}".format(subj, action))
                    filename = '{0}/{1}/{2}.txt'.format(self.path_to_data, subj, action)
                    ## read .csv files
                    the_sequence = data_utils_for_bighand.readCSVasFloat(filename)
                    n, d = the_sequence.shape
                    even_list = range(0, n, self.sample_rate)
                    num_frames = len(even_list)
                    the_sequence = np.array(the_sequence[even_list, :])
                    the_sequence = torch.from_numpy(the_sequence).float().cuda()

                    p3d = the_sequence
                    # self.p3d[(subj, action, subact)] = p3d.view(num_frames, -1).cpu().data.numpy()
                    self.p3d[key] = p3d.view(num_frames, -1).cpu().data.numpy()

                    valid_frames = np.arange(0, num_frames - seq_len + 1, skip_rate)

                    # tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    key += 1
                else:
                    print("Reading subject {0}, action {1}".format(subj, action))
                    filename = '{0}/{1}/{2}.txt'.format(self.path_to_data, subj, action)
                    ## read .csv files
                    the_sequence = data_utils_for_bighand.readCSVasFloat(filename)
                    n, d = the_sequence.shape
                    even_list = range(0, n, self.sample_rate)
                    num_frames = len(even_list)
                    the_sequence = np.array(the_sequence[even_list, :])
                    the_sequence = torch.from_numpy(the_sequence).float().cuda()

                    p3d = the_sequence
                    # self.p3d[(subj, action, subact)] = p3d.view(num_frames, -1).cpu().data.numpy()
                    self.p3d[key] = p3d.view(num_frames, -1).cpu().data.numpy()

                    valid_frames = np.arange(0, num_frames - seq_len + 1, skip_rate)

                    # tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    key += 1

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        pose = self.p3d[key][fs]
        observed = pose.copy() / 1000.0

        if self.miss_type == 'no_miss':
            mask = np.zeros((pose.shape[0], pose.shape[1]))
            mask[0:self.in_n, :] = 1
            mask[self.in_n:self.in_n + self.out_n, :] = 0
        elif self.miss_type == 'random':
            # Random Missing with Random Probability
            mask = np.zeros((self.in_n, pose.shape[1] // 3, 3))
            p_miss = np.random.uniform(0., 1., size=[self.in_n, pose.shape[1] // 3])
            p_miss_rand = np.random.uniform(0., 1.)
            mask[p_miss > p_miss_rand] = 1.0
            mask = mask.reshape((self.in_n, pose.shape[1]))
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        elif self.miss_type == 'random_joints':
            # Random Joint Missing
            mask = np.zeros((self.in_n, pose.shape[1]))
            p_miss = self.miss_rate * np.ones((pose.shape[1], 1))
            for i in range(0, pose.shape[1], 3):
                A = np.random.uniform(0., 1., size=[self.in_n, ])
                B = A > p_miss[i]
                mask[:, i] = 1. * B
                mask[:, i + 1] = 1. * B
                mask[:, i + 2] = 1. * B
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        elif self.miss_type == 'random_right_leg':
            # Right Leg Random Missing
            mask = np.ones((self.in_n, pose.shape[1]))
            rand = np.random.choice(self.in_n, size=math.floor(self.miss_rate * self.in_n), replace=False)
            right_leg = [1, 2, 3]
            for i in right_leg:
                mask[rand, 3 * i] = 0.
                mask[rand, 3 * i + 1] = 0.
                mask[rand, 3 * i + 2] = 0.
        elif self.miss_type == 'random_left_arm_right_leg':
            # Left Arm and Right Leg Random Missing
            mask = np.ones((self.in_n, pose.shape[1]))
            rand = np.random.choice(self.in_n, size=math.floor(self.miss_rate * self.in_n), replace=False)
            left_arm_right_leg = [1, 2, 3, 17, 18, 19]
            for i in left_arm_right_leg:
                mask[rand, 3 * i] = 0.
                mask[rand, 3 * i + 1] = 0.
                mask[rand, 3 * i + 2] = 0.
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        elif self.miss_type == 'structured_joint':
            # Structured Joint Missing (Continuous)
            mask = np.ones((self.in_n, pose.shape[1]))
            rand = np.random.choice(self.in_n - 10, size=1, replace=False)
            right_leg = [1, 2, 3]
            for i in right_leg:
                mask[rand[0]:rand[0] + 10, 3 * i] = 0.
                mask[rand[0]:rand[0] + 10, 3 * i + 1] = 0.
                mask[rand[0]:rand[0] + 10, 3 * i + 2] = 0.
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        elif self.miss_type == 'structured_frame':
            # Structured Frame Missing (Continuous)
            mask = np.ones((self.in_n, pose.shape[1]))
            rand = np.random.choice(self.in_n - 5, size=1, replace=False)
            mask[rand[0]:rand[0] + 5, ] = 0.
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        elif self.miss_type == 'random_frame':
            # Random Frame Missing
            mask = np.ones((self.in_n, pose.shape[1]))
            rand = np.random.choice(self.in_n, size=math.floor(self.miss_rate * self.in_n), replace=False)
            mask[rand, :] = 0.
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        elif self.miss_type == 'noisy_50':
            # Noisy Leg with Sigma=50
            mask = np.ones((self.in_n, pose.shape[1]))
            leg = [1, 2, 3, 6, 7, 8]
            sigma = 50
            noise = np.random.normal(0, sigma, size=observed.shape)
            noise[self.in_n:, :] = 0
            for i in range(0, self.in_n):
                missing_leg_joints = random.sample(leg, 3)
                for j in range(3):
                    mask[i, missing_leg_joints[j] * 3] = 0
                    mask[i, missing_leg_joints[j] * 3 + 1] = 0
                    mask[i, missing_leg_joints[j] * 3 + 2] = 0

                    noise[i, missing_leg_joints[j] * 3] = 0
                    noise[i, missing_leg_joints[j] * 3 + 1] = 0
                    noise[i, missing_leg_joints[j] * 3 + 2] = 0
            observed = (observed * 1000 + noise) / 1000
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        elif self.miss_type == 'noisy_25':
            # Noisy Leg with Sigma=25
            mask = np.ones((self.in_n, pose.shape[1]))
            leg = [1, 2, 3, 6, 7, 8]
            sigma = 25
            noise = np.random.normal(0, sigma, size=observed.shape)
            noise[self.in_n:, :] = 0
            for i in range(0, self.in_n):
                missing_leg_joints = random.sample(leg, 3)
                for j in range(3):
                    mask[i, missing_leg_joints[j] * 3] = 0
                    mask[i, missing_leg_joints[j] * 3 + 1] = 0
                    mask[i, missing_leg_joints[j] * 3 + 2] = 0

                    noise[i, missing_leg_joints[j] * 3] = 0
                    noise[i, missing_leg_joints[j] * 3 + 1] = 0
                    noise[i, missing_leg_joints[j] * 3 + 2] = 0
            observed = (observed * 1000 + noise) / 1000
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        else:
            mask = np.zeros((pose.shape[0], pose.shape[1]))
            mask[0:self.in_n, :] = 1
            mask[self.in_n:self.in_n + self.out_n, :] = 0

        data = {
            "pose": observed[:, :],
            "pose_32": pose,
            "mask": mask.copy()[:, :],
            "timepoints": np.arange(self.in_n + self.out_n)
        }

        return data

# if __name__ == '__main__':
#     input_n = 10
#     output_n = 25
#     skip_rate = 1
#     dataset = Datasets('../datasets/',input_n,output_n,skip_rate, split=2)
#     print('>>> Training dataset length: {:d}'.format(dataset.__len__()))