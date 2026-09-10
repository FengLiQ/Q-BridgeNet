import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
import random
import numpy as np
import copy
import pickle
import json
import pathlib
import gzip
from config import POSE_LIST, FACE_LIST, LEFT_LIST, RIGHT_LIST, MAX_SIGN_LEN, MAX_CLIP_LEN, DEVICE


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


# load sub-pose
def load_part_kp(skeletons):
    kp2d_ori_body = []
    kp2d_ori_left = []
    kp2d_ori_right = []
    kp2d_ori_face = []
    for skeleton in skeletons:
        skeleton = skeleton[0]
        kp2d_ori_body.append(skeleton[POSE_LIST, :])
        kp2d_ori_face.append(skeleton[FACE_LIST, :])
        kp2d_ori_left.append(skeleton[LEFT_LIST, :])
        kp2d_ori_right.append(skeleton[RIGHT_LIST, :])

    kp2d_ori_body = torch.tensor(np.stack(kp2d_ori_body))
    kp2d_ori_face = torch.tensor(np.stack(kp2d_ori_face))
    kp2d_ori_left = torch.tensor(np.stack(kp2d_ori_left))
    kp2d_ori_right = torch.tensor(np.stack(kp2d_ori_right))

    kps_ori = torch.cat([kp2d_ori_body, kp2d_ori_face, kp2d_ori_left, kp2d_ori_right], dim=1).float().clip(min=0, max=1)
    return kps_ori


class S2T_Dataset(Dataset):
    def __init__(self, label_path, pose_dirs, online_load=False):
        super(S2T_Dataset, self).__init__()
        self.max_length = MAX_SIGN_LEN
        self.raw_data = load_dataset_file(label_path)
        self.pose_dir = pose_dirs
        self.list = list(self.raw_data.keys())

        self.pose_ori = {}
        self.seg_data = {}
        self.ex_text = {}

        if online_load:
            print("online load, no data loaded here")
        else:
            print("offline load")
            for idx, item in enumerate(self.list):
                if idx % 5000 == 0:
                    print(str(idx) + "data loaded")
                sample = self.raw_data[item]
                self.pose_ori[item] = self.load_pose(sample['video_path'])
        print(self)

    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]

        text = sample['text']
        if "gloss" in sample.keys():
            gloss = " ".join(sample['gloss'])
        else:
            gloss = ''
        
        name_sample = sample['name']
        if key in self.pose_ori:
            pose_ori = self.pose_ori[key]
        else:
            pose_ori = self.load_pose(sample['video_path'])

        return name_sample, pose_ori, text, gloss

    def load_seg(self, seg_path):
        self.seg_data = load_dataset_file(seg_path)

    def ini_seg(self):
        seg = {}
        for item in self:
            name = item[0]
            sign_item = item[1]
            start = 0
            length = len(sign_item)
            seg[name] = []
            while start < length:
                if length - start > MAX_CLIP_LEN:
                    k = random.randint(1, MAX_CLIP_LEN)
                else:
                    k = length - start
                seg[name].append(k)
                start += k
        self.seg_data = seg

    def get_seg(self, key):
        seg = []
        if key in self.seg_data:
            seg = self.seg_data[key]
        return seg

    def load_ex_text(self, path):
        self.ex_text = load_dataset_file(path)
        self.list = [key for key in self.ex_text if key in self.raw_data]
        print(self)

    def get_ex_text(self, key):
        if key in self.ex_text:
            return self.ex_text[key]
        return None
    
    def load_pose(self, path):
        pose = pickle.load(open(os.path.join(self.pose_dir, path.replace(".mp4", '.pkl')), 'rb'))
            
        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0
                
        if duration > self.max_length:
            tmp = list(range(self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp) + start
            
        skeletons = pose['keypoints']
        skeletons_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])

        skeletons = skeletons_tmp
        kps_ori = load_part_kp(skeletons)
        return kps_ori

    def __str__(self):
        return f'#total {len(self)}'


class CLIP_Dataset4(Dataset):
    def __init__(self, feat, length_idx, aug_times=0):
        self.feat = []
        for i in range(1, 5):
            for feat_data in feat[length_idx * 4 + i]:
                ext_feat = feat_data[-1].unsqueeze(0).repeat(4 - i, 1)
                self.feat.append(torch.cat((feat_data, ext_feat), dim=0))
                for k in range(aug_times):
                    aug = (torch.randn_like(feat_data) * 0.0015 + feat_data).clip(0, 1)
                    self.feat.append(torch.cat((aug, ext_feat), dim=0))

        self.feat = torch.stack(self.feat).to(DEVICE)

    def __getitem__(self, index):
        return self.feat[index], index

    def __len__(self):
        return len(self.feat)
