import os
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms.functional as Fvt
from torch.utils.data import Dataset
from utils.data import read_flo



class KITTI_Train(Dataset):
    def __init__(self, data_root, flow_root, resolution):
        self.data_root = data_root
        self.flow_root = flow_root
        self.resolution = resolution # (h, w)
        self.data_dir = [data_root]

        self.frames = []
        days = [x for x in os.listdir(data_root) if x.startswith('2011')]
        scenes = []
        for day in days:
            scenes += [x for x in os.listdir(os.path.join(data_root, '{}'.format(day))) if x.startswith('2011')]
        for scene in scenes:
            #ignore calibration images
            if scene in ['2011_09_26_drive_0119_sync',
                        '2011_09_28_drive_0225_sync',
                        '2011_09_29_drive_0108_sync',
                        '2011_09_30_drive_0072_sync',
                        '2011_10_03_drive_0058_sync']:
                continue
            day = scene[:10]
            l_video_frames = sorted(os.listdir(os.path.join(data_root, day, scene, 'image_02', 'data')))[:-1]
            r_video_frames = sorted(os.listdir(os.path.join(data_root, day, scene, 'image_03', 'data')))[:-1]
            
            assert l_video_frames == r_video_frames

            for video_frame in l_video_frames:
                self.frames.append('{} {} 02'.format(scene, video_frame[:-4]))
                self.frames.append('{} {} 03'.format(scene, video_frame[:-4]))

            assert l_video_frames == r_video_frames
                
        
        self.len_dataset = len(self.frames)
        print(f"Loaded {self.len_dataset} frames.")

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        dataset_dict = {}

        frame = self.frames[index]
        scene, f_id, side = frame.split(' ')
        
        rgb = self.get_rgb(scene, f_id, side)
        dataset_dict['rgb'] = rgb
        
        flow = self.get_flow(scene, f_id, side)
        dataset_dict['flow'] = flow
        dataset_dict['sem_seg'] = self.get_sem_seg(scene, f_id, side)

        dataset_dict['category'] = scene
        dataset_dict['frame_id'] = torch.tensor(int(f_id))
        dataset_dict['gap'] = 'gap1'
        dataset_dict['height'] = torch.tensor(self.resolution[0])
        dataset_dict['width'] = torch.tensor(self.resolution[1])
        
        
        return [dataset_dict]

    def get_rgb(self, scene, f_id, side):
        day = scene[:10]
        rgb_path = os.path.join(self.data_root, day, scene, 'image_{}'.format(side), 'data', '{}.png'.format(f_id))
        with open(rgb_path, 'rb') as f:
            with Image.open(f) as img:
                rgb = img.convert('RGB')
        rgb = Fvt.pil_to_tensor(rgb)
        return Fvt.resize(rgb, self.resolution, interpolation=Fvt.InterpolationMode.BICUBIC).to(torch.float32)

    def get_flow(self, scene, f_id, side):
        day = scene[:10]
        flow_path = os.path.join(self.flow_root, day, scene, 'image_{}'.format(side), 'data', '{}.png'.format(f_id))
        flow = cv2.imread(flow_path, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
        flow = flow[:,:,::-1]
        flow = flow[:, :, :2].astype(np.float32)
        flow = (flow - 2**15) / 64.0
        
        h, w, _ = flow.shape
        if (h, w) != self.resolution:
            flow = cv2.resize(flow, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_NEAREST)
            flow[:, :, 0] = flow[:, :, 0] * self.resolution[1] / w
            flow[:, :, 1] = flow[:, :, 1] * self.resolution[0] / h
        
        return torch.from_numpy(flow).permute(2, 0, 1)


    def get_sem_seg(self, scene, f_id, side):
        return torch.from_numpy(np.zeros((self.resolution))).to(torch.int64)



