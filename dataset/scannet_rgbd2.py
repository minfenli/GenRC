import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from glob import glob

class ScanNetRGBD2(Dataset):
    def __init__(self, root_path, scene_name, init_frames=0.1, z_offset=0, split='train', resize=True, axis_aligment=False):
        self.root_path = root_path
        self.scene_name = scene_name

        # because 240*180 images are not big enough for IronDepth, we resize input images.
        self.resize = resize

        self.axis_align = axis_aligment
        axis_align_path = os.path.join(self.root_path, self.scene_name, 'axis_alignment.txt')
        if self.axis_align and not os.path.isfile(axis_align_path):
            print("File is not found:", axis_align_path)
            print("So, only swap axis")
            self.axis_align = False

        self.all_indices = self.__get_all_indices()
        self.split = split
        self.num_init_frames, self.init_frame_indices = self.__get_num_init_frames(init_frames)
        if not axis_aligment:
            self.axis_swap_matrix = np.array([[1., 0., 0., 0.], 
                                    [0., 1., 0., 0.], 
                                    [0., 0., 1., 0.], 
                                    [0., 0., 0., 1.]]).astype('float32')
        else:
            self.axis_swap_matrix = np.array([[1., 0., 0., 0.], 
                                    [0., 0., 1., -z_offset], 
                                    [0., -1., 0., 0.], 
                                    [0., 0., 0., 1.]]).astype('float32')
        
    def __len__(self):
        return self.num_init_frames

    def __getitem__(self, idx):
        return self.__get_frame_data(idx)

    def get_all_w2cs(self, exclude_init_frames=False, init_frames=False):
        assert not (init_frames and exclude_init_frames)

        w2cs = []
        for frame_idx in self.all_indices:
            if init_frames and not frame_idx in self.init_frame_indices:
                    continue
            if exclude_init_frames and frame_idx in self.init_frame_indices:
                    continue
            w2cs.append(self.__get_frame_w2c(frame_idx))
        
        return w2cs

    def __get_frame_w2c(self, frame_idx):
        frame_idx = str(frame_idx)
        pose_path = os.path.join(self.root_path, self.scene_name, 'pose', f'{frame_idx}.txt')
        pose = np.loadtxt(pose_path).astype('float32') # c2w
        if self.axis_align:
            axis_align_path = os.path.join(self.root_path, self.scene_name, 'axis_alignment.txt')
            axis_align_matrix = np.loadtxt(axis_align_path).astype('float32')
            return np.linalg.inv(self.axis_swap_matrix@axis_align_matrix@pose)
        else:
            return np.linalg.inv(self.axis_swap_matrix@pose)

    def __get_frame_data(self, idx):
        frame_idx = str(self.init_frame_indices[idx])
        intrinsic_path = os.path.join(self.root_path, self.scene_name, 'intrinsic', 'intrinsic_depth.txt')
        depth_intrinsic = np.loadtxt(intrinsic_path).astype('float32')

        pose_path = os.path.join(self.root_path, self.scene_name, 'pose', f'{frame_idx}.txt')
        depth_path = os.path.join(self.root_path, self.scene_name, 'depth', f'{frame_idx}.png')
        color_path = os.path.join(self.root_path, self.scene_name, 'color', f'{frame_idx}.jpg')

        color_image = cv2.imread(color_path)[...,::-1]
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        if self.resize:
            color_image = cv2.resize(color_image, (640, 480), interpolation=cv2.INTER_AREA)

            # inpaint the black borders
            inpaint_mask = np.ones(color_image.shape[:2]).astype('uint8')
            inpaint_mask[6:-6, 6:-6] = 0
            color_image = cv2.inpaint(color_image, inpaint_mask, 3, cv2.INPAINT_TELEA)
        else:
            # inpaint the black borders
            inpaint_mask = np.ones(color_image.shape[:2]).astype('uint8')
            inpaint_mask[2:-2, 2:-2] = 0
            color_image = cv2.inpaint(color_image, inpaint_mask, 3, cv2.INPAINT_TELEA)

        depth_image = cv2.imread(depth_path, -1) # read 16bit grayscale image
        depth_shift = 1000.
        depth = depth_image.astype('float32')/depth_shift

        if self.resize:
            depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)

        pose = np.loadtxt(pose_path).astype('float32') # c2w

        instance_mask = np.zeros(depth.shape[:2], dtype='bool')

        if self.axis_align:
            axis_align_path = os.path.join(self.root_path, self.scene_name, 'axis_alignment.txt')
            axis_align_matrix = np.loadtxt(axis_align_path).astype('float32')
            return dict(image=color_image, depth=depth, instance_mask=instance_mask, K=depth_intrinsic, w2c=np.linalg.inv(self.axis_swap_matrix@axis_align_matrix@pose))
        else:
            return dict(image=color_image, depth=depth, instance_mask=instance_mask, K=depth_intrinsic, w2c=np.linalg.inv(self.axis_swap_matrix@pose))

    def __get_all_indices(self):
        file_paths = sorted(glob(os.path.join(self.root_path, self.scene_name, 'color', '*.jpg')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        indices = [int(file_path.split('/')[-1].split('.')[0]) for file_path in file_paths]
        return indices

    def __get_num_init_frames(self, init_frames):
        if isinstance(init_frames, float):
            num_init_frames = np.clip(init_frames, 0, 1)
            num_init_frames = int(np.ceil(num_init_frames*len(self.all_indices)))
            indices = np.linspace(0, len(self.all_indices)-1, num_init_frames).round().astype('int')
            frames_indices = [self.all_indices[idx] for idx in indices]
        elif isinstance(init_frames, list):
            num_init_frames = len(init_frames)
            frames_indices = [self.all_indices[idx] for idx in init_frames]
        else:
            raise ValueError("init_frames")
        
        if self.split != 'train':
            num_init_frames = len(self.all_indices) - num_init_frames
            _frames_indices = []
            for idx in self.all_indices:
                if idx not in frames_indices:
                    _frames_indices.append(idx)
            frames_indices = _frames_indices

        return num_init_frames, frames_indices
    
    def get_train_frames(self):
        if self.split == 'train':
            train_frame_indices = self.init_frame_indices
        else:
            train_frame_indices = []
            test_frame_indices = self.init_frame_indices
            for idx in self.all_indices:
                if idx not in test_frame_indices:
                    train_frame_indices.append(idx)

        images = []

        for idx in train_frame_indices:
            frame_idx = str(idx)
            color_path = os.path.join(self.root_path, self.scene_name, 'color', f'{frame_idx}.jpg')
            color_image = cv2.imread(color_path)[...,::-1]
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            if self.resize:
                color_image = cv2.resize(color_image, (640, 480), interpolation=cv2.INTER_AREA)
                # inpaint the black borders
                inpaint_mask = np.ones(color_image.shape[:2]).astype('uint8')
                inpaint_mask[6:-6, 6:-6] = 0
                color_image = cv2.inpaint(color_image, inpaint_mask, 3, cv2.INPAINT_TELEA)
            else:
                # inpaint the black borders
                inpaint_mask = np.ones(color_image.shape[:2]).astype('uint8')
                inpaint_mask[2:-2, 2:-2] = 0
                color_image = cv2.inpaint(color_image, inpaint_mask, 3, cv2.INPAINT_TELEA)
            
            images.append(color_image)
        
        return images
        

