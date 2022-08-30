import os
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as F

class VideoDataset(torch.utils.data.Dataset):
    """
    Reads frames from video files
    """
    def __init__(self, 
                 datafolder, 
                 frame_number_map,
                 channels_first,
                 start_idx=0,
                 end_idx=None):
        """
        Initializing the dataset with images and labels
        """
        self.datafolder = datafolder
        self.frame_number_map = frame_number_map
        self.channels_first = channels_first

        self._setup_frame_map()
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else self.length

    def _setup_frame_map(self):
        self._video_names = np.array(list(self.frame_number_map.keys()))
        # IMPORTANT: the frame number map should be sorted for self.get_video_name to work
        frame_nums = np.array([self.frame_number_map[k] for k in self._video_names])
        self._frame_numbers = frame_nums[:, 0] - 1 # start values
        assert np.all(np.diff(self._frame_numbers) > 0), "Frame number map is not sorted"

        self.length = frame_nums[-1, 1] # last value is the total number of frames

    def get_frame_info(self, global_index):
        """ Returns corresponding video name and frame number"""
        video_idx = np.searchsorted(self._frame_numbers, global_index) - 1
        frame_index = global_index - (self._frame_numbers[video_idx] + 1)
        return self._video_names[video_idx], frame_index
    
    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        idx = self.start_idx + idx
        video_name, frame_index = self.get_frame_info(idx)

        video_path = os.path.join(self.datafolder, video_name + '.avi')
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
       
        cap = cv2.VideoCapture(video_path)
        num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index < 0 or frame_index >= num_video_frames:
            raise ValueError(f"Frame {frame_index} not found in video {video_name}!")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        img_pil = Image.fromarray(frame)
        img_out = img_pil.convert('RGB')
        img_out = np.array(img_out)
        if self.channels_first:
            img_out = img_out.swapaxes(0,2)
        return img_out


class SimCLRDataset(VideoDataset):
    def __init__(self, 
                 datafolder, 
                 keypoints,
                 channels_first=False,
                 transform=None,
                 start_idx=0,
                 end_idx=None,
                 bounding_box_out=False):
        self.transform = transform
        self.keypoints = keypoints
        self.bounding_box_out = bounding_box_out
        frame_number_map = {video_id: (idx * 1800, (idx + 1) * 1800) for idx, video_id in enumerate(keypoints.keys())}
        super().__init__(datafolder, 
                 frame_number_map,
                 channels_first,
                 start_idx=start_idx,
                 end_idx=end_idx)
    
    def __getitem__(self, idx):
        idx = self.start_idx + idx
        video_name, frame_index = self.get_frame_info(idx)

        video_path = os.path.join(self.datafolder, video_name + '.avi')
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
       
        cap = cv2.VideoCapture(video_path)
        num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index < 0 or frame_index >= num_video_frames:
            raise ValueError(f"Frame {frame_index} not found in video {video_name}!")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index.item())
        success, frame = cap.read()

        img_pil = Image.fromarray(frame)
        
        # Get bounding box
        bbox_type = np.random.randint(-1, 3)
        if bbox_type == -1:
            bbox = self.keypoints[video_name]['bbox']
        else:
            bbox = self.keypoints[video_name][f'bbox_{bbox_type}']
        
        if self.transform is not None:
            try:
                img_tensor = self.transform(img_pil, bbox[frame_index])
            except IndexError as e:
                print(e)
                img_tensor = self.transform(img_pil, [0, 0, 512, 512])
        else:
            img_tensor = F.to_tensor(img_pil)
        if self.bounding_box_out:
            return {
                'img': img_tensor, 
                'bbox': self.keypoints[video_name]['bbox'][frame_index],
                'bbox_0': self.keypoints[video_name]['bbox_0'][frame_index],
                'bbox_1': self.keypoints[video_name]['bbox_1'][frame_index],
                'bbox_2': self.keypoints[video_name]['bbox_2'][frame_index]
                }
        return img_tensor


def create_bounding_box(data_dir):

    if not os.path.exists(os.path.join(data_dir, f'submission_bbox.npy')):
        ########## Prepare bounding boxes from keypoints ##########

        scaling_factor = 1

        # Preparing some bounding box information to be used for cropping frames during training
        keypoints = np.load(os.path.join(data_dir, f'submission_keypoints.npy'), allow_pickle=True).item()

        # Bounding Box for all mice
        padbbox = 0
        crop_size = 512
        for sk in tqdm(keypoints['sequences'].keys()):
            kp = keypoints['sequences'][sk]['keypoints']
            bboxes = []
            for frame_idx in range(len(kp)):
                allcoords = np.int32(kp[frame_idx].reshape(-1, 2))
                minvals = max(np.min(allcoords[:, 0]) - padbbox, 0), max(np.min(allcoords[:, 1]) - padbbox, 0)
                maxvals = min(np.max(allcoords[:, 0]) + padbbox, crop_size), min(np.max(allcoords[:, 1]) + padbbox, crop_size)
                bbox = (*minvals, *maxvals)
                bbox = np.array(bbox)
                bbox = np.int32(bbox * scaling_factor)
                bboxes.append(bbox)
            keypoints['sequences'][sk]['bbox'] = np.array(bboxes)

        # Bounding Box for one mouse at a time
        for i in range(3):
            padbbox = 0
            crop_size = 512
            for sk in tqdm(keypoints['sequences'].keys()):
                kp = keypoints['sequences'][sk]['keypoints']
                bboxes = []
                for frame_idx in range(len(kp)):
                    allcoords = kp[frame_idx, i]
                    minvals = max(np.min(allcoords[:, 0]) - padbbox, 0), max(np.min(allcoords[:, 1]) - padbbox, 0)
                    maxvals = min(np.max(allcoords[:, 0]) + padbbox, crop_size), min(np.max(allcoords[:, 1]) + padbbox, crop_size)
                    bbox = (*minvals, *maxvals)
                    bbox = np.array(bbox)
                    bbox = np.int32(bbox * scaling_factor)
                    bboxes.append(bbox)
                keypoints['sequences'][sk][f'bbox_{i}'] = np.array(bboxes)

        # Can save it you want and load later
        np.save(os.path.join(data_dir, f'submission_bbox.npy'), keypoints)
    else:
        keypoints = np.load(os.path.join(data_dir, f'submission_bbox.npy'), allow_pickle=True).item()
    
    return keypoints