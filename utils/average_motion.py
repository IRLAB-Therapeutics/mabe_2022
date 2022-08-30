import numpy as np
from numpy.linalg import norm

from utils.keypoint_dataset import DATA_DIR, MouseKeypointSnippetDataset, get_keypoints_map

keypoints_map = get_keypoints_map(DATA_DIR, "submission", filled_holes=True)
keypoint_dataset = MouseKeypointSnippetDataset(keypoints_map, 1800)

v_arr = []
for i, seq in enumerate(keypoint_dataset):
    v = np.gradient(seq[:,:,:10], axis=0)
    v = norm(v, axis=3)
    v = np.sum(v, axis=2)
    v = np.max(v, axis=1)
    v_arr.append(v)
v_arr = np.concatenate(v_arr)

np.save("cache/average_motion.npy", v_arr)