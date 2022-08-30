from copy import deepcopy
import torch
import numpy as np
import os

class MouseKeypointSnippetDataset(torch.utils.data.Dataset):
    """Pytorch dataset of keypoint snippets.

        Each snippet is a slice from a single sequence, an array of shape
        (snippet_length, num_mice=3, num_keypoints=12, keypoint_dimensionality=2)
    """
    def __init__(self, sequence_keypoints_map, snippet_length=80):
        self.sequence_keypoints_map = sequence_keypoints_map
        self.snippet_length = snippet_length

        self.idx_keypoints_map = self._compute_idx_keypoints_map()

    def _compute_idx_keypoints_map(self):
        """Compute a mapping from index to (sequence_id, starting_index)."""

        end_index = -self.snippet_length + 1 if self.snippet_length > 1 else None
        idx_keypoints_map = []
        for sequence_id, sequence_info in self.sequence_keypoints_map.items():

            for i, _ in enumerate(sequence_info["keypoints"][:end_index]):
                idx_keypoints_map.append((sequence_id, i))
        
        return idx_keypoints_map

    def __len__(self):
        return len(self.idx_keypoints_map)

    def __getitem__(self, idx):
        sequence_id, starting_index = self.idx_keypoints_map[idx]
        sequence_keypoints = self.sequence_keypoints_map[sequence_id]["keypoints"]
        return sequence_keypoints[starting_index:starting_index + self.snippet_length]


def get_keypoints_map(data_dir, set='all', filled_holes=False):
    """
    _summary_

    Args:
        data_dir (_type_): _description_
        set (str, optional): _description_. Defaults to 'all'.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if filled_holes:
        submission_keypoints = np.load(os.path.join(data_dir, "submission_keypoints_filled_holes.npy"), allow_pickle=True).item()
        user_train_keypoints = np.load(os.path.join(data_dir, "train_keypoints_filled_holes.npy"), allow_pickle=True).item()
    else:
        submission_keypoints = np.load(os.path.join(data_dir, "submission_keypoints.npy"), allow_pickle=True).item()
        user_train_keypoints = np.load(os.path.join(data_dir, "user_train.npy"), allow_pickle=True).item()

    if set == 'all':
        sequence_keypoints_map = deepcopy(user_train_keypoints["sequences"])
        sequence_keypoints_map.update(submission_keypoints["sequences"])
    elif set == 'train':
        sequence_keypoints_map = deepcopy(user_train_keypoints["sequences"])
    elif set == 'submission':
        sequence_keypoints_map = deepcopy(submission_keypoints["sequences"])
    else:
        raise ValueError(f"Incorrect argument, set='{set}'. Valid opitions: ['all', 'train', 'submission']")
    return sequence_keypoints_map


DATA_DIR = '/data/behavior-representation'


if __name__=='__main__':

    user_train_keypoints = get_keypoints_map(DATA_DIR, "train")
    submission_keypoints = get_keypoints_map(DATA_DIR, "submission")
    sequence_keypoints_map = get_keypoints_map(DATA_DIR, "all")

    assert len(sequence_keypoints_map) == \
        len(user_train_keypoints) + len(submission_keypoints)

    keypoint_dataset = MouseKeypointSnippetDataset(sequence_keypoints_map, 4)
    print(keypoint_dataset[0])
    print(keypoint_dataset[19076])
    print(len(keypoint_dataset))
    print(keypoint_dataset[0].shape)
