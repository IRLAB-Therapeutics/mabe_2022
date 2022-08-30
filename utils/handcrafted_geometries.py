import os
import numpy as np
from numpy.linalg import norm

from tqdm import tqdm
from utils.find_cage_features import find_corners, get_feeding_location

from utils.keypoint_dataset import DATA_DIR, MouseKeypointSnippetDataset, get_keypoints_map

import matplotlib.pyplot as plt

from utils.video_dataset import VideoDataset

NUM_MICE = 3
BODY_KEYPOINTS= (0, 9)
MAX_SIZE = 512
CLIP_LENGTH = 1800

BP = { 
    'nose': 0, 
    'left_ear': 1,  
    'right_ear': 2,  
    'neck': 3, 
    'left_forepaw': 4, 
    'right_forepaw': 5, 
    'center_back': 6, 
    'left_hindpaw': 7, 
    'right_hindpaw': 8, 
    'tail_base': 9, 
    'tail_middle': 10, 
    'tail_tip': 11
}

# ############### Helper functions ###############

def polygon_area(m):
    x, y = m[:, :, :, 0], m[:, :, :, 1]
    correction = x[:, :, -1] * y[:, :, 0] - y[:, :, -1]* x[:, :, 0]
    main_area = np.sum(x[:, :, :-1] * y[:, :, 1:], axis=-1) - np.sum(y[:, :, :-1] * x[:, :, 1:], axis=-1)
    return 0.5*np.abs(main_area + correction)


def encode_angle(angle):
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def normalize_mouse(sequence):
    sample = sequence
    
    m_vector = sample[:, :, 3, :] - sample[:, :, 9, :]
    m_unit_vector = m_vector / norm(m_vector, axis=-1)[..., np.newaxis]

    basis = np.stack([m_unit_vector, np.stack([-m_unit_vector[:, :, 1], m_unit_vector[:, :, 0]], axis=-1)], axis=-1)
    pos = sample[:, :, :, :] - (sample[:, :, 3, :][:, :, np.newaxis, :] + sample[:, :, 9, :][:, :, np.newaxis, :])/2

    out = np.zeros_like(sample)
    for i in range(sample.shape[0]):
        for j in range(3):
            out[i, j] = np.dot(pos[i, j], basis[i, j])
    return out


def replace_nans_with_feature_mean(array):
    col_mean = np.nanmean(array, axis=0)
    print("Mean of each feature:")
    print(col_mean)

    inds = np.where(np.isnan(array))

    array[inds] = np.take(col_mean, inds[1])
    return array


# ############### Individual mouse features ###############

class DistanceFeature:
    def __init__(self, keypoints, bp1, bp2):
        self.bp1, self.bp2 = bp1, bp2
        dist = norm(keypoints[:, :, self.bp1, :] - keypoints[:, :, self.bp2, :], axis=-1)
        self.mean, self.std = dist.mean(), dist.std()

    def plot_distribution(self, keypoints):
        dist = norm(keypoints[:, :, self.bp1, :] - keypoints[:, :, self.bp2, :], axis=-1)
        plt.hist(dist.flatten(), range=[0, 100], bins=50)
        plt.show()

    def __call__(self, sequence):
        dist = norm(sequence[:, :, self.bp1, :] - sequence[:, :, self.bp2, :], axis=-1)
        return (dist - self.mean) / self.std


class BodyLengthFeature(DistanceFeature):
    def __init__(self, keypoints):
        super().__init__(keypoints, BP['nose'], BP['tail_base'])


class PawHeadDistanceFeature:
    def __init__(self, keypoints):
        dist = self._calc_distance(keypoints)
        self.mean, self.std = dist.mean(), dist.std()
    
    def _calc_distance(self, kps):
        head_center = np.array([kps[:, :, BP['left_ear'], :], kps[:, :, BP['right_ear'], :], kps[:, :, BP['nose'], :]]).mean(0)
        dist_left = norm(kps[:, :, BP['left_forepaw'], :] - head_center, axis=-1)
        dist_right = norm(kps[:, :, BP['left_forepaw'], :] - head_center, axis=-1)
        dist = np.min(np.array([dist_left, dist_right]), axis=0)
        return dist

    def plot_distribution(self, keypoints):
        dist = self._calc_distance(keypoints)
        plt.hist(dist.flatten(), range=[0, 20], bins=50)
        plt.show()

    def __call__(self, sequence):
        dist = self._calc_distance(sequence)
        return (dist - self.mean) / self.std


class PawNoseDistanceFeature:
    def __init__(self, keypoints):
        dist = self._calc_distance(keypoints)
        self.mean, self.std = dist.mean(), dist.std()
    
    def _calc_distance(self, kps):
        dist_left = norm(kps[:, :, BP['left_forepaw'], :] - kps[:, :, BP['nose'], :], axis=-1)
        dist_right = norm(kps[:, :, BP['left_forepaw'], :] - kps[:, :, BP['nose'], :], axis=-1)
        dist = np.min(np.array([dist_left, dist_right]), axis=0)
        return dist

    def plot_distribution(self, keypoints):
        dist = self._calc_distance(keypoints)
        plt.hist(dist.flatten(), range=[0, 40], bins=50)
        plt.show()

    def __call__(self, sequence):
        dist = self._calc_distance(sequence)
        return (dist - self.mean) / self.std


class AreaFeature:
    def __init__(self, keypoints, surface_keypoints):
        self.surface_keypoints = surface_keypoints
        area = polygon_area(keypoints[:, :, self.surface_keypoints, :])
        self.mean, self.std = area.mean(), area.std()

    def plot_distribution(self, keypoints):
        area = polygon_area(keypoints[:, :, self.surface_keypoints, :])
        plt.hist(area.flatten(), range=[0, 200], bins=50)
        plt.show()

    def __call__(self, sequence):
        area = polygon_area(sequence[:, :, self.surface_keypoints, :])
        return (area - self.mean) / self.std


class BodyAreaFeature(AreaFeature):
    def __init__(self, keypoints):
        super().__init__(keypoints=keypoints, surface_keypoints=(BP['left_forepaw'], BP['right_forepaw'], BP['left_hindpaw'], BP['right_hindpaw']))


class HeadAreaFeature(AreaFeature):
    def __init__(self, keypoints):
        super().__init__(keypoints=keypoints, surface_keypoints=(BP['nose'], BP['right_ear'], BP['left_ear']))


class SpeedFeature:
    def __init__(self):
        pass
    def __call__(self, sequence):
        center = sequence.mean(2)
        speed = norm(np.gradient(center, axis=0), axis=-1)
        return speed


class InternalAngleFeature:
    def __init__(self, keypoints):
        angle = self._calculate_angle(keypoints)
        self.mean, self.std = angle.mean(), angle.std()

    def _calculate_angle(self, kps):
        neck_nose = kps[:, :, BP['nose'], :] - kps[:, :, BP['neck'], :]
        neck_tailbase = kps[:, :, BP['neck'], :] - kps[:, :, BP['tail_base'], :]
        dot = np.einsum('bmi,bmi->bm', neck_nose, neck_tailbase).astype(np.float32)
        vector_norm = (norm(neck_nose, axis=-1) * norm(neck_tailbase, axis=-1))
        angle = np.arccos(np.divide(dot, vector_norm, out=np.zeros_like(dot), where=vector_norm!=0))
        return angle

    def plot_distribution(self, keypoints):
        angle = self._calculate_angle(keypoints)
        plt.hist(angle.flatten(), range=[0, np.pi], bins=50)
        plt.show()

    def __call__(self, sequence):
        angle = self._calculate_angle(sequence)
        return encode_angle((angle - self.mean) / self.std)


class DirectionChangeFeature:
    def __call__(self, sequence) -> np.array:
        heading = self._angle(sequence[:, :, BP['nose']], sequence[:, :, BP['center_back']])
        heading_change = np.gradient(heading, 1)[0]
        encoded_heading_change = encode_angle(heading_change)
        return encoded_heading_change.reshape(-1, 6)

    def plot_distribution(self, keypoints):
        heading = self._angle(keypoints[:, :, BP['nose']], keypoints[:, :, BP['center_back']])
        heading_change = np.gradient(heading)
        plt.hist(np.array(heading_change).flatten(), range=[-np.pi/10, np.pi/10], bins=50)
        plt.show()

    def _angle(self, seq1, seq2):
        return ((np.arctan2(seq1[..., 0] - seq2[..., 0], seq1[..., 1] - seq2[..., 1]) + np.pi / 2) % (np.pi * 2))


def get_mouse_features(kps, features_objects):
    [body_length, head_paw_distance, nose_paw_distance, body_area, head_area, internal_angle, direction_change, speed_feature] = features_objects

    feature_array = []
    feature_array.append(body_length(kps))
    feature_array.append(head_paw_distance(kps))
    feature_array.append(nose_paw_distance(kps))
    feature_array.append(body_area(kps))
    feature_array.append(head_area(kps))
    feature_array.append(internal_angle(kps).reshape(-1, 6))
    feature_array.append(direction_change(kps))
    feature_array.append(speed_feature(kps))
    feature_array = np.hstack(feature_array).astype(np.float32)

    return feature_array

# Environment
def get_mouse_cage_features(m_body, corners, feeding_location):
    front_feeding_distance = norm(m_body[0] - feeding_location)
    min_corner_distance = np.inf
    for point in m_body:
        for corner in corners:
            corner_distance = norm(corner - point)
            min_corner_distance = min(min_corner_distance, corner_distance)
    mouse_length = norm(m_body[0] - m_body[1])

    return [front_feeding_distance, min_corner_distance, mouse_length]


def get_mice_cage_features(kps, corners):
    feeding_location = get_feeding_location(*corners[:2])

    features = []
    for m0 in range(NUM_MICE):
        m_body = kps[m0, BODY_KEYPOINTS]
        mouse_cage_features = get_mouse_cage_features(m_body, corners, feeding_location)
        features.append(mouse_cage_features)
    features = np.array(features)
    min_features = features.min(axis=0)
    max_features = features.max(axis=0)

    return list(min_features) + list(max_features)

# ############### Group features ###############

class TriangleAreaFeature:
    def __init__(self, keypoints) -> None:
        self.max = self.get_stats(keypoints)

    def get_stats(self, keypoints):
        centers = keypoints[:, :, BP['center_back']]
        m1 = centers[:, 0]
        m2 = centers[:, 1]
        m3 = centers[:, 2]

        m1_m2 = m2 - m1
        m1_m3 = m3 - m1
        areas = np.abs(0.5 * (m1_m2[:, 0] * m1_m3[:, 1] - m1_m3[:, 0] * m1_m2[:, 1]))
        return areas.max()

    def __call__(self, sequence):
        centers = sequence[:, :, BP['center_back']]
        m1 = centers[:, 0]
        m2 = centers[:, 1]
        m3 = centers[:, 2]

        m1_m2 = m2 - m1
        m1_m3 = m3 - m1
        return np.abs(0.5 * (m1_m2[:, 0] * m1_m3[:, 1] - m1_m3[:, 0] * m1_m2[:, 1])) / self.max


class TriangleAngleFeatures:
    def __init__(self) -> None:
        pass

    def __call__(self, sequence):
        centers = sequence[:, :, BP['center_back']]
        m1 = centers[:, 0]
        m2 = centers[:, 1]
        m3 = centers[:, 2]

        m1_m2 = m2 - m1
        m1_m3 = m3 - m1
        m2_m3 = m3 - m2

        dot = np.einsum('bi,bi->b', m1_m2, m1_m3).astype(np.float32)
        vector_norm = (norm(m1_m2, axis=-1) * norm(m1_m3, axis=-1))
        angle_1 = np.arccos(np.divide(dot, vector_norm, out=np.zeros_like(dot), where=vector_norm!=0))

        dot = np.einsum('bi,bi->b', m1_m2, m2_m3).astype(np.float32)
        vector_norm = (norm(m1_m2, axis=-1) * norm(m1_m3, axis=-1))
        angle_2 = np.arccos(np.divide(dot, vector_norm, out=np.zeros_like(dot), where=vector_norm!=0))

        dot = np.einsum('bi,bi->b', m1_m3, m2_m3).astype(np.float32)
        vector_norm = (norm(m1_m2, axis=-1) * norm(m1_m3, axis=-1))
        angle_3 = np.arccos(np.divide(dot, vector_norm, out=np.zeros_like(dot), where=vector_norm!=0))
        
        return np.stack((encode_angle(np.stack((angle_1, angle_2, angle_3), axis=-1).min(1)), encode_angle(np.stack((angle_1, angle_2, angle_3), axis=-1).max(1))), axis=-1).reshape(-1, 4)


def get_group_features(kps, group_features_objects):
    triangle_area, triangle_angle = group_features_objects

    feature_array = []
    feature_array.append(triangle_area(kps).reshape(-1, 1))
    feature_array.append(triangle_angle(kps))

    feature_array = np.hstack(feature_array).astype(np.float32)
    return feature_array

# ############### Mouse pair features ###############

def get_mouse_mouse_features(m0_body, m1_body):
    m0_vector = m0_body[0] - m0_body[1]
    m0_unit_vector = m0_vector / norm(m0_vector)
    m1_vector = m1_body[0] - m1_body[1]
    m1_unit_vector = m1_vector / norm(m1_vector)

    features = []

    front_front_distance = norm(m1_body[0] - m0_body[0])
    front_rear_distance = norm(m1_body[0] - m0_body[1])
    rear_front_distance = norm(m1_body[1] - m0_body[0])
    rear_rear_distance = norm(m1_body[1] - m0_body[1])
    features.extend([front_front_distance, front_rear_distance, rear_front_distance, rear_rear_distance])

    orientation_angle = np.arccos(m0_unit_vector.T @ m1_unit_vector)
    front_front_unit_vector = (m1_body[0] - m0_body[0]) / norm(m1_body[0] - m0_body[0])
    position_angle = np.arccos(m0_unit_vector.T @ front_front_unit_vector)
    oa_cos, oa_sin = encode_angle(orientation_angle)
    pa_cos, pa_sin = encode_angle(position_angle)
    features.extend([oa_cos, oa_sin, pa_cos, pa_sin])
    
    return features


def get_mice_mice_features(kps):
    features = []
    for m0 in range(NUM_MICE):
        m0_body = kps[m0, BODY_KEYPOINTS] / MAX_SIZE
        for m1 in range(m0+1, NUM_MICE):
            m1_body = kps[m1, BODY_KEYPOINTS] / MAX_SIZE
            mouse_mouse_features_1 = get_mouse_mouse_features(m0_body, m1_body)
            mouse_mouse_features_2 = get_mouse_mouse_features(m1_body, m0_body)
            features.append(mouse_mouse_features_1)
            features.append(mouse_mouse_features_2)
    features = np.array(features)
    min_features = features.min(axis=0)
    max_features = features.max(axis=0)

    return list(np.sort(features, axis=0).flatten())#list(min_features) + list(max_features)


def get_static_features(kps, corners):
    mice_mice_features = get_mice_mice_features(kps)
    mice_cage_features = get_mice_cage_features(kps, corners)
    
    return np.array(mice_mice_features + mice_cage_features)


if __name__ == "__main__":
    keypoints_map = get_keypoints_map(DATA_DIR, "submission", filled_holes=True)
    keypoint_dataset = MouseKeypointSnippetDataset(keypoints_map, CLIP_LENGTH)

    frame_number_map = np.load(os.path.join(DATA_DIR, 'frame_number_map.npy'), allow_pickle=True).item()
    video_size = 'full_size'
    video_set = 'submission'
    video_data_dir = os.path.join(DATA_DIR, 'videos', video_size, video_set)

    dataset = VideoDataset(video_data_dir, frame_number_map, channels_first=False)

    all_keypoints = []
    for _, values in keypoints_map.items():
        all_keypoints.append(values['keypoints'])
    all_keypoints = np.vstack(all_keypoints)

    feature_array = []

    # Individual
    body_length = BodyAreaFeature(all_keypoints)
    head_paw_distance = PawHeadDistanceFeature(all_keypoints)
    nose_paw_distance = PawNoseDistanceFeature(all_keypoints)
    body_area = BodyAreaFeature(all_keypoints)
    head_area = HeadAreaFeature(all_keypoints)
    internal_angle = InternalAngleFeature(all_keypoints)
    direction_change = DirectionChangeFeature()
    speed_feature = SpeedFeature()

    # Group
    triangle_area = TriangleAreaFeature(all_keypoints)
    triangle_angle = TriangleAngleFeatures()

    features_objects = [body_length, head_paw_distance, nose_paw_distance, body_area, head_area, internal_angle, direction_change, speed_feature]
    group_features_objects = [triangle_area, triangle_angle]

    for i, keypoint_clip in enumerate(tqdm(keypoint_dataset)):
        keypoint_clip = keypoint_clip.astype(np.int64)
        frame = dataset[i * CLIP_LENGTH]
        corners = find_corners(frame)
        clip_features = []
        for kps in keypoint_clip:
            static_features = get_static_features(kps, corners)
            clip_features.append(static_features)
        
        individual_mouse_features = get_mouse_features(keypoint_clip, features_objects)
        group_mouse_features = get_group_features(keypoint_clip, group_features_objects)

        clip_features = np.array(clip_features)
        clip_features = np.concatenate((clip_features, individual_mouse_features), axis=1)
        clip_features = np.concatenate((clip_features, group_mouse_features), axis=1)

        # Dynamic features through gradient of keypoints and clip features
        dynamic_features_5 = np.gradient(clip_features, 5, axis=0)
        dynamic_features_kps_5 = norm(np.gradient(keypoint_clip, 5, axis=0), axis=-1).reshape(-1, 12*3)

        clip_features = np.concatenate((clip_features, dynamic_features_5, dynamic_features_kps_5), axis=1)

        feature_array.extend(clip_features)

    feature_array = np.array(feature_array, dtype=np.float32)

    feature_array = replace_nans_with_feature_mean(feature_array)
    
    np.save('cache/handcrafted_features.npy', feature_array)
