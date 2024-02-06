import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import datetime

from datasets import load_dataset, DatasetDict, Dataset, ClassLabel, Features

from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import array
import numbers
import warnings
from collections.abc import Iterable
from numbers import Integral, Real

import scipy.sparse as sp
from scipy import linalg

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils._param_validation import Hidden, Interval, StrOptions, validate_params
from sklearn.utils.random import sample_without_replacement

from collections import deque

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--level', type=int, default=1,
                        help='level')
    parser.add_argument('--label_flip', type=int, default=0,
                        help='label flip')   
    args = parser.parse_args()
    return args

random.seed(42)
np.random.seed(42)

def find_RHS_closest_value_on_dim(X, val, dim=0):
    # Calculate the difference between each element in the specified dimension and the target value
    differences = X[:, dim] - val

    # Get the indices where the differences are positive (on the right-hand side)
    rhs_indices = np.where(differences > 0)[0]

    if rhs_indices.size > 0:
        # If there are elements on the right-hand side, find the index of the closest one
        closest_index = rhs_indices[np.argmin(differences[rhs_indices])]
        closest_value = X[closest_index, dim]
        return closest_index, closest_value
    else:
        # If there are no elements on the right-hand side, return None
        return None

def find_eps_close_value_on_dim(X, val, dim=0):
    # Calculate the difference between each element in the specified dimension and the target value
    differences = X[:, dim] - val

    # Get the indices where the differences are positive (on the right-hand side)
    close_indices = np.where(torch.abs(differences) <= eps)[0]
    return close_indices


def threshold_to_data_dim(X, split_threshold, split_dimension, status):
    output = []
    for threshold, dimension, mask in zip(split_threshold, split_dimension, status):
        threshold_mat = np.zeros(X.shape[0])
        dim = dimension.nonzero()[0][0]
        val = threshold[dim]
        #X_ = X * (1 - mask.reshape(-1, 1)) * 1e8 + X
        threshold_mat = np.abs(X[:, dim] - val) # distance measure 
        #closest_index, closest_value = find_eps_close_value_on_dim(X_, val, dim=dim)
        #threshold_mat[closest_index] = 1.0
        output.append(threshold_mat)
    return output


def _find_parent(idx):
    return (idx-1) // 2

def _find_path(idx):
    # idx is the index of the leaf node
    # return the path from the leaf node to the root, excluding the root
    if idx == 0:
        return []
    return [idx] + _find_path((idx - 1)// 2)

def _get_decision(decision_lst, path):
    final_decision = np.ones_like(decision_lst[0])

    for idx in path:
        if idx % 2 == 1: #Last split = TRUE
            final_decision = final_decision * decision_lst[_find_parent(idx)]
        else: #Last split = FALSE
            final_decision = final_decision * (1 - decision_lst[_find_parent(idx)])
    return final_decision
    

def make_square_blobs(
    n_samples=100,
    n_features=2,
    *,
    centers=None,
    boxes=None,
    cluster_std=1.0,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=None,
    return_centers=False,
):

    generator = check_random_state(random_state)

    if isinstance(n_samples, numbers.Integral):
        # Set n_centers by looking at centers arg
        if centers is None:
            centers = 3

        if isinstance(centers, numbers.Integral):
            n_centers = centers
            centers = generator.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )

        else:
            centers = check_array(centers)
            n_features = centers.shape[1]
            n_centers = centers.shape[0]

    else:
        # Set n_centers by looking at [n_samples] arg
        n_centers = len(n_samples)
        if centers is None:
            centers = generator.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )
        if not isinstance(centers, Iterable):
            raise ValueError(
                "Parameter `centers` must be array-like. Got {!r} instead".format(
                    centers
                )
            )
        if len(centers) != n_centers:
            raise ValueError(
                "Length of `n_samples` not consistent with number of "
                f"centers. Got n_samples = {n_samples} and centers = {centers}"
            )
        centers = check_array(centers)
        n_features = centers.shape[1]

    # stds: if cluster_std is given as list, it must be consistent
    # with the n_centers
    if hasattr(cluster_std, "__len__") and len(cluster_std) != n_centers:
        raise ValueError(
            "Length of `clusters_std` not consistent with "
            "number of centers. Got centers = {} "
            "and cluster_std = {}".format(centers, cluster_std)
        )

    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.full(len(centers), cluster_std)

    if isinstance(n_samples, Iterable):
        n_samples_per_center = n_samples
    else:
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers

        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1

    cum_sum_n_samples = np.cumsum(n_samples_per_center)
    X = np.empty(shape=(sum(n_samples_per_center), n_features), dtype=np.float64)
    y = np.empty(shape=(sum(n_samples_per_center),), dtype=int)

    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        start_idx = cum_sum_n_samples[i - 1] if i > 0 else 0
        end_idx = cum_sum_n_samples[i]
        # This is the line that really matters
        x_low, y_low, x_high, y_high = boxes[i]
        low_array = np.array([x_low, y_low])
        high_array = np.array([x_high, y_high])
        X[start_idx:end_idx] = generator.uniform(
            low=low_array, high=high_array, size=(n, n_features)
        )
        y[start_idx:end_idx] = i

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if return_centers:
        return X, y, centers
    else:
        return X, y
    

def multi_level_xor(seq_len=512, level=1, low=-1, high=1, control_factor=1):

    def _generate_binary_tree(x_low, y_low, x_high, y_high, depth=0, split_dimension="x"):
        if depth == 0:
            if split_dimension == "x":
                return [np.array([(x_high + x_low) / 2, -100])]
            else:  # split_dimension == "y"
                return [np.array([-100, (y_high + y_low) / 2])]

        result = []
        max_depth = depth
        queue = deque([(x_low, y_low, x_high, y_high, depth, split_dimension)])

        rnd_lst = []

        while queue:
            x_low, y_low, x_high, y_high, depth, split_dimension = queue.popleft()

            if depth == 0:
                continue

            if split_dimension == "x":
                delta = (x_high - x_low) / 2
                if len(rnd_lst) < max_depth-depth+1:
                    rnd_lst.append(np.random.uniform(low=-delta, high=delta))
                x_mid = control_factor * (x_low + x_high) / 2 + (1 - control_factor) * rnd_lst[max_depth-depth]
                #x_mid = (x_low + x_high) / 2 + (1 - control_factor) * np.random.uniform(low=-delta, high=delta)
                #x_mid = (x_low + x_high) / 2
                result.append(np.array([x_mid, -100]))
                queue.append((x_mid, y_low, x_high, y_high, depth-1, "y"))
                queue.append((x_low, y_low, x_mid, y_high, depth-1, "y"))
                
            else:  # split_dimension == "y"
                delta = (y_high - y_low) / 2
                if len(rnd_lst) < max_depth-depth+1:
                    rnd_lst.append(np.random.uniform(low=-delta, high=delta))
                y_mid = control_factor * (y_low + y_high) / 2 + (1 - control_factor) * rnd_lst[max_depth-depth]
                #y_mid = (y_low + y_high) / 2 + (1 - control_factor) * np.random.uniform(low=-delta, high=delta)
                #y_mid = (y_low + y_high) / 2
                result.append(np.array([-100, y_mid]))
                queue.append((x_low, y_mid, x_high, y_high, depth-1, "x"))
                queue.append((x_low, y_low, x_high, y_mid, depth-1, "x"))
        return result


    def split_bounding_boxes(bounding_boxes, threshold):
        axis = 0 if threshold[1] == -100 else 1  # Get the axis of the split (0 for X-axis, 1 for Y-axis)
        new_bounding_boxes = []

        for bbox in bounding_boxes:
            x_min, y_min, x_max, y_max = bbox
            split_value = threshold[axis]

            if axis == 0:  # Split on X-axis
                # Check if the split value is within the current bounding box
                if x_min < split_value < x_max:
                    bbox_left = (x_min, y_min, min(x_max, split_value), y_max)
                    bbox_right = (max(x_min, split_value), y_min, x_max, y_max)
                    new_bounding_boxes.extend([bbox_left, bbox_right])
                else:
                    new_bounding_boxes.append(bbox)
            else:  # Split on Y-axis
                # Check if the split value is within the current bounding box
                if y_min < split_value < y_max:
                    bbox_left = (x_min, y_min, x_max, min(y_max, split_value))
                    bbox_right = (x_min, max(y_min, split_value), x_max, y_max)
                    new_bounding_boxes.extend([bbox_left, bbox_right])
                else:
                    new_bounding_boxes.append(bbox)
                    
        return new_bounding_boxes
    
    def _find_bounding_boxes(thresholds, low, high):
        bounding_boxes = [(low, low, high, high)]  # Initialize with a bounding box covering the entire plane
        for threshold in thresholds:
            bounding_boxes = split_bounding_boxes(bounding_boxes, threshold)
        return bounding_boxes

    def _sort_bounding_boxes_zigzag(bounding_boxes, rows, columns):
        sorted_bounding_boxes = []

        # Sort the bounding boxes in a zig-zag order for a grid with even rows and columns
        for row in range(rows):
            if row % 2 == 0:  # For even rows, sort from left to right
                sorted_bounding_boxes.extend(sorted(bounding_boxes[row * columns : (row + 1) * columns], key=lambda bbox: bbox[0]))
            else:  # For odd rows, sort from right to left
                sorted_bounding_boxes.extend(sorted(bounding_boxes[row * columns : (row + 1) * columns], key=lambda bbox: bbox[0], reverse=True))
        return sorted_bounding_boxes

    def _set_unifrom_seq_len(seq_len, bounding_boxes):
        seq_len_lst = []
        area_lst = []
        for bbox in bounding_boxes:
            area_lst.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        seq_len_lst = [int(seq_len * area / sum(area_lst)) for area in area_lst]
        seq_len_lst[-1] = seq_len - sum(seq_len_lst[:-1])
        return seq_len_lst


    total_partition = 2**(level)
    normalizer = 2**level
    centers = []
    lows, highs = [], []
    for row_partition in range(0, total_partition):
        if row_partition % 2 == 0:
            for column_partition in range(0, total_partition):
                centers.append([(low + row_partition*(high-low)/total_partition + low + (row_partition+1)*(high-low)/total_partition)/2, (low + (column_partition)*(high-low)/total_partition + low + (column_partition+1)*(high-low)/total_partition)/2])
        else:
            for column_partition in range(total_partition-1, -1, -1):
                centers.append([(low + row_partition*(high-low)/total_partition + low + (row_partition+1)*(high-low)/total_partition)/2, (low + (column_partition)*(high-low)/total_partition + low + (column_partition+1)*(high-low)/total_partition)/2])
    
    #X, y = make_square_blobs(n_samples = seq_len, n_features=2, centers=centers, cluster_std=(high-low)/total_partition/2)
    

    rtg = []
    status = []
    decision_lst = []
    split_threshold, split_dimension = [], []
    split_threshold = _generate_binary_tree(x_low=low, y_low=low, x_high=high, y_high=high, depth=((2*level)), split_dimension="x")

    #lows, highs = _get_lows_highs(lows, highs)
    boxes = _find_bounding_boxes(split_threshold, low, high)
    boxes = sorted(boxes, key=lambda bbox: bbox[1])
    boxes = _sort_bounding_boxes_zigzag(boxes, rows=2**level, columns=2**level)
    #boxes = sorted(boxes, key=lambda bbox: bbox[0])
    #boxes = sorted(boxes, key=lambda bbox: bbox[1])
    seq_len_lst = _set_unifrom_seq_len(seq_len, boxes)
    X, y = make_square_blobs(n_samples = seq_len_lst, n_features=2, centers=centers, boxes=boxes, cluster_std=(high-low)/total_partition/2, shuffle=True)
    
    y[(y % 2==0)] = 0
    y[(y % 2==1)] = 1

    for level_ctr in range(2*level):
        for node_ctr in range(2**level_ctr):
            status_mat = np.ones(X.shape[0])

            if level_ctr <= (2*level - 1):
                rtg.append(1.0)
            else:
                rtg.append(1.0 - node_ctr / 2**level_ctr)
            s_dimension = np.zeros(X.shape[1])
            if level_ctr % 2 == 0:
                #vertical split
                s_dimension[0] = 1
                dim = 0
            else:
                #horizontal split
                s_dimension[1] = 1
                dim = 1
            split_dimension.append(s_dimension)
            decision = (X[:, dim] < split_threshold[len(split_dimension)-1][dim]).astype(np.int8)
            decision_lst.append(decision)

            path = _find_path(len(split_dimension)-1)
            status_mat = _get_decision(decision_lst, path)
            status.append(status_mat)
    split_threshold_idx = threshold_to_data_dim(X, split_threshold, split_dimension, status)
    return X, y, rtg, status, split_threshold_idx, split_dimension, split_threshold


def generate_multi_xor(data_dimension, seq_len=1024, level=1, low=-1, high=1, control_factor=1):
    # Generate synthetic data
    X, y, rtg, status, split_threshold, split_dimension, split_threshold_val = multi_level_xor(seq_len=seq_len, level=level, low=low, high=high, control_factor=control_factor)
    X = np.concatenate((X, np.random.uniform(low=low, high=high, size=(seq_len,data_dimension-2))), axis=-1)
    # make sure the dimensions are correct after random permutation
    permutation = np.random.permutation(X.shape[1])
    X = np.take(X,permutation,axis=1)
    split_dimension = [np.take(np.concatenate((x, np.zeros(data_dimension-2))),permutation,axis=0) for x in split_dimension]
    #split_threshold = [np.take(np.concatenate((x, np.zeros(data_dimension-2))),permutation,axis=0) for x in split_threshold]

    X = X.astype(np.float32)
    split_dimension = [x.argmax(-1) for x in split_dimension]
    #split_threshold = [x.argmax(-1) for x in split_threshold]
    split_threshold = [x.astype(np.float32) for x in split_threshold]
    status = [x.astype(np.float32) for x in status]
    return X, y, [1.0], status, split_threshold, split_dimension, split_threshold_val


def process_data(data_size, data_dimension, seq_len=1024, level=1, control_factor=1, label_flip=0, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    for ctr in range(0, data_size):
        X, y, rtg, status, split_threshold, split_dimension, split_threshold_val = generate_multi_xor(data_dimension, seq_len, level=level, control_factor=control_factor)
        label_flip_mask = np.random.binomial(1, label_flip, size=y.shape)
        noisy_y = (y + label_flip_mask) % 2
        y_onehot = np.zeros((y.size, y.max() + 1))
        y_onehot[np.arange(y.size),y] = 1.0
        y_onehot = y_onehot.astype(np.float32)

        noisy_y_onehot = np.zeros((noisy_y.size, noisy_y.max() + 1))
        noisy_y_onehot[np.arange(noisy_y.size),noisy_y] = 1.0
        noisy_y_onehot = noisy_y_onehot.astype(np.float32)
        # Note: RTG Used purley as a placeholder for now
        data_dict = {"id": ctr, "input_x": X, "input_y": noisy_y_onehot, "input_y_clean": y_onehot,
                     "rtg": rtg, "status": status,
                     "split_threshold": split_threshold, "split_dimension": split_dimension}
        yield data_dict
    return 


if __name__ == "__main__":
    args = parse_args()
    data_size = 10000
    eval_size = 10000
    level = args.level #XOR LEVEL
    data_dimension = 10
    seq_len=256
    if level == 1:
        control_factor = 0.05
    elif level == 2:
        control_factor = 0.25
    #train_label_flip = 0.15
    label_flip = args.label_flip / 100
    train_label_flip = label_flip

    train_set = Dataset.from_generator(process_data, gen_kwargs={"data_size": data_size, "data_dimension": data_dimension, "seq_len": seq_len, "level": level, "control_factor": control_factor, "label_flip": train_label_flip, "random_seed": 42})
    validation_set = Dataset.from_generator(process_data, gen_kwargs={"data_size": eval_size, "data_dimension": data_dimension, "seq_len": seq_len, "level": level, "control_factor": control_factor, "label_flip": label_flip, "random_seed": 142})
    test_set = Dataset.from_generator(process_data, gen_kwargs={"data_size": eval_size, "data_dimension": data_dimension, "seq_len": seq_len, "level": level, "control_factor": control_factor, "label_flip": label_flip, "random_seed": 242})

    treedata = DatasetDict({'train': train_set, 'validation': validation_set, 'test': test_set})