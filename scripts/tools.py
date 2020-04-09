#
# Created on Wed Apr 08 2020
#
# Copyright (c) 2020 HITSZ-NRSL
# All rights reserved
#
# Author: EpsAvlc
#

#!/usr/bin/python3  
# -*- coding: utf-8 -*-
import numpy as np
import os
from pandas import read_csv

def load_segments(folder, filename="segments_database.csv"):
    # container
    segments = []

    # extract and store point data
    from pandas import read_csv

    file_path = os.path.join(folder, filename)
    extracted_data = read_csv(file_path, delimiter=" ").values

    segment_ids = extracted_data[:, 0].astype(int)
    duplicate_ids = extracted_data[:, 1].astype(int)
    points = extracted_data[:, 2:]

    #     complete_ids = list(zip(segment_ids, duplicate_ids))
    id_changes = []
    for i in range(len(segment_ids)):
        if i > 0 and (segment_ids[i] != (segment_ids[i - 1]) or duplicate_ids[i] != duplicate_ids[i - 1]):
            id_changes.append(i)
    #     complete_ids = list(zip(segment_ids, duplicate_ids))
    #     id_changes = []
    #     for i, complete_id in enumerate(complete_ids):
    #         if i > 0 and complete_id != complete_ids[i - 1]:
    #             id_changes.append(i)

    segments = np.split(points, id_changes)

    segment_ids = [ids[0] for ids in np.split(segment_ids, id_changes)]
    duplicate_ids = [ids[0] for ids in np.split(duplicate_ids, id_changes)]

    if len(set(zip(segment_ids, duplicate_ids))) != len(segment_ids):
        raise ValueError(
            "Id collision when importing segments. Two segments with same id exist in file."
        )

    print(
            "  Found "
            + str(len(segments))
            + " segments from "
            + str(np.unique(segment_ids).size)
            + " sequences"
    )
    return segments, segment_ids, duplicate_ids

def load_labels(folder, filename="labels_database.csv"):
    segment_ids = []
    labels = []

    file_path = os.path.join(folder, filename)
    if os.path.isfile(file_path):
        with open(file_path) as inputfile:
            for line in inputfile:
                split_line = line.strip().split(" ")

                segment_ids.append(int(split_line[0]))
                labels.append(int(split_line[1]))

    print("  Found labels for " + str(len(labels)) + " segment ids")
    return labels, segment_ids

def load_features(folder, filename="features_database.csv"):
    # containers
    segment_ids = []
    duplicate_ids = []
    features = []
    feature_names = []

    file_path = os.path.join(folder, filename)
    if filename:
        with open(file_path) as inputfile:
            for line in inputfile:
                split_line = line.strip().split(" ")

                # feature names
                if len(feature_names) == 0:
                    feature_names = split_line[2::2]

                # id
                segment_id = split_line[0]
                segment_ids.append(int(segment_id))
                duplicate_id = split_line[1]
                duplicate_ids.append(int(duplicate_id))

                # feature values
                features.append(np.array([float(i) for i in split_line[3::2]]))

    print("  Found features for " + str(len(features)) + " segments", end="")
    if "autoencoder_feature1" in feature_names:
        print("(incl. autoencoder features)", end="")
    print(" ")
    return features, feature_names, segment_ids, duplicate_ids

def load_positions(folder, filename="positions_database.csv"):
    segment_ids = []
    duplicate_ids = []
    positions = []

    file_path = os.path.join(folder, filename)
    if os.path.isfile(file_path):
        with open(file_path) as inputfile:
            for line in inputfile:
                split_line = line.strip().split(" ")

                segment_ids.append(int(split_line[0]))
                duplicate_ids.append(int(split_line[1]))

                segment_position = list(map(float, split_line[2:]))
                positions.append(segment_position)

    print("  Found positions for " + str(len(positions)) + " segments")
    return positions, segment_ids, duplicate_ids

def load_matches(folder, filename="matches_database.csv"):
    # containers
    matches = []

    file_path = os.path.join(folder, filename)

    if os.path.isfile(file_path):
        with open(file_path) as inputfile:
            for line in inputfile:
                split_line = line.strip().split(" ")
                matches.append([int(float(i)) for i in split_line if i != ""])

    print("  Found " + str(len(matches)) + " matches")
    return np.array(matches)

def load_merges(folder, filename="merge_events_database.csv"):
    merge_timestamps = []
    merges = []

    file_path = os.path.join(folder, filename)
    with open(file_path) as inputfile:
        for line in inputfile:
            split_line = line.strip().split(" ")
            merge_timestamps.append(int(split_line[0]))
            merges.append(list(map(int, split_line[1:])))

    print("  Found " + str(len(merges)) + " merge events")
    return merges, merge_timestamps