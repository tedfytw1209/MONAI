# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Tien
Implement average precision, average recall, IOU, mean IOU for objection detection task
This script is almost same with 
The changes include 1) code reformatting, 2) docstrings.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

from monai.metrics.utils import ignore_background
from monai.utils import MetricReduction

import logging as logger
from collections.abc import Sequence
from typing import Any

import numpy as np


class OBJDetectMetric():

    def __init__(
        self,
        classes: Sequence[str]
    ):
        """
        Class to compute required Obj detection metrics
        Metrics computed includes, (At what IoU threshold?)
        -average precision: the area under the PR curve (already in coco.py)
        -average recall (already in coco.py)
        -IOU: Intersection / Union of matched points?
        -mean IOU ?

        Args:
            classes (Sequence[str]): name of each class (index needs to correspond to predicted class indices!)
            iou_list (Sequence[float]): specific thresholds where ap is evaluated and saved
            iou_range (Sequence[float]): (start, stop, step) for mAP iou thresholds
            max_detection (Sequence[int]): maximum number of detections per image
            verbose (bool): log time needed for evaluation
        """
        pass