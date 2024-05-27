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

from .metric import Metric
import logging as logger
from collections.abc import Sequence
from typing import Any

import numpy as np


class OBJDetectMetric(Metric):

    def __init__(
        self,
        classes: Sequence[str]
    ):
        pass