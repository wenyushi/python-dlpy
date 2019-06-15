#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

'''
Utility functions for the DLPy package by weshiz
The module contains lots of python third party dependence
'''

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import re
import six
import swat as sw
import string
import xml.etree.ElementTree as ET
from swat.cas.table import CASTable
from PIL import Image
import warnings
import platform
import collections
from itertools import repeat
import math
from scipy.misc import imresize
from itertools import permutations
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
import matplotlib.patheffects as PathEffects
from sklearn.cluster import KMeans


def scatter(x, labels, subtitle = None, color_scale=2):
    palette = np.array(sns.color_palette("hls", 10))
    f = plt.figure(figsize = (8, 8))
    ax = plt.subplot(aspect = 'equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw = 0, s = 40,
                    c = palette[labels.astype(np.int) * color_scale])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    if subtitle is not None:
        plt.suptitle(subtitle)
    plt.savefig(subtitle)