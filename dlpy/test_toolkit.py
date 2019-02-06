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

''' Test Toolkit functions for the DLPy package '''

import sys
import numpy as np


def check_monotonicity(history, columns, tolerance=0, decreasing=True):
    '''

    Perform a monotonicity check with term(s) of interest, such as fit error, loss and IOU.

    history : data frame
        The object to check.
    columns : list of string
        The term(s) of interest, such as Loss, FitError, ValidLoss. It is column(s) name of history data frame.
    tolerance : int
        Fault tolerant, number of peak appear in monotone. Default is set to 0 which is strictly checking.
        Default : 0
    decreasing : Boolean
        Set the order of monotonicity. Monotonically decreasing as default.
        Default : True

    '''
    if type(columns) is str:
        columns = [columns]
    for col in columns:
        content = history[col]
        if decreasing:
            res = [x > y for x, y in zip(content, content[1:])]
        else:
            res = [x < y for x, y in zip(content, content[1:])]
        peak = res.count(False)
        if peak >= tolerance:
            raise ValueError('{} term fails with monotonicity check while calling check_monotonicity().\n'
                             'The error happens when loss, fit error, IOU does not decrease/increase as expected. \n'
                             'Please compare with according benchmark. \n'
                             'Learn more about check_monotonicity(), '
                             'please go to {}/dlpy/test_toolkit.py'.format(col, sys.path[-1]))
    print('check monotonicity passed')


def almost_match(history, benchmark, significant=4):
    '''
    Check if history and benchmark are almost same.

    history : data frame
        The object to check.
    benchmark : dictionary
        The expected object.
    significant : int, optional
        Desired precision.
        Default: 4
    '''
    for key, value in benchmark.items:
        # assert(history['key'] != len(value), 'The length of benchmark and history should be equal.')
        for actual, desired in zip(history['key'], value):
            np.testing.assert_approx_equal(actual, desired, significant)

