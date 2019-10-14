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

import os
import sys
import numpy as np
import json
import platform
try:
    from utils import DLPyError
except:
    from .utils import DLPyError


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
    for key, value in benchmark.items():
        # assert(history['key'] != len(value), 'The length of benchmark and history should be equal.')
        for actual, desired in zip(history[key], value):
            np.testing.assert_approx_equal(actual, desired, significant)


def check_dlscore_astore_score_match(s, data_table, model_table, model_weights, decimal=0.001, **kwargs):
    df = s.fetch(data_table, sastypes=False).Fetch
    where_clause = ''  # the where clause to select one observation if _path_ exists.
    where_column = ''
    if '_path_' in df.columns:
        where_column = '_path_'
    elif '_filename_0'in df.columns:
        where_column = '_filename_0'
    else:
        raise DLPyError('{} doesn\'t contain _path_'.format(['_path', '_filename_0']))

    where_clause = '{} eq "{}"'.format(where_column, df.loc[0, where_column])
    num_ob = s.numrows(dict(name=data_table, where=where_clause)).numrows
    if num_ob < 1:
        raise DLPyError('Something is wrong when selecting observation in check_dlscore_astore_score_match().'
                        ' Please contact Wenyu.')

    s.dlscore(model=model_table, initWeights=model_weights,
              table=dict(name=data_table, where=where_clause),
              casout={'name': 'dlscore_results', 'replace': True}, **kwargs)

    s.dlexportmodel(casout=dict(name='export', replace=True),
                    initWeights=model_weights, modelTable=model_table)

    # enable gpu on astore
    if 'gpu' in kwargs:
        kwargs.pop('gpu')
        kwargs['options'] = [dict(name='usegpu', value='1'),
                            dict(name='NDEVICES', value=1),
                            dict(name='DEVICE0', value='0')]
        kwargs['nthreads'] = 1

    s.score(rstore='export', table=dict(name=data_table, where=where_clause),
            out=dict(name='astore_results', replace=True), **kwargs)

    dlscore_df = s.fetch('dlscore_results').Fetch
    score_df = s.fetch('astore_results').Fetch

    layers_info = s.fetch(dict(name=model_table, where=' _DLKey1_ eq "layertype"'), to=10000).Fetch

    task_layer_numval = [5, 11, 13, 19, 23, 26]  # output, keypoints, segmentation, rpn, clustering
    task_layer_indices = layers_info[layers_info['_DLNumVal_'].isin(task_layer_numval)]['_DLLayerID_'].values

    output_layer_indices = layers_info[layers_info['_DLNumVal_'] == 5]['_DLLayerID_'].values
    seg_layer_indices = layers_info[layers_info['_DLNumVal_'] == 19]['_DLLayerID_'].values

    drop_cols = []  # drop the columns which are not in astore casout
    if task_layer_indices.shape[0] > 1:
        for i in output_layer_indices:
            drop_cols.append('_DL_PredLevel{}_'.format(int(i)))
            drop_cols.append('_DL_PredP{}_'.format(int(i)))
        for i in seg_layer_indices:
            # todo check columns to be removed
            drop_cols.append('_DL_Pixel_Acc_')
            drop_cols.append('_DL_Mean_Acc_')
            drop_cols.append('_DL_Mean_IU_')
            drop_cols.append('_DL_Freq_IU_')
    elif output_layer_indices.shape[0] > 1:
        # single task
        if output_layer_indices.shape[0] == 1:
            drop_cols.append('_DL_PredLevel_')
            drop_cols.append('_DL_PredP_')
    elif seg_layer_indices.shape[0] > 1:
        drop_cols.append('_DL_Pixel_Acc_')
        drop_cols.append('_DL_Mean_Acc_')
        drop_cols.append('_DL_Mean_IU_')
        drop_cols.append('_DL_Freq_IU_')

    keep = [c for c in dlscore_df.columns if c not in drop_cols]

    dlscore_np = dlscore_df[keep].values
    score_np = score_df.values

    for v1, v2 in zip(dlscore_np[0], score_np[0]):
        if type(v1) in [float, int]:
            np.testing.assert_almost_equal(v1, v2, decimal=decimal)
        elif type(v1) == str:
            np.testing.assert_string_equal(v1, v2)
        else:
            raise DLPyError('Unexpect error. Contact Wenyu.')

    print('dlscore and astore.score are matching.')


def convert_to_notebook(table_test_file, save_to_folder, server='dlgrd009', port=13315):
    '''
    Convert Castest table test to jupyter notebook.

    table_test_file : str
        Point to the table test file to be converted.
    save_to_folder : str
        Point to the directory where the jupyter notebook is stored.
    server : str
        Server name.
        Default : 'dgrd008'
    port : int
        Port
        Default : 13315

    '''
    pattern_line_num = dict()
    filename = os.path.split(table_test_file)
    func_name = filename[1].split('.')[0]
    needles = ['def', func_name]
    f_in = open(table_test_file)
    data = f_in.readlines()
    for i, line in enumerate(data):
        if line.startswith('def'):
            pattern_line_num['def'] = i
        if line.startswith(func_name):
            pattern_line_num[func_name] = i


    pre_func_code = [line for line in data[:pattern_line_num['def']]
                     if not (line.startswith('from connect import *') or line.startswith('s = connect()') or
                             line.startswith('ast(s)'))]
    ast_lib_code = ["from swat import *\n",
        "s = CAS('{}.unx.sas.com', {})\n".format(server, port),
        "s.table.addcaslib(activeonadd=False, datasource={'srctype':'path'}, name='ast', path='/dept/ast/data', subdirectories=True)"]
    func_code = [line[4:] for line in data[pattern_line_num['def']+1: pattern_line_num[func_name]-3]]

    notebook_dict = {"cells": [],
                     "metadata": {},
                     "nbformat": 4,
                     "nbformat_minor": 2}

    notebook_dict["cells"] = [{"cell_type": "code",
                               "metadata": {},
                               "execution_count": 1,
                               "outputs": [],
                               "source": pre_func_code},
                              {"cell_type": "code",
                               "metadata": {},
                               "execution_count": 2,
                               "outputs": [],
                               "source": ast_lib_code},
                              {"cell_type": "code",
                               "metadata": {},
                               "execution_count": 3,
                               "outputs": [],
                               "source": func_code}
                              ]
    if platform.system() == 'Windows':
        save_path = r"{}\{}.ipynb".format(save_to_folder, func_name)
    else:
        save_path = "{}/{}.ipynb".format(save_to_folder, func_name)

    with open(save_path, "w") as outfile:
        json.dump(notebook_dict, outfile, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-tf', '--test_file', help = 'Point to the table test file to be converted.',
                        required = True, type = str)
    parser.add_argument('-sf', '--save_to_folder', help = 'Point to the directory where the jupyter notebook is stored.',
                        required = True, type = str)
    parser.add_argument('-s', '--server', help = 'machine name', default = 'dlgrd009', required = False, type = str)
    parser.add_argument('-port', help = 'integer: port number', default = 13300, required = False)

    args = parser.parse_args()
    test_file = args.test_file
    save_to_folder = args.save_to_folder
    server = args.server
    port = args.port

    convert_to_notebook(test_file, save_to_folder, server, port)
