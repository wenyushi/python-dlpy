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

# NOTE: This test requires a running CAS server.  You must use an ~/.authinfo
#       file to specify your username and password.  The CAS host and port must
#       be specified using the CASHOST and CASPORT environment variables.
#       A specific protocol ('cas', 'http', 'https', or 'auto') can be set using
#       the CASPROTOCOL environment variable.

import os
import swat
import swat.utils.testing as tm
import numpy as np
import json
from dlpy.layers import (InputLayer, Detection, Conv2d, Pooling, Dense, OutputLayer,
                         Recurrent, Keypoints, BN, Res, Concat, Reshape)
from dlpy.model_conversion.sas_tvm_parse import tvm_to_sas
import unittest


class TestTVMModel(unittest.TestCase):
    '''
    Please locate the images.sashdat file under the datasources to the DLPY_DATA_DIR.
    '''
    server_type = None
    s = None
    server_sep = '/'
    data_dir = None
    data_dir_local = None

    @classmethod
    def setUpClass(cls):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        cls.s = swat.CAS('dlgrd010', 23309)
        cls.server_type = tm.get_cas_host_type(cls.s)
        cls.server_sep = '\\'
        if cls.server_type.startswith("lin") or cls.server_type.startswith("osx"):
            cls.server_sep = '/'

        if 'DLPY_DATA_DIR' in os.environ:
            cls.data_dir = os.environ.get('DLPY_DATA_DIR')
            if cls.data_dir.endswith(cls.server_sep):
                cls.data_dir = cls.data_dir[:-1]
            cls.data_dir += cls.server_sep

        if 'DLPY_DATA_DIR_LOCAL' in os.environ:
            cls.data_dir_local = os.environ.get('DLPY_DATA_DIR_LOCAL')
            if cls.data_dir_local.endswith(cls.server_sep):
                cls.data_dir_local = cls.data_dir_local[:-1]
            cls.data_dir_local += cls.server_sep

    def test_model1(self):
        yolo_anchors = [9.778984, 5.595484, 2.98513, 3.924927, 11.580547, 11.534526, 5.183913, 8.863515, 1.09074,
                        1.433951]

        output_laye = Detection(name = 'Detection1',
                                detection_model_type = 'yolov2',
                                anchors = yolo_anchors,
                                predictions_per_grid = 5,
                                class_number = 313,
                                softmax_for_class_prob = True,
                                object_scale = 5.0,
                                prediction_not_a_object_scale = 1.0,
                                class_scale = 1.0,
                                coord_scale = 1.0,
                                act = 'LOGISTIC',
                                grid_number = 13,
                                coord_type = 'YOLO',
                                detection_threshold = 0.3,
                                iou_threshold = 0.3)

        tvm_graph_path = r'\\sashq\root\data\DeepLearn\weshiz\TVM\Tiny_yolov2.json'
        tvm_graph = json.load(open(tvm_graph_path))
        tvm_params_dict = np.load(r'\\sashq\root\data\DeepLearn\weshiz\TVM\Tiny_yolov2_params.npy')
        tvm_params_dict = tvm_params_dict[()]

        # build_graph(self.s, 'tvm_model', tvm_graph_path, output_laye)
        tvm_to_sas(self.s, 'tvm_mode', tvm_graph, tvm_params_dict, output_laye)

    def test_model2(self):
        from dlpy.layers import Segmentation
        output_laye = Segmentation(name='Segmentation_1',
                                   outputImageType='BASE64', outputImageProb=False)
        path = "/cas/DeepLearn/models/RCafe"
        tvm_graph_path = f'{path}/model_zdm2ii_graph.json'
        tvm_graph = json.load(open(tvm_graph_path))
        tvm_params_dict = np.load(f'{path}/model_zdm2ii_param2.npy', allow_pickle=True)
        tvm_params_dict = tvm_params_dict[()]
        # output_laye = None
        model_fusion = tvm_to_sas(self.s, 'tvm_mode', tvm_graph, tvm_params_dict, ['inputlayer_1'], output_laye)

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()
