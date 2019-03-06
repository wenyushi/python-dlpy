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
#

import swat
import swat.utils.testing as tm
from dlpy.sequential import Sequential
from dlpy.layers import *
from dlpy.blocks import Bidirectional
from dlpy.utils import DLPyError


class TestSequential(tm.TestCase):
    # Create a class attribute to hold the cas host type
    server_type = None
    s = None

    @classmethod
    def setUpClass(cls):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        cls.s = swat.CAS()
        cls.server_type = tm.get_cas_host_type(cls.s)

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()

    def test_1(self):
        with self.assertRaises(DLPyError):
            Sequential(self.s, layers='', model_table='table1')

    def test_11(self):
        with self.assertRaises(DLPyError):
            model1 = Sequential(self.s, model_table='table11')
            model1.add(Conv2d(8, 7))
            model1.compile()

    def test_2(self):
        layers = [InputLayer(), Dense(n=32), OutputLayer()]
        Sequential(self.s, layers=layers, model_table='table2')

    def test_22(self):
        layers = [InputLayer(3, 4), Conv2d(8, 7), BN(), Dense(n=32), OutputLayer()]
        Sequential(self.s, layers=layers, model_table='table22')

    def test_3(self):
        model1 = Sequential(self.s, model_table='table3')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.pop()

    def test_4(self):
        model1 = Sequential(self.s, model_table='table4')
        model1.add(Conv2d(8, 7))
        model1.add(InputLayer(3, 224, 224))
        model1.switch(0, 1)

    def test_5(self):
        with self.assertRaises(DLPyError):
            model1 = Sequential(self.s, model_table='table5')
            model1.compile()

    def test_6(self):
        model1 = Sequential(self.s, model_table='table6')
        model1.add(Bidirectional(n=10, n_blocks=3))
        model1.add(OutputLayer())

    def test_simple_cnn_seq1(self):
        model1 = Sequential(self.s, model_table='table7')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

    def test_simple_cnn_seq2(self):
        model1 = Sequential(self.s, model_table='table8')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))
        model1.print_summary()

    def test_new_bidirectional1(self):
        model = Sequential(self.s, model_table='new_table1')
        model.add(Bidirectional(n=10))
        model.add(OutputLayer())
        model.print_summary()

    def test_new_bidirectional2(self):
        model = Sequential(self.s, model_table='new_table2')
        model.add(Bidirectional(n=10, n_blocks=3))
        model.add(OutputLayer())
        model.print_summary()

    def test_new_bidirectional3(self):
        model = Sequential(self.s, model_table='new_table3')
        model.add(Bidirectional(n=[10, 20, 30], n_blocks=3))
        model.add(OutputLayer())
        model.print_summary()

    def test_new_bidirectional4(self):
        model = Sequential(self.s, model_table='new_table4')
        model.add(InputLayer())
        model.add(Recurrent(n=10, name='rec1'))
        model.add(Bidirectional(n=20, src_layers=['rec1']))
        model.add(OutputLayer())
        model.print_summary()

    def test_new_bidirectional5(self):
        model = Sequential(self.s, model_table='new_table5')
        model.add(InputLayer())
        model.add(Recurrent(n=10, name='rec1'))
        model.add(Bidirectional(n=20, src_layers=['rec1']))
        model.add(Recurrent(n=10))
        model.add(OutputLayer())
        model.print_summary()

    def test_new_bidirectional6(self):
        model = Sequential(self.s, model_table='new_table5')
        model.add(InputLayer())
        r1 = Recurrent(n=10, name='rec1')
        model.add(r1)
        model.add(Bidirectional(n=20, src_layers=[r1]))
        model.add(Recurrent(n=10))
        model.add(OutputLayer())
        model.print_summary()

    def test_unet(self):
        model = Sequential(conn = self.s, model_table = 'model_table')

        model.add(InputLayer(n_channels = 1, width = 512, height = 512,
                             scale = 1.0 / 255, random_flip = 'None',
                             random_crop = 'None'))
        # conv1 512
        model.add(Conv2d(8, width = 3, act = 'relu', include_bias = False, stride = 1))
        # model.add(BN(act=actx))
        model.add(Pooling(width = 2, height = 2, stride = 2, pool = 'max'))
        # conv2 256
        model.add(Conv2d(16, width = 3, act = 'relu', include_bias = False, stride = 1))
        # model.add(BN(act=actx))
        model.add(Pooling(width = 2, height = 2, stride = 2, pool = 'max'))
        # conv3 128
        model.add(Transconvo(n_filters = 8, output_size = (256, 256, 8), height = 3, padding = 1,
                             width = 3, act = 'relu', include_bias = False, stride = 2))
        # model.add(BN(act=actx))
        # conv4 256
        model.add(Transconvo(n_filters = 2, output_size = (512, 512, 2), height = 3, padding = 1,
                             width = 3, act = 'relu', include_bias = False, stride = 2))
        # model.add(BN(act=actx))
        # 512
        model.add(Segmentation(n_class = 2))
        model.print_summary()

    def test_FCMP(self):
        self.s.loadactionset('fcmpact')
        self.s.addRoutines(
            routineCode = {
                '''
                function forward_prop( srcHeight, srcWidth, srcDepth, srcY[*], weights[*], y_out[*] );
                    outargs y_out;
                    nWeights = 0;
                    nNeurons = dim(y_out);
                    srcNeurons = srcHeight*srcWidth*srcDepth;
                    featuremapsize = srcHeight*srcWidth;
                    gramsize = srcDepth*srcDepth;
    
                    array result_temp[1] /nosymbols;
                    call dynamic_array(result_temp, srcDepth, srcDepth);
    
                    array matrix_reshape[1] /nosymbols;
                    call dynamic_array(matrix_reshape, srcDepth, featuremapsize);
                    /* reshape srcY tensor into (srcDepth, (srcHeight, srcWidth))*/
                    do i=0 to srcNeurons;
                        matrix_reshape[int((i-1)/featuremapsize)+1, mod((i-1),featuremapsize)+1] = srcY[i];
                    end;
                    put matrix_reshape=;
                    array mat1_t[1] /nosymbols;
                    call dynamic_array(mat1_t, featuremapsize, srcDepth);
    
                    /* mat1_t srcY tensor into ((srcHeight, srcWidth), srcDepth)*/
                    call transpose(matrix_reshape, mat1_t);
    
                    /* mat1_t srcY tensor into (srcDepth, srcDepth)*/
                    call mult(matrix_reshape, mat1_t, result_temp);
    
                    /* reshape result_temp and write into y_out */
                    do i=1 to gramsize;
                        y_out[i] = result_temp[int((i-1)/srcDepth)+1, mod((i-1),srcDepth)+1]/featuremapsize;
                    end;
                    put y_out=;
                    return;
                endsub;
    
                function back_prop( srcHeight, srcWidth, srcDepth, srcY[*], Y[*], weights[*], deltas[*],
                                    gradient_out[*], srcDeltas_out[*]);
                    /* deltas: partial derivative wrt Y[*] shape of its is (srcDepth, srcDepth);*/
                    outargs srcDeltas_out, gradient_out;
                    nWeights = 0;
                    nNeurons = dim(Y);
                    srcNeurons = srcHeight*srcWidth*srcDepth;
                    featuremapsize = srcHeight*srcWidth;
                    gramsize = srcDepth*srcDepth;
    
                    array result_temp[1] /nosymbols;
                    call dynamic_array(result_temp, srcDepth, featuremapsize);
    
                    array deltas_reshape[1] /nosymbols;
                    call dynamic_array(deltas_reshape, srcDepth, srcDepth);
                    do i=1 to gramsize;
                        deltas_reshape[int((i-1)/srcDepth)+1, mod((i-1),srcDepth)+1] = deltas[i]/featuremapsize;
                    end;
    
                    array srcY_temp[1] /nosymbols;
                    call dynamic_array(srcY_temp, srcDepth, featuremapsize);
                    do i=1 to srcNeurons;
                        srcY_temp[int((i-1)/featuremapsize)+1, mod((i-1),featuremapsize)+1] = srcY[i];
                    end;
    
                    call mult(deltas_reshape, srcY_temp, result_temp);
    
                    do i=1 to srcNeurons;
                        srcDeltas_out[i] = result_temp[int((i-1)/featuremapsize)+1, mod((i-1),featuremapsize)+1];
                    end;
                    return;
                endsub;
                '''},
            package = "pkg",
            saveTable = 1,
            funcTable = dict(name = "gramfcmp", caslib = "casuser", replace = 1)
        )
        self.s.sessionProp.setsessopt(cmplib = 'CASUSER.gramfcmp')
        model_table = 'cifar10'
        factor = 8
        model = Sequential(conn = self.s, model_table = model_table)

        model.add(InputLayer(n_channels = 3, width = 4, height = 4, scale = 1.0 / 255,
                             offsets = (103.939 / 255, 116.779 / 255, 123.68 / 255),
                             random_flip = 'none', random_crop = 'unique'))

        model.add(Conv2d(n_filters = 64 / factor, width = 3, height = 3, stride = 1))
        # model.add(Conv2d(n_filters=64/factor, width=3, height=3, stride=1))
        # model.add(Pooling(width=2, height=2, stride=2, pool='max'))
        # # 14
        # model.add(Conv2d(n_filters=128/factor, width=3, height=3, stride=1))
        # model.add(Conv2d(n_filters=128/factor, width=3, height=3, stride=1))
        # model.add(Pooling(width=2, height=2, stride=2, pool='max'))
        # # 7
        # model.add(Conv2d(n_filters=256/factor, width=3, height=3, sride=1))
        # model.add(Conv2d(n_filters=256/factor, width=3, height=3, stride=1))
        # model.add(Conv2d(n_filters=256/factor, width=3, height=3, stride=1))

        model.add(FCMP(width = 8, height = 8, depth = 1, forward_func = 'forward_prop', backward_func = 'back_prop',
                       n_weights = 0))
        model.add(OutputLayer(act = 'softmax', n = 10))
