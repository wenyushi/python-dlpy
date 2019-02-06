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

''' A Network is way to compose layers: the topological form of a Model '''

import os

from dlpy.layers import Layer
from dlpy.utils import DLPyError, input_table_check, random_name, check_caslib, caslibify, get_server_path_sep
from .layers import InputLayer, Conv2d, Pooling, BN, Res, Concat, Dense, OutputLayer, Keypoints, Detection
import collections
import pandas as pd


class Network(Layer):

    '''
    Network

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    inputs: iter-of-Layers, optional
        Specifies some input layer(s) to instantiate a Network
    outputs: iter-of-Layers, optional
        Specifies some output layer(s) to instantiate a Network
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
        Default: None
    model_weights : CASTable or string or dict, optional
        Specifies the CASTable containing weights of the deep learning model.
        If not specified, random initial will be used.
        Default: None
    Returns
    -------
    :class:`Model`

    '''

    def __init__(self, conn, inputs=None, outputs=None, model_table=None, model_weights=None):
        if model_table is not None and any(i is not None for i in [inputs, outputs]):
            raise DLPyError('Either parameter model_table or inputs and outputs needs to be set.\n'
                            'The following cases are valid.\n'
                            '1. model_table = "your_model_table"; inputs = None; outputs = None.\n'
                            '2. model_table = None; inputs = input_layer(s); outputs = output_layer.'
                            )
        self._init_model(conn, model_table, model_weights)
        if all(i is None for i in [inputs, outputs, model_table]):
            return
        if self.__class__.__name__ == 'Model':
            if None in [inputs, outputs]:
                raise DLPyError('Parameter inputs and outputs are required.')
            self._map_graph_network(inputs, outputs)

    def _init_model(self, conn, model_table=None, model_weights=None):
        conn.loadactionset(actionSet='deeplearn', _messagelevel='error')

        self.conn = conn

        if model_table is None:
            model_table = dict(name=random_name('Model', 6))

        model_table_opts = input_table_check(model_table)

        if 'name' not in model_table_opts.keys():
            model_table_opts.update(**dict(name=random_name('Model', 6)))

        self.model_name = model_table_opts['name']
        self.model_table = model_table_opts

        if model_weights is None:
            self.model_weights = self.conn.CASTable('{}_weights'.format(self.model_name))
        else:
            self.set_weights(model_weights)

        self.layers = []
        self.model_type = 'CNN'
        self.best_weights = None
        self.target = None
        self.num_params = None

    def _map_graph_network(self, inputs, outputs):
        """propagate all of layers"""
        def build_map(start):
            if start.name is None:
                start.count_instances()
                start.name = str(start.__class__.__name__) + '_' + str(type(start).number_of_instances)
            """if the node is visited, continue; the layer is a input layer and not added"""
            if start in inputs or start in self.layers:
                if start not in self.layers:
                    self.layers.append(start)
                return
            for layer in start.src_layers:
                build_map(layer)
                ''' if all of src_layer of layer is in layers list, add it in layers list'''
                if all(i in self.layers for i in start.src_layers):
                    self.layers.append(start)
                # set the layer's depth
                layer.depth = 0 if str(layer.__class__.__name__) == 'InputLayer' \
                    else max([i.depth for i in layer.src_layers]) + 1
            return

        if not isinstance(inputs, collections.Iterable):
            inputs = [inputs]
        if any(x.__class__.__name__ != 'InputLayer' for x in inputs):
            raise DLPyError('Input layers should be input layer type.')
        if not isinstance(outputs, collections.Iterable):
            outputs = [outputs]
        if not all(x.can_be_last_layer for x in outputs):
            raise DLPyError('Output layers can only be {}'\
                            .format([i.__name__ for i in Layer.__subclasses__() if i.can_be_last_layer]))
        for layer in outputs:
            build_map(layer)
            layer.depth = max([i.depth for i in layer.src_layers]) + 1

        return

    def compile(self):
        ''' parse the network nodes and process CAS Action '''
        rt = self._retrieve_('deeplearn.buildmodel',
                             model=dict(name=self.model_name, replace=True), type=self.model_type)

        if rt.severity > 1:
            raise DLPyError('cannot build model, there seems to be a problem.')
        self.num_params = 0
        for layer in self.layers:
            option = layer.to_model_params()
            rt = self._retrieve_('deeplearn.addlayer', model = self.model_name, **option)
            if rt.severity > 1:
                raise DLPyError('there seems to be an error while adding the ' + layer.name + '.')
            if layer.num_weights is None:
                num_weights = 0
            else:
                num_weights = layer.num_weights

            if layer.num_bias is None:
                num_bias = 0
            else:
                num_bias = layer.num_bias

            self.num_params += num_weights + num_bias
        print('NOTE: Model compiled successfully.')

    def _retrieve_(self, _name_, message_level='error', **kwargs):
        ''' Call a CAS action '''
        return self.conn.retrieve(_name_, _messagelevel=message_level, **kwargs)

    @classmethod
    def from_table(cls, input_model_table, display_note = True, output_model_table = None):
        '''
        Create a Model object from CAS table that defines a deep learning model

        Parameters
        ----------
        input_model_table : CASTable
            Specifies the CAS table that defines the deep learning model.
        display_note : bool, optional
            Specifies whether to print the note when generating the model table.
        output_model_table : string or dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default: None

        Returns
        -------
        :class:`Model`

        '''
        model = cls(conn = input_model_table.get_connection(), model_table = output_model_table)
        model_name = model._retrieve_('table.fetch',
                                      table = dict(where = '_DLKey1_= "modeltype"',
                                                   **input_model_table.to_table_params()))
        model_name = model_name.Fetch['_DLKey0_'][0]
        if display_note:
            print(('NOTE: Model table is attached successfully!\n'
                   'NOTE: Model is named to "{}" according to the '
                   'model name in the table.').format(model_name))
        model.model_name = model_name
        model.model_table.update(**input_model_table.to_table_params())
        model.model_weights = model.conn.CASTable('{}_weights'.format(model_name))

        model_table = input_model_table.to_frame()
        for layer_id in range(int(model_table['_DLLayerID_'].max()) + 1):
            layer_table = model_table[model_table['_DLLayerID_'] == layer_id]
            layer_type = layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                                   'layertype'].tolist()[0]
            if layer_type == 1:
                model.layers.append(extract_input_layer(layer_table = layer_table))
            elif layer_type == 2:
                model.layers.append(extract_conv_layer(layer_table = layer_table))
            elif layer_type == 3:
                model.layers.append(extract_pooling_layer(layer_table = layer_table))
            elif layer_type == 4:
                model.layers.append(extract_fc_layer(layer_table = layer_table))
            elif layer_type == 5:
                model.layers.append(extract_output_layer(layer_table = layer_table))
            elif layer_type == 8:
                model.layers.append(extract_batchnorm_layer(layer_table = layer_table))
            elif layer_type == 9:
                model.layers.append(extract_residual_layer(layer_table = layer_table))
            elif layer_type == 10:
                model.layers.append(extract_concatenate_layer(layer_table = layer_table))
            elif layer_type == 11:
                model.layers.append(extract_detection_layer(layer_table = layer_table))
        conn_mat = model_table[['_DLNumVal_', '_DLLayerID_']][
            model_table['_DLKey1_'].str.contains('srclayers')].sort_values('_DLLayerID_')
        layer_id_list = conn_mat['_DLLayerID_'].tolist()
        src_layer_id_list = conn_mat['_DLNumVal_'].tolist()

        for row_id in range(conn_mat.shape[0]):
            layer_id = int(layer_id_list[row_id])
            src_layer_id = int(src_layer_id_list[row_id])
            if model.layers[layer_id].src_layers is None:
                model.layers[layer_id].src_layers = [model.layers[src_layer_id]]
            else:
                model.layers[layer_id].src_layers.append(model.layers[src_layer_id])

        return model

    @classmethod
    def from_sashdat(cls, conn, path, output_model_table = None):
        '''
        Generate a model object using the model information in the sashdat file

        Parameters
        ----------
        conn : CAS
            The CAS connection object.
        path : string
            The path of the sashdat file, the path has to be accessible
            from the current CAS session.
        output_model_table : string or dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default: None

        Returns
        -------
        :class:`Model`

        '''
        model = cls(conn, model_table = output_model_table)
        model.load(path = path)
        return model

    @classmethod
    def from_caffe_model(cls, conn, input_network_file, output_model_table = None,
                         model_weights_file = None, **kwargs):
        '''
        Generate a model object from a Caffe model proto file (e.g. *.prototxt), and
        convert the weights (e.g. *.caffemodel) to a SAS capable file (e.g. *.caffemodel.h5).

        Parameters
        ----------
        conn : CAS
            The CAS connection object.
        input_network_file : string
            Fully qualified file name of network definition file (*.prototxt).
        model_weights_file : string, optional
            Fully qualified file name of model weights file (*.caffemodel)
            Default: None
        output_model_table : string or dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default: None

        Returns
        -------
        :class:`Model`

        '''
        from .model_conversion.sas_caffe_parse import caffe_to_sas

        if output_model_table is None:
            output_model_table = dict(name = random_name('caffe_model', 6))

        model_table_opts = input_table_check(output_model_table)

        if 'name' not in model_table_opts.keys():
            model_table_opts.update(**dict(name = random_name('caffe_model', 6)))

        model_name = model_table_opts['name']

        output_code = caffe_to_sas(input_network_file, model_name, network_param = model_weights_file, **kwargs)
        exec(output_code)
        temp_name = conn
        exec('sas_model_gen(temp_name)')
        input_model_table = conn.CASTable(**model_table_opts)
        model = cls.from_table(input_model_table = input_model_table)
        return model

    @classmethod
    def from_keras_model(cls, conn, keras_model, output_model_table = None,
                         include_weights = False, input_weights_file = None):
        '''
        Generate a model object from a Keras model object

        Parameters
        ----------
        conn : CAS
            The CAS connection object.
        keras_model : keras_model object
            Specifies the keras model to be converted.
        output_model_table : string or dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default: None
        include_weights : bool, optional
            Specifies whether to load the weights of the keras model.
            Default: True
        input_weights_file : string, optional
            A fully specified client side path to the HDF5 file that stores
            the keras model weights. Only effective when include_weights=True.
            If None is given, the current weights in the keras model will be used.
            Default: None

        Returns
        -------
        :class:`Model`

        '''

        from .model_conversion.sas_keras_parse import keras_to_sas
        if output_model_table is None:
            output_model_table = dict(name = random_name('caffe_model', 6))

        model_table_opts = input_table_check(output_model_table)

        if 'name' not in model_table_opts.keys():
            model_table_opts.update(**dict(name = random_name('caffe_model', 6)))

        model_name = model_table_opts['name']

        output_code = keras_to_sas(model = keras_model, model_name = model_name)
        exec(output_code)
        temp_name = conn
        exec('sas_model_gen(temp_name)')
        input_model_table = conn.CASTable(**model_table_opts)
        model = cls.from_table(input_model_table = input_model_table)

        if include_weights:
            from .model_conversion.write_keras_model_parm import write_keras_hdf5, write_keras_hdf5_from_file
            temp_HDF5 = os.path.join(os.getcwd(), '{}_weights.kerasmodel.h5'.format(model_name))
            if input_weights_file is None:
                write_keras_hdf5(keras_model, temp_HDF5)
            else:
                write_keras_hdf5_from_file(keras_model, input_weights_file, temp_HDF5)
            print('NOTE: the model weights has been stored in the following file:\n'
                  '{}'.format(temp_HDF5))
        return model

    @classmethod
    def from_onnx_model(cls, conn, onnx_model, output_model_table = None,
                        offsets = None, scale = None, std = None, output_layer = None):
        '''
        Generate a Model object from ONNX model.

        Parameters
        ----------
        conn : CAS
            Specifies the CAS connection object.
        onnx_model : ModelProto
            Specifies the ONNX model.
        output_model_table : string or dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default: None
        offsets : int-list, optional
            Specifies the values to be subtracted from the pixel values
            of the input data, used if the data is an image.
        scale : float, optional
            Specifies the scaling factor to apply to each image.
        std : string, optional
            Specifies how to standardize the variables in the input layer.
            Valid Values: MIDRANGE, NONE, STD
        output_layer : Layer object, optional
            Specifies the output layer of the model. If no output
            layer is specified, the last layer is automatically set
            as :class:`OutputLayer` with SOFTMAX activation.

        Returns
        -------
        :class:`Model`

        '''

        from .model_conversion.sas_onnx_parse import onnx_to_sas
        if output_model_table is None:
            output_model_table = dict(name = random_name('onnx_model', 6))

        model_table_opts = input_table_check(output_model_table)

        if 'name' not in model_table_opts.keys():
            model_table_opts.update(**dict(name = random_name('onnx_model', 6)))

        model_name = model_table_opts['name']

        _layers = onnx_to_sas(onnx_model, model_name, output_layer)
        if offsets is not None:
            _layers[0].config.update(offsets = offsets)
        if scale is not None:
            _layers[0].config.update(scale = scale)
        if std is not None:
            _layers[0].config.update(std = std)
        if len(_layers) == 0:
            raise DLPyError('Unable to import ONNX model.')

        conn.loadactionset('deeplearn', _messagelevel = 'error')
        rt = conn.retrieve('deeplearn.buildmodel',
                           _messagelevel = 'error',
                           model = dict(name = model_name, replace = True),
                           type = 'CNN')
        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError('Cannot build model, there seems to be a problem.')

        for layer in _layers:
            option = layer.to_model_params()
            rt = conn.retrieve('deeplearn.addlayer', _messagelevel = 'error',
                               model = model_name, **option)
            if rt.severity > 1:
                for m in rt.messages:
                    print(m)
                raise DLPyError('There seems to be an error while adding the '
                                + layer.name + '.')

        input_model_table = conn.CASTable(**model_table_opts)
        model = cls.from_table(input_model_table = input_model_table)
        print('NOTE: Successfully imported ONNX model.')
        return model

    @property
    def summary(self):
        if self.model_type == 'CNN':
            return pd.concat([x.summary for x in self.layers], ignore_index = True)
        else:
            return pd.concat([x.rnn_summary for x in self.layers], ignore_index = True)

    def __load_layer_ids(self):
        try:
            model_table_rows = self.conn.table.fetch(self.model_table, maxrows = 1000000, to = 1000000).Fetch
        except:
            model_table_rows = None
        if model_table_rows is not None:
            layer_ids = {}
            import math
            for index, row in model_table_rows.iterrows():
                if not math.isnan(row['_DLLayerID_']):
                    layer_ids[row['_DLKey0_']] = int(row['_DLLayerID_'])

            for l in self.layers:
                l.layer_id = layer_ids[l.name.lower()]

    def print_summary(self):
        ''' Display a table that summarizes the model architecture '''
        try:
            if len(self.layers) > 0 and self.layers[0].layer_id is None:
                self.__load_layer_ids()

            from IPython.display import display

            if self.model_type == 'CNN':
                if self.num_params is None:
                    self.num_params = 0
                    for l in self.layers:
                        if l.num_weights is not None:
                            self.num_params += l.num_weights
                        if l.num_bias is not None:
                            self.num_params += l.num_bias

                total = pd.DataFrame([['', '', '', '', '', '', '', self.num_params]],
                                     columns = ['Layer Id', 'Layer', 'Type', 'Kernel Size', 'Stride', 'Activation',
                                                'Output Size', 'Number of Parameters'])
                display(pd.concat([self.summary, total], ignore_index = True))
            else:
                display(self.summary)

        except ImportError:
            print(self.summary)

    def _repr_html_(self):
        return self.summary._repr_html_()

    def plot_network(self):
        '''
        Display a graph that summarizes the model architecture.

        Returns
        -------
        :class:`graphviz.dot.Digraph`

        '''
        return model_to_graph(self)

    def _repr_svg_(self):
        return self.plot_network()._repr_svg_()

    def set_weights(self, weight_tbl):
        '''
        Assign weights to the Model object

        Parameters
        ----------
        weight_tbl : CASTable or string or dict
            Specifies the weights CAS table for the model

        '''
        weight_tbl = input_table_check(weight_tbl)
        weight_name = self.model_name + '_weights'

        if weight_tbl['name'].lower() != weight_name.lower():
            self._retrieve_('table.partition',
                            casout=dict(replace=True, name=self.model_name + '_weights'),
                            table=weight_tbl)

        self.model_weights = self.conn.CASTable(name=self.model_name + '_weights')
        print('NOTE: Model weights attached successfully!')

    def load(self, path, display_note=True):
        '''
        Load the deep learning model architecture from existing table

        Parameters
        ----------
        path : string
            Specifies the absolute server-side path of the table file.
        display_note : bool
            Specifies whether to print the note when generating the model table.

        '''

        cas_lib_name, file_name = caslibify(self.conn, path, task='load')

        self._retrieve_('table.loadtable',
                        caslib=cas_lib_name,
                        path=file_name,
                        casout=dict(replace=True, **self.model_table))

        model_name = self._retrieve_('table.fetch',
                                     table=dict(where='_DLKey1_= "modeltype"',
                                                **self.model_table)).Fetch['_DLKey0_'][0]

        if model_name.lower() != self.model_name.lower():
            self._retrieve_('table.partition',
                            casout=dict(replace=True, name=model_name),
                            table=self.model_name)

            self._retrieve_('table.droptable', **self.model_table)
            if display_note:
                print(('NOTE: Model table is loaded successfully!\n'
                       'NOTE: Model is renamed to "{}" according to the '
                       'model name in the table.').format(model_name))
            self.model_name = model_name
            self.model_table['name'] = model_name
            self.model_weights = self.conn.CASTable('{}_weights'.format(self.model_name))

        model_table = self.conn.CASTable(self.model_name).to_frame()
        for layer_id in range(int(model_table['_DLLayerID_'].max()) + 1):
            layer_table = model_table[model_table['_DLLayerID_'] == layer_id]
            layer_type = layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                                   'layertype'].tolist()[0]
            if layer_type == 1:
                self.layers.append(extract_input_layer(layer_table=layer_table))
            elif layer_type == 2:
                self.layers.append(extract_conv_layer(layer_table=layer_table))
            elif layer_type == 3:
                self.layers.append(extract_pooling_layer(layer_table=layer_table))
            elif layer_type == 4:
                self.layers.append(extract_fc_layer(layer_table=layer_table))
            elif layer_type == 5:
                self.layers.append(extract_output_layer(layer_table=layer_table))
            elif layer_type == 8:
                self.layers.append(extract_batchnorm_layer(layer_table=layer_table))
            elif layer_type == 9:
                self.layers.append(extract_residual_layer(layer_table=layer_table))
            elif layer_type == 10:
                self.layers.append(extract_concatenate_layer(layer_table=layer_table))
            elif layer_type == 11:
                self.layers.append(extract_detection_layer(layer_table=layer_table))

        conn_mat = model_table[['_DLNumVal_', '_DLLayerID_']][
            model_table['_DLKey1_'].str.contains('srclayers')].sort_values('_DLLayerID_')
        layer_id_list = conn_mat['_DLLayerID_'].tolist()
        src_layer_id_list = conn_mat['_DLNumVal_'].tolist()

        for row_id in range(conn_mat.shape[0]):
            layer_id = int(layer_id_list[row_id])
            src_layer_id = int(src_layer_id_list[row_id])
            if self.layers[layer_id].src_layers is None:
                self.layers[layer_id].src_layers = [self.layers[src_layer_id]]
            else:
                self.layers[layer_id].src_layers.append(self.layers[src_layer_id])

        # Check if weight table is in the same path
        _file_name_, _extension_ = os.path.splitext(file_name)

        _file_name_list_ = list(self._retrieve_('table.fileinfo',
                                                caslib=cas_lib_name,
                                                includeDirectories=False).FileInfo.Name)

        if (_file_name_ + '_weights' + _extension_) in _file_name_list_:
            print('NOTE: ' + _file_name_ + '_weights' + _extension_ +
                  ' is used as model weigths.')

            self._retrieve_('table.loadtable',
                            caslib=cas_lib_name,
                            path=_file_name_ + '_weights' + _extension_,
                            casout=dict(replace=True, name=self.model_name + '_weights'))
            self.set_weights(self.model_name + '_weights')

            if (_file_name_ + '_weights_attr' + _extension_) in _file_name_list_:
                print('NOTE: ' + _file_name_ + '_weights_attr' + _extension_ +
                      ' is used as weigths attribute.')
                self._retrieve_('table.loadtable',
                                caslib=cas_lib_name,
                                path=_file_name_ + '_weights_attr' + _extension_,
                                casout=dict(replace=True,
                                            name=self.model_name + '_weights_attr'))
                self.set_weights_attr(self.model_name + '_weights_attr')

        if cas_lib_name is not None:
            self._retrieve_('table.dropcaslib', message_level = 'error', caslib = cas_lib_name)

    def load_weights(self, path, labels=False):
        '''
        Load the weights form a data file specified by ‘path’

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the file that
            contains the weight table.

        Notes
        -----
        Currently support HDF5 and sashdat files.

        '''

        server_sep = get_server_path_sep(self.conn)

        dir_name, file_name = path.rsplit(server_sep, 1)
        if file_name.lower().endswith('.sashdat'):
            self.load_weights_from_table(path)
        elif file_name.lower().endswith('caffemodel.h5'):
            self.load_weights_from_caffe(path, labels=labels)
        elif file_name.lower().endswith('kerasmodel.h5'):
            self.load_weights_from_keras(path, labels=labels)
        elif file_name.lower().endswith('onnxmodel.h5'):
            self.load_weights_from_keras(path, labels=labels)
        else:
            raise DLPyError('Weights file must be one of the follow types:\n'
                            'sashdat, caffemodel.h5 or kerasmodel.h5.\n'
                            'Weights load failed.')

    def load_weights_from_caffe(self, path, labels=False):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.
        labels : CASTable, optional
            Specifies the table that contains the imagenet1k labels.

        '''
        if labels:
            self.load_weights_from_file_with_labels(path=path, format_type='CAFFE')
        else:
            self.load_weights_from_file(path=path, format_type='CAFFE')

    def load_weights_from_keras(self, path, labels=False):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.

        '''
        if labels:
            self.load_weights_from_file_with_labels(path=path, format_type='KERAS')
        else:
            self.load_weights_from_file(path=path, format_type='KERAS')

    def load_weights_from_file(self, path, format_type='KERAS'):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.
        format_type : KERAS, CAFFE
            Specifies the source framework for the weights file

        '''
        cas_lib_name, file_name = caslibify(self.conn, path, task='load')

        self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                        modelWeights=dict(replace=True,
                                          name=self.model_name + '_weights'),
                        formatType=format_type, weightFilePath=file_name,
                        caslib=cas_lib_name)

        self.set_weights(self.model_name + '_weights')

        if cas_lib_name is not None:
            self._retrieve_('table.dropcaslib', message_level = 'error', caslib = cas_lib_name)

    def load_weights_from_file_with_labels(self, path, format_type='KERAS'):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.
        format_type : KERAS, CAFFE
            Specifies the source framework for the weights file

        '''
        cas_lib_name, file_name = caslibify(self.conn, path, task='load')

        from dlpy.utils import get_imagenet_labels_table
        label_table = get_imagenet_labels_table(self.conn)

        self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                        modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                        formatType=format_type, weightFilePath=file_name, caslib=cas_lib_name,
                        labelTable=label_table);

        self.set_weights(self.model_name + '_weights')

        if cas_lib_name is not None:
            self._retrieve_('table.dropcaslib', message_level = 'error', caslib = cas_lib_name)

    def load_weights_from_table(self, path):
        '''
        Load the weights from a file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the file that
            contains the weight table.

        '''
        cas_lib_name, file_name = caslibify(self.conn, path, task='load')

        self._retrieve_('table.loadtable',
                        caslib=cas_lib_name,
                        path=file_name,
                        casout=dict(replace=True, name=self.model_name + '_weights'))

        self.set_weights(self.model_name + '_weights')

        _file_name_, _extension_ = os.path.splitext(file_name)

        _file_name_list_ = list(
            self._retrieve_('table.fileinfo', caslib=cas_lib_name,
                            includeDirectories=False).FileInfo.Name)

        if (_file_name_ + '_attr' + _extension_) in _file_name_list_:
            print('NOTE: ' + _file_name_ + '_attr' + _extension_ +
                  ' is used as weigths attribute.')
            self._retrieve_('table.loadtable',
                            caslib=cas_lib_name,
                            path=_file_name_ + '_attr' + _extension_,
                            casout=dict(replace=True,
                                        name=self.model_name + '_weights_attr'))

            self.set_weights_attr(self.model_name + '_weights_attr')

        self.model_weights = self.conn.CASTable(name=self.model_name + '_weights')

        if cas_lib_name is not None:
            self._retrieve_('table.dropcaslib', message_level = 'error', caslib = cas_lib_name)

    def set_weights_attr(self, attr_tbl, clear=True):
        '''
        Attach the weights attribute to the model weights

        Parameters
        ----------
        attr_tbl : CASTable or string or dict
            Specifies the CAS table that contains the weights attribute table
        clear : bool, optional
            Specifies whether to drop the attribute table after attach it
            into the weight table.

        '''
        self._retrieve_('table.attribute',
                        task='ADD', attrtable=attr_tbl,
                        **self.model_weights.to_table_params())

        if clear:
            self._retrieve_('table.droptable',
                            table=attr_tbl)

        print('NOTE: Model attributes attached successfully!')

    def load_weights_attr(self, path):
        '''
        Load the weights attribute form a sashdat file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the file that
            contains the weight attribute table.

        '''
        server_sep = get_server_path_sep(self.conn)
        dir_name, file_name = path.rsplit(server_sep, 1)
        try:
            flag, cas_lib_name = check_caslib(self.conn, dir_name)
        except:
            flag = False
            cas_lib_name = random_name('Caslib', 6)
            self._retrieve_('table.addcaslib',
                            name=cas_lib_name, path=dir_name,
                            activeOnAdd=False, dataSource=dict(srcType='DNFS'))

        self._retrieve_('table.loadtable',
                        caslib=cas_lib_name,
                        path=file_name,
                        casout=dict(replace=True,
                                    name=self.model_name + '_weights_attr'))

        self.set_weights_attr(self.model_name + '_weights_attr')

        if not flag:
            self._retrieve_('table.dropcaslib', caslib=cas_lib_name)

    def save_to_astore(self, path = None, **kwargs):
        """
        Save the model to an astore object, and write it into a file.

        Parameters
        ----------
        path : string
            Specifies the client-side path to store the model astore.
            The path format should be consistent with the system of the client.

        """
        self.conn.loadactionset('astore', _messagelevel = 'error')

        CAS_tbl_name = self.model_name + '_astore'

        self._retrieve_('deeplearn.dlexportmodel',
                        casout = dict(replace = True, name = CAS_tbl_name),
                        initWeights = self.model_weights,
                        modelTable = self.model_table,
                        randomCrop = 'none',
                        randomFlip = 'none',
                        **kwargs)

        model_astore = self._retrieve_('astore.download',
                                       rstore = CAS_tbl_name)

        file_name = self.model_name + '.astore'
        if path is None:
            path = os.getcwd()

        if not os.path.isdir(path):
            os.makedirs(path)

        file_name = os.path.join(path, file_name)
        with open(file_name, 'wb') as file:
            file.write(model_astore['blob'])
        print('NOTE: Model astore file saved successfully.')

    def save_to_table(self, path):
        """
        Save the model as SAS dataset

        Parameters
        ----------
        path : string
            Specifies the server-side path to store the model tables.

        """
        self.save_to_table_with_caslibify(path)

    def save_to_table_with_caslibify(self, path):
        """
        Save the model as SAS dataset

        Parameters
        ----------
        path : string
            Specifies the server-side path to store the model tables.

        """
        # import os
        # if path.endswith(os.path.sep):
        #    path = path[:-1]

        caslib, path_remaining = caslibify(self.conn, path, task = 'save')

        _file_name_ = self.model_name.replace(' ', '_')
        _extension_ = '.sashdat'
        model_tbl_file = path_remaining + _file_name_ + _extension_
        weight_tbl_file = path_remaining + _file_name_ + '_weights' + _extension_
        attr_tbl_file = path_remaining + _file_name_ + '_weights_attr' + _extension_

        if self.model_table is not None:
            ch = self.conn.table.tableexists(self.model_weights)
            if ch.exists > 0:
                rt = self._retrieve_('table.save', table = self.model_table, name = model_tbl_file, replace = True,
                                     caslib = caslib)
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('something is wrong while saving the model to a table!')
        if self.model_weights is not None:
            ch = self.conn.table.tableexists(self.model_weights)
            if ch.exists > 0:
                rt = self._retrieve_('table.save', table = self.model_weights, name = weight_tbl_file,
                                     replace = True, caslib = caslib)
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('something is wrong while saving the model weights to a table!')

                CAS_tbl_name = random_name('Attr_Tbl')
                rt = self._retrieve_('table.attribute', task = 'convert', attrtable = CAS_tbl_name,
                                     **self.model_weights.to_table_params())
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('something is wrong while extracting the model attributes!')

                rt = self._retrieve_('table.save', table = CAS_tbl_name, name = attr_tbl_file, replace = True,
                                     caslib = caslib)
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('something is wrong while saving the model attributes to a table!')

        print('NOTE: Model table saved successfully.')

        if caslib is not None:
            self._retrieve_('table.dropcaslib', message_level = 'error', caslib = caslib)

    def save_weights_csv(self, path):
        '''
        Save model weights table as csv

        Parameters
        ----------
        path : string
            Specifies the server-side path to store the model
            weights csv.

        '''
        weights_table_opts = input_table_check(self.model_weights)
        weights_table_opts.update(**dict(groupBy = '_LayerID_',
                                         groupByMode = 'REDISTRIBUTE',
                                         orderBy = '_WeightID_'))
        self.conn.partition(table = weights_table_opts,
                            casout = dict(name = self.model_weights.name,
                                          replace = True))

        caslib, path_remaining = caslibify(self.conn, path, task = 'save')
        _file_name_ = self.model_name.replace(' ', '_')
        _extension_ = '.csv'
        weights_tbl_file = path_remaining + _file_name_ + '_weights' + _extension_
        rt = self._retrieve_('table.save', table = weights_table_opts,
                             name = weights_tbl_file, replace = True, caslib = caslib)
        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError('something is wrong while saving the the model to a table!')

        print('NOTE: Model weights csv saved successfully.')

        if caslib is not None:
            self._retrieve_('table.dropcaslib', message_level = 'error', caslib = caslib)

    def save_to_onnx(self, path, model_weights = None):
        '''
        Save to ONNX model

        Parameters
        ----------
        path : string
            Specifies the client-side path to save the ONNX model.
        model_weights : string, optional
            Specifies the client-side path of the csv file of the
            model weights table.  If no csv file is specified, the
            weights will be fetched from the CAS server.  This can
            take a long time to complete if the size of model weights
            is large.

        '''

        from .model_conversion.write_onnx_model import sas_to_onnx
        if model_weights is None:
            try:
                self.model_weights.numrows()
            except:
                raise DLPyError('No model weights yet. Please load weights or'
                                ' train the model first.')
            print('NOTE: Model weights will be fetched from server')
            model_weights = self.model_weights
        else:
            print('NOTE: Model weights will be loaded from csv.')
            model_weights = pd.read_csv(model_weights)
        model_table = self.conn.CASTable(**self.model_table)
        onnx_model = sas_to_onnx(layers = self.layers,
                                 model_table = model_table,
                                 model_weights = model_weights)
        file_name = self.model_name + '.onnx'
        if path is None:
            path = os.getcwd()

        if not os.path.isdir(path):
            os.makedirs(path)

        file_name = os.path.join(path, file_name)

        with open(file_name, 'wb') as f:
            f.write(onnx_model.SerializeToString())

        print('NOTE: ONNX model file saved successfully.')

    def deploy(self, path, output_format = 'astore', model_weights = None, **kwargs):
        """
        Deploy the deep learning model to a data file

        Parameters
        ----------
        path : string
            Specifies the client-side path to store the model files.
        output_format : string, optional
            Specifies the format of the deployed model
            Valid Values: astore, castable, or onnx
            Default: astore
        model_weights : string, optional
            Specifies the client-side path to the csv file of the
            model weights table.  Only effective when
            output_format='onnx'.  If no csv file is specified when
            deploying to ONNX, the weights will be fetched from the
            CAS server.  This may take a long time to complete if
            the size of model weights is large.

        Notes
        -----
        Currently, this function supports sashdat, astore, and onnx formats.

        More information about ONNX can be found at: https://onnx.ai/

        DLPy supports ONNX version >= 1.3.0, and Opset version 8.

        For ONNX format, currently supported layers are convo, pool,
        fc, batchnorm, residual, concat, and detection.

        If dropout is specified in the model, train the model using
        inverted dropout, which can be specified in :class:`Optimizer`.
        This will ensure the results are correct when running the
        model during test phase.


        """
        if output_format.lower() == 'astore':
            self.save_to_astore(path = path, **kwargs)
        elif output_format.lower() in ('castable', 'table'):
            self.save_to_table(path = path)
        elif output_format.lower() == 'onnx':
            self.save_to_onnx(path, model_weights = model_weights)
        else:
            raise DLPyError('output_format must be "astore", "castable", "table",'
                            'or "onnx"')

    def count_params(self):
        ''' Count the total number of parameters in the model '''
        count = 0
        for layer in self.layers:

            if layer.num_weights is None:
                num_weights = 0
            else:
                num_weights = layer.num_weights

            if layer.num_bias is None:
                num_bias = 0
            else:
                num_bias = layer.num_bias

            count += num_weights + num_bias
        return int(count)


def layer_to_node(layer):
    '''
    Convert layer configuration to a node in the model graph

    Parameters
    ----------
    layer : Layer object
        Specifies the layer to be converted.

    Returns
    -------
    dict
        Options that can be passed to graph configuration.

    '''
    if layer.type == 'recurrent':
        label = '%s(%s)' % (layer.name, layer.type)
    else:
        if layer.kernel_size:
            label = '%s %s(%s)' % ('x'.join('%s' % x for x in layer.kernel_size), layer.name, layer.type)
        elif layer.output_size:
            if not isinstance(layer.output_size, collections.Iterable):
                label = '%s %s(%s)' % (layer.output_size, layer.name, layer.type)
            else:
                label = '%s %s(%s)' % ('x'.join('%s' % x for x in layer.output_size), layer.name, layer.type)
        else:
            label = '%s(%s)' % (layer.name, layer.type)

    if isinstance(layer.color_code, (list, tuple)):
        fg = layer.color_code[0]
        bg = layer.color_code[1]
    else:
        fg = layer.color_code[:7]
        bg = layer.color_code

    return dict(name = layer.name, label = ' %s ' % label,
                fillcolor = bg, color = fg, margin = '0.2,0.0', height = '0.3')


def layer_to_edge(layer):
    '''
    Convert layer to layer connection to an edge in the model graph

    Parameters
    ----------
    layer : Layer object
        Specifies the layer to be converted.

    Returns
    -------
    dict
        Options that can be passed to graph configuration.

    '''
    gv_params = []
    for item in layer.src_layers:
        label = ''
        if layer.type is not 'input':
            if isinstance(item.output_size, (tuple, list)):
                label = ' %s ' % ' x '.join('%s' % x for x in item.output_size)
            else:
                label = ' %s ' % item.output_size
        gv_params.append(dict(label = label, tail_name = '{}'.format(item.name),
                              head_name = '{}'.format(layer.name)))

    if layer.type == 'recurrent':
        gv_params.append(dict(label = '', tail_name = '{}'.format(layer.name),
                              head_name = '{}'.format(layer.name)))
    return gv_params


def model_to_graph(model):
    '''
    Convert model configuration to a graph

    Parameters
    ----------
    model : Model object
        Specifies the model to be converted.

    Returns
    -------
    :class:`graphviz.dot.Digraph`

    '''
    import graphviz as gv

    model_graph = gv.Digraph(name = model.model_name,
                             node_attr = dict(shape = 'record', style = 'filled', fontname = 'helvetica'),
                             edge_attr = dict(fontname = 'helvetica', fontsize = '10'))
    # can be added later for adjusting figure size.
    # fixedsize='True', width = '4', height = '1'))

    #   model_graph.attr(label=r'DAG for {}:'.format(model.model_name),
    #                    labelloc='top', labeljust='left')
    #   model_graph.attr(fontsize='16')

    for layer in model.layers:
        if layer.type == 'input':
            model_graph.node(**layer_to_node(layer))
        else:
            model_graph.node(**layer_to_node(layer))
            for gv_param in layer_to_edge(layer):
                model_graph.edge(color = '#5677F3', **gv_param)

    return model_graph


def get_num_configs(keys, layer_type_prefix, layer_table):
    '''
    Extract the numerical options from the model table

    Parameters
    ----------
    keys : list-of-strings
        Specifies the list of numerical variables
    layer_type_prefix : string
        Specifies the prefix of the options in the model table
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    dict
        Options that can be passed to layer definition

    '''
    layer_config = dict()
    for key in keys:
        try:
            layer_config[key] = layer_table['_DLNumVal_'][
                layer_table['_DLKey1_'] == layer_type_prefix + '.' +
                key.lower().replace('_', '')].tolist()[0]
        except IndexError:
            pass
    return layer_config


def get_str_configs(keys, layer_type_prefix, layer_table):
    '''
    Extract the str options from the model table

    Parameters
    ----------
    keys : list-of-strings
        Specifies the list of str variables.
    layer_type_prefix : string
        Specifies the prefix of the options in the model table.
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    dict
        Options that can be passed to layer definition.

    '''
    layer_config = dict()
    for key in keys:
        try:
            layer_config[key] = layer_table['_DLChrVal_'][
                layer_table['_DLKey1_'] == layer_type_prefix + '.' +
                key.lower().replace('_', '')].tolist()[0]
        except IndexError:
            pass
    return layer_config


def extract_input_layer(layer_table):
    '''
    Extract layer configuration from an input layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    dict
        Options that can be passed to layer definition

    '''
    num_keys = ['n_channels', 'width', 'height', 'dropout', 'scale']
    input_layer_config = dict()
    input_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]
    input_layer_config.update(get_num_configs(num_keys, 'inputopts', layer_table))

    input_layer_config['offsets'] = []
    try:
        input_layer_config['offsets'].append(
            int(layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                          'inputopts.offsets'].tolist()[0]))
    except IndexError:
        pass
    try:
        input_layer_config['offsets'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                      'inputopts.offsets.0'].tolist()[0])
    except IndexError:
        pass
    try:
        input_layer_config['offsets'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                      'inputopts.offsets.1'].tolist()[0])
    except IndexError:
        pass
    try:
        input_layer_config['offsets'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                      'inputopts.offsets.2'].tolist()[0])
    except IndexError:
        pass

    if layer_table['_DLChrVal_'][layer_table['_DLKey1_'] ==
                                 'inputopts.crop'].tolist()[0] == 'No cropping':
        input_layer_config['random_crop'] = 'none'
    else:
        input_layer_config['random_crop'] = 'unique'

    if layer_table['_DLChrVal_'][layer_table['_DLKey1_'] ==
                                 'inputopts.flip'].tolist()[0] == 'No flipping':
        input_layer_config['random_flip'] = 'none'
    # else:
    #     input_layer_config['random_flip']='hv'

    layer = InputLayer(**input_layer_config)
    return layer


def extract_conv_layer(layer_table):
    '''
    Extract layer configuration from a convolution layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    dict
        Options that can be passed to layer definition

    '''
    num_keys = ['n_filters', 'width', 'height', 'stride', 'std', 'mean',
                'init_bias', 'dropout', 'truncation_factor', 'init_b', 'trunc_fact']
    str_keys = ['act', 'init']

    conv_layer_config = dict()
    conv_layer_config.update(get_num_configs(num_keys, 'convopts', layer_table))
    conv_layer_config.update(get_str_configs(str_keys, 'convopts', layer_table))
    conv_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if 'trunc_fact' in conv_layer_config.keys():
        conv_layer_config['truncation_factor'] = conv_layer_config['trunc_fact']
        del conv_layer_config['trunc_fact']
    if conv_layer_config.get('act') == 'Leaky Activation function':
        conv_layer_config['act'] = 'Leaky'

    dl_numval = layer_table['_DLNumVal_']
    if dl_numval[layer_table['_DLKey1_'] == 'convopts.no_bias'].any():
        conv_layer_config['include_bias'] = False
    else:
        conv_layer_config['include_bias'] = True

    padding_width = dl_numval[layer_table['_DLKey1_'] == 'convopts.pad_left'].tolist()[0]
    padding_height = dl_numval[layer_table['_DLKey1_'] == 'convopts.pad_top'].tolist()[0]
    if padding_width != -1:
        conv_layer_config['padding_width'] = padding_width
    if padding_height != -1:
        conv_layer_config['padding_height'] = padding_height

    layer = Conv2d(**conv_layer_config)
    return layer


def extract_pooling_layer(layer_table):
    '''
    Extract layer configuration from a pooling layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    dict
        Options that can be passed to layer definition

    '''
    num_keys = ['width', 'height', 'stride']
    str_keys = ['act', 'poolingtype']

    pool_layer_config = dict()
    pool_layer_config.update(get_num_configs(num_keys, 'poolingopts', layer_table))
    pool_layer_config.update(get_str_configs(str_keys, 'poolingopts', layer_table))

    pool_layer_config['pool'] = pool_layer_config['poolingtype'].lower().split(' ')[0]
    del pool_layer_config['poolingtype']
    pool_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    padding_width = layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'poolingopts.pad_left'].tolist()[0]
    padding_height = layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'poolingopts.pad_top'].tolist()[0]
    if padding_width != -1:
        pool_layer_config['padding_width'] = padding_width
    if padding_height != -1:
        pool_layer_config['padding_height'] = padding_height

    layer = Pooling(**pool_layer_config)
    return layer


def extract_batchnorm_layer(layer_table):
    '''
    Extract layer configuration from a batch normalization layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    dict
        Options that can be passed to layer definition

    '''
    bn_layer_config = dict()
    bn_layer_config.update(get_str_configs(['act'], 'bnopts', layer_table))
    bn_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]
    if bn_layer_config.get('act') == 'Leaky Activation function':
        bn_layer_config['act'] = 'Leaky'

    layer = BN(**bn_layer_config)
    return layer


def extract_residual_layer(layer_table):
    '''
    Extract layer configuration from a residual layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    dict
        Options that can be passed to layer definition

    '''

    res_layer_config = dict()

    res_layer_config.update(get_str_configs(['act'], 'residualopts', layer_table))
    res_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = Res(**res_layer_config)
    return layer


def extract_concatenate_layer(layer_table):
    '''
    Extract layer configuration from a concatenate layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    dict
        Options that can be passed to layer definition

    '''

    concat_layer_config = dict()

    concat_layer_config.update(get_str_configs(['act'], 'residualopts', layer_table))
    concat_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = Concat(**concat_layer_config)
    return layer


def extract_detection_layer(layer_table):
    detection_layer_config = dict()

    detection_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = Detection(**detection_layer_config)
    return layer


def extract_fc_layer(layer_table):
    '''
    Extract layer configuration from a fully connected layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    dict
        Options that can be passed to layer definition

    '''
    num_keys = ['n', 'width', 'height', 'stride', 'std', 'mean',
                'init_bias', 'dropout', 'truncation_factor', 'init_b', 'trunc_fact']
    str_keys = ['act', 'init']

    fc_layer_config = dict()
    fc_layer_config.update(get_num_configs(num_keys, 'fcopts', layer_table))
    fc_layer_config.update(get_str_configs(str_keys, 'fcopts', layer_table))
    fc_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'fcopts.no_bias'].any():
        fc_layer_config['include_bias'] = False
    else:
        fc_layer_config['include_bias'] = True

    if 'trunc_fact' in fc_layer_config.keys():
        fc_layer_config['truncation_factor'] = fc_layer_config['trunc_fact']
        del fc_layer_config['trunc_fact']
    if fc_layer_config.get('act') == 'Leaky Activation function':
        fc_layer_config['act'] = 'Leaky'

    layer = Dense(**fc_layer_config)
    return layer


def extract_output_layer(layer_table):
    '''
    Extract layer configuration from an output layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    dict
        Options that can be passed to layer definition

    '''
    num_keys = ['n', 'width', 'height', 'stride', 'std', 'mean',
                'init_bias', 'dropout', 'truncation_factor', 'init_b', 'trunc_fact']
    str_keys = ['act', 'init']

    output_layer_config = dict()
    output_layer_config.update(get_num_configs(num_keys, 'outputopts', layer_table))
    output_layer_config.update(get_str_configs(str_keys, 'outputopts', layer_table))
    output_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'outputopts.no_bias'].any():
        output_layer_config['include_bias'] = False
    else:
        output_layer_config['include_bias'] = True

    if 'trunc_fact' in output_layer_config.keys():
        output_layer_config['truncation_factor'] = output_layer_config['trunc_fact']
        del output_layer_config['trunc_fact']

    layer = OutputLayer(**output_layer_config)
    return layer


def extract_keypoints_layer(layer_table):
    # TODO
    keypoints_layer_config = dict()
    keypoints_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]
    layer = Keypoints(**keypoints_layer_config)
    return layer