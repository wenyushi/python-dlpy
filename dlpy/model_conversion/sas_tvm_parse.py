#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

''' Convert TVM models to SAS models '''

import os
import json
import re

import h5py
import numpy as np

from dlpy import layers

TVM_OP = ['nn_leaky_relu', 'nn_relu', 'add', 'nn_relu', 'nn_conv2d_transpose', 'nn_max_pool2d', 'nn_conv2d', 'concatenate']
ACT_OP = ['nn_leaky_relu', 'nn_relu']

# mapping ONNX ops to SAS activations
_act_map = {
    'nn_relu': 'RELU',
    'nn_leaky_relu': 'LEAKY',
    'Log': 'LOGISTIC',
    'Softmax': 'SOFTMAX',
    'Identity': 'IDENTITY',
    'Cos': 'COS',
    'Sin': 'SIN',
    'Exp': 'EXP',
    'Elu': 'ELU',
    'Softplus': 'SOFTPLUS'
}


class TVMParseError(ValueError):
    '''
    Used to indicate an error in parsing TVM model definition

    '''


def meta_fusion(name, operators, t_size, output_size):
    '''return layer type and layer options'''
    op_configs = dict()
    # op_configs['name'] = name
    input_size = t_size[0]
    operator_type = operators[0] if len(operators) != 0 else None

    if operator_type in ['nn_conv2d', 'nn_conv2d_transpose']:
        transpose = False if operator_type == 'nn_conv2d' else True
        op_configs['type'] = 'convo'
            
        # the second op can be bias addition or activation
        kernel_size = t_size[1]
        op_configs['n_filters'] = kernel_size[1] if transpose else kernel_size[0]
        op_configs['height'] = kernel_size[-2]
        op_configs['width'] = kernel_size[-1]
        op_configs['stride'] = None

        if transpose:
            if output_size[-1] % input_size[-1] != 0:
                raise TVMParseError('{}: Only support same padding.'.format(name))
            else:
                op_configs['stride_horizontal'] = int(output_size[-1] / input_size[-1])

            if output_size[-2] % input_size[-2] != 0:
                raise TVMParseError('{}: Only support same padding.'.format(name))
            else:
                op_configs['stride_vertical'] = int(output_size[-2] / input_size[-2])
            
            if op_configs['width'] % 2:
                print('WARNING: Horizontal padding is set as (kernel width - 1) / 2.')
                op_configs['padding_width'] = (op_configs['width'] - 1) / 2
            else:
                raise TVMParseError('{}: Kernel shape has to be odd.'.format(name))
                
            if op_configs['height'] % 2:
                print('WARNING: Vertical padding is set as (kernel height - 1) / 2.')
                op_configs['padding_height'] = (op_configs['height'] - 1) / 2
            else:
                raise TVMParseError('{}: Kernel shape has to be odd.'.format(name))
            
            stride = (op_configs['stride_vertical'], op_configs['stride_horizontal'])
            kernel_size = (op_configs['height'], op_configs['width'])
            padding = (op_configs['padding_height'], op_configs['padding_width'] )
            input_map_size = input_size[2:]
            output_map_size = output_size[2:]
            min_sizes = [(input_map_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[i] for i in range(2)]
            op_configs['output_padding_height'], op_configs['output_padding_width'] = tuple([output_map_size[i] - min_sizes[i] for i in range(2)])
        else:
            if input_size[-1] % output_size[-1] != 0:
                raise TVMParseError('{}: Only support same padding.'.format(name))
            else:
                op_configs['stride_horizontal'] = int(input_size[-1] / output_size[-1])

            if input_size[-2] % output_size[-2] != 0:
                raise TVMParseError('{}: Only support same padding.'.format(name))
            else:
                op_configs['stride_vertical'] = int(input_size[-2] / output_size[-2])

        # check ops order after conv2d is fusible
        if len(operators) == 1:
            op_configs['act'] = 'identity'
            op_configs['include_bias'] = False
        elif len(operators) == 2:
            if operators not in ['add'] + ACT_OP:
                raise TVMParseError('{}: Fail to fuse a conv2d with another one operator.'.format(name))
        elif len(operators) == 3:
            if operators[1] != 'add' or operators[2] not in ACT_OP:
                raise TVMParseError('{}: Fail to fuse a conv2d with another two operators.'.format(name))
        else:
            raise TVMParseError('{}: Too many operators to fuse in a conv2d.'.format(name))
        for op in operators[1:]:
            if op == 'add':
                bias_shape = t_size[2]
                if bias_shape[1] != op_configs['n_filters']:
                    raise TVMParseError('{}: Bias shape doesn\'t match number'
                                        ' of convolution kernel filter.'.format(name))
                elif bias_shape[-2] == 1 and bias_shape[-1] == 1:
                    op_configs['include_bias'] = True
                else:
                    raise TVMParseError('{}: Fail to fuse bias into convolution. '
                                        'Please check bias parameter.'.format(name))
            # check activation
            if op in ACT_OP:
                op_configs['act'] = _act_map[op]
    elif operator_type == 'nn_max_pool2d':
        op_configs['type'] = 'pool'
        op_configs['pool'] = 'max'
        op_configs['height'] = 2
        op_configs['width'] = 2
        print('WARNING: Assume MaxPooling2D has kernel size as 2 by 2.')
        # if input_size[-1] % output_size[-1] != 0:
        #     raise TVMParseError('{}: Only support same padding.'.format(name))
        # else:
        op_configs['stride'] = None
        op_configs['stride_horizontal'] = int(round(input_size[-1] / output_size[-1]))

        # if input_size[-2] % output_size[-2] != 0:
        #     raise TVMParseError('{}: Only support same padding.'.format(name))
        # else:
        op_configs['stride_vertical'] = int(round(input_size[-2] / output_size[-2]))
    elif operator_type == 'concatenate':
        op_configs['type'] = 'concat'
    else:
        TVMParseError('{} contains unsupported layer type.'.format(name))

    return op_configs


def build_graph(conn, model_table, tvm_graph, input_layers, output_layer=None):
    # model = Sequential(conn=conn, model_table=model_table)
    dlpy_layers = []

    nodes = tvm_graph['nodes']
    arg_nodes = tvm_graph['arg_nodes']
    attrs = tvm_graph['attrs']
    shape = attrs['shape'][1]

    layer_mapper = dict()

    def search_ops(f_name):
        ''' search operators in the node '''
        if f_name in TVM_OP and f_name not in ops:
            for op in ops:
                if op.startswith(f_name):
                    return
            ops.append(f_name)
            return
        for o in TVM_OP:
            # avoid duplicate eg. nn_conv2d_transpose and nn_conv2d
            l = re.split(r"({}|\(|\))".format(o), f_name)
            l = [x for x in l if x != '']
            if len(l) == 1 and l[0] == f_name:
                continue
            for i in l:
                search_ops(i)

    for idx, node in enumerate(nodes):
        if node['op'] == 'null' and node['name'] not in input_layers:
            continue
        else:
            name = node['name']
            # source node indices
            src_nodes = [inp[0] for inp in node['inputs']]
            # source tensor sizes
            t_size = [shape[n] for n in src_nodes]  # a list of lists
            out_size = shape[idx]
            # check if the node can be converted to an input layer
            if len(src_nodes) == 1 and nodes[src_nodes[0]]['op'] == 'null':
                t_size = t_size[0]
                layer = layers.InputLayer(name=name, height=t_size[-2], width=t_size[-1],
                                          n_channels=t_size[-3], norm_stds=None, offsets=None)
            elif len(src_nodes) == 0:
                arg_nodes.remove(idx)
                layer = layers.InputLayer(name=name, height=out_size[-2], width=out_size[-1], n_channels=out_size[-3],
                                          norm_stds=None, offsets=None)
            else:
                op_name = name[6:] if name.startswith('fused_') else name
                ops = []
                search_ops(op_name)
                # check if ops can be fusion for SAS DL
                op_dict = meta_fusion(name, ops, t_size, out_size)
                # get source layers
                src_layers = [layer_mapper[nodes[sn]['name']] for sn in src_nodes if sn not in arg_nodes]
                # create layer
                for l in layers.Layer.__subclasses__():
                    if l.type == op_dict['type']:
                        del op_dict['type']
                        # if it is _Conv, assume it is Conv2d
                        if l == layers._Conv:
                            if 'output_padding_height' in op_dict:
                                layer = layers.Conv2DTranspose(name=name, src_layers=src_layers, **op_dict)
                            else:
                                layer = layers.Conv2d(name=name, src_layers=src_layers, **op_dict)
                        else:
                            layer = l(name=name, src_layers=src_layers, **op_dict)
                        break

            layer_mapper[name] = layer
            dlpy_layers.append(layer)
            # model.add(layer)

    if output_layer:
        output_layer.src_layers = [dlpy_layers[-1]]
        dlpy_layers.append(output_layer)
        # model.add(output_layer)

    return dlpy_layers


def write_weights_hdf5(dlpy_layers, graph, tensor_dict, name):
    '''
    Write SAS compatible HDF5 weights file

    Parameters
    ----------
    layers : list of Layers
        Specifies the layers of the model.
    graph : json
        Specifies a GraphProto object.
    tensor_dict : dict of numpy.ndarray
        Specifies the dictionary of weight tensors.
    name : string
        Specifies the name of the model.

    '''
    temp_HDF5 = os.path.join(os.getcwd(), '{}_weights.tvm.h5'.format(name))
    f_out = h5py.File(temp_HDF5, 'w')
    weight_layers = [l for l in dlpy_layers if l.type in ['convo', 'fc', 'batchnorm', 'groupconvo', 'transconvo']]
    f_out.attrs['layer_names'] = [l.name.encode('utf8') for l in weight_layers]

    nodes = graph['nodes']
    # arg_nodes = graph['arg_nodes']
    # heads = graph['heads']
    # attrs = graph['attrs']
    # shape = attrs['shape'][1]

    for layer in weight_layers:
        new_weight_names = []
        g_out = f_out.create_group(layer.name)
        # find weights params name list
        # find layer's node in tvm_graph
        node = [n for n in nodes if n['name'] == layer.name][0]
        # get inputs node indices
        src_nodes = [int(inp[0]) for inp in node['inputs']]  # idx value
        # get input node name
        params_name = [nodes[s]['name'] for s in src_nodes]  # eg. p0, p1, nn_conv2d
        weights = []
        for p in params_name:
            if p in tensor_dict.keys():
                weights.append(tensor_dict[p])
        # weights = [v for k, v in tensor_dict.items() if k in params_name]

        if layer.type in ['convo', 'fc', 'groupconvo', 'transconvo']:
            # check bias op following the node
            # to see if we need to include any bias weights
            if layer.config['include_bias']:
                weights[-1] = weights[-1].flatten()
            # for n in onnx_get_out_nodes(graph, node):
            #     if is_bias_op(graph, n):
            #         for i in n.input:
            #             if tensor_dict.get(i) is not None:
            #                 weights.append(tensor_dict[i].flatten())
            for w in weights:
                if len(w.shape) > 1:
                    dset_name = layer.name + '/' + 'kernel:0'
                    # check if need to transpose fc weight
                    if len(w.shape) == 2:
                        # check if transposed was specified in Gemm op
                        if node.op_type == 'Gemm':
                            for attr in node.attribute:
                                if attr.name == 'transB':
                                    if attr.i == 1:
                                        w = np.transpose(w, (1, 0))
                        if w.shape[1] == layer.config['n']:
                            w = np.transpose(w, (1, 0))
                    g_out.create_dataset(dset_name.encode('utf8'), data=w)
                    new_weight_names.append(dset_name.encode('utf8'))
                else:
                    dset_name = layer.name + '/' + 'bias:0'
                    g_out.create_dataset(dset_name.encode('utf8'), data=w)
                    new_weight_names.append(dset_name.encode('utf8'))
        # elif layer.type == 'batchnorm':
        #     template_names = ['gamma:0', 'beta:0', 'moving_mean:0',
        #                       'moving_variance:0']
        #     template_names = [layer.name + '/' + i for i in template_names]
        #     if len(weights) != 4:
        #         raise OnnxParseError('Incorrect batchnorm weights')
        #     for idx, w in enumerate(weights):
        #         if idx == 3:
        #             # clip variance to avoid error on cas
        #             w = np.clip(w, a_min = 1e-12, a_max = 1e10)
        #             g_out.create_dataset(template_names[idx].encode('utf8'), data = w)
        #             new_weight_names.append(template_names[idx].encode('utf8'))
        #         else:
        #             g_out.create_dataset(template_names[idx].encode('utf8'), data = w)
        #             new_weight_names.append(template_names[idx].encode('utf8'))

        g_out.attrs['weight_names'] = new_weight_names

    f_out.close()
    print('NOTE: Successfully written weights file as '
          + temp_HDF5)
    return temp_HDF5


def save_params_dict(params_dict, save_to):
    if not save_to.endswith('.npy'):
        raise ValueError('save_to should be a path with .npy extension.')
    for key, item in params_dict.items():
        params_dict[key] = item.asnumpy()
    np.save(save_to, params_dict)


def tvm_to_sas(conn, model_table, tvm_graph, tvm_params_dict, input_layers, output_layer=None):
    '''
    Generate SAS model from TVM model

    Parameters
    ----------
    model : TVM
        Specifies the loaded ONNX model.
    model_name : string, optional
        Specifies the name of the model.
    output_layer : Layer object, optional
        Specifies the output layer of the model. If no output
        layer is specified, the last layer is automatically set
        as :class:`OutputLayer` with SOFTMAX activation.

    Returns
    -------
    list
        List of Layers

    '''
    # graph_params_path = "Tiny_yolov2_params.params"
    # with open(graph_params_path, 'wb') as fo:
    #     fo.write(relay.save_param_dict(params_d))
    from dlpy import Model
    from dlpy.utils import DLPyError

    dlpy_layers = build_graph(conn, model_table, tvm_graph, input_layers,output_layer)

    write_weights_hdf5(dlpy_layers, tvm_graph, tvm_params_dict, model_table)

    conn.loadactionset('deeplearn', _messagelevel = 'error')
    rt = conn.retrieve('deeplearn.buildmodel',
                       _messagelevel = 'error',
                       model = dict(name = model_table, replace = True),
                       type = 'CNN')
    if rt.severity > 1:
        for msg in rt.messages:
            print(msg)
        raise DLPyError('Cannot build model, there seems to be a problem.')

    for layer in dlpy_layers:
        option = layer.to_model_params()
        rt = conn.retrieve('deeplearn.addlayer', _messagelevel = 'error',
                           model = model_table, **option)
        if rt.severity > 1:
            for m in rt.messages:
                print(m)
            raise DLPyError('There seems to be an error while adding the '
                            + layer.name + '.')

    input_model_table = conn.CASTable(name=model_table)
    model = Model.from_table(input_model_table = input_model_table)
    print('NOTE: Successfully imported TVM model.')
    return model

