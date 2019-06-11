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

''' Pre-built deep learning models '''

import os
import warnings

from .sequential import Sequential
from .blocks import ResBlockBN, ResBlock_Caffe, DenseNetBlock, Bidirectional
from .caffe_models import (model_vgg16, model_vgg19, model_resnet50,
                           model_resnet101, model_resnet152)
from .keras_models import model_inceptionv3
from .layers import (Input, InputLayer, Conv2d, Pooling, Dense, BN, OutputLayer, Detection, Concat, Reshape, Recurrent,
                     Conv2DTranspose, Segmentation, RegionProposal, ROIPooling, FastRCNN, GroupConv2d, ChannelShuffle,
                     Res, GlobalAveragePooling2D)
from .model import Model
from .utils import random_name, DLPyError


def TextClassification(conn, model_table='text_classifier', neurons=10, n_blocks=3, rnn_type='gru'):
    '''
    Generates a text classification model

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    neurons : int, optional
        Specifies the number of neurons to be in each layer.
        Default: 10
    n_blocks : int, optional
        Specifies the number of bidirectional blocks to be added to the model.
        Default: 3
    rnn_type : string, optional
        Specifies the type of the rnn layer.
        Default: RNN
        Valid Values: RNN, LSTM, GRU

    Returns
    -------
    :class:`Sequential`

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if n_blocks >= 2:
        model = Sequential(conn=conn, model_table=model_table)
        b = Bidirectional(n=neurons, name='bi_'+rnn_type+'_layer_', n_blocks=n_blocks-1, rnn_type=rnn_type)
        model.add(b)
        model.add(Bidirectional(n=neurons, output_type='encoding', src_layers=b.get_last_layers(), rnn_type=rnn_type,
                                name='bi_'+rnn_type+'_lastlayer_',))
        model.add(OutputLayer())
    elif n_blocks == 1:
        model = Sequential(conn=conn, model_table=model_table)
        model.add(Bidirectional(n=neurons, output_type='encoding', rnn_type=rnn_type))
        model.add(OutputLayer())
    else:
        raise DLPyError('The number of blocks for a text classification model should be at least 1.')

    return model


def TextGeneration(conn, model_table='text_generator', neurons=10, max_output_length=15, n_blocks=3, rnn_type='gru'):
    '''
    Generates a text generation model.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    neurons : int, optional
        Specifies the number of neurons to be in each layer.
        Default: 10
    n_blocks : int, optional
        Specifies the number of bidirectional blocks to be added to the model.
        Default: 3
    max_output_length : int, optional
        Specifies the maximum number of tokens to generate
        Default: 15
    rnn_type : string, optional
        Specifies the type of the rnn layer.
        Default: RNN
        Valid Values: RNN, LSTM, GRU

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if n_blocks >= 3:
        model = Sequential(conn=conn, model_table=model_table)
        b = Bidirectional(n=neurons, name='bi_'+rnn_type+'_layer_', n_blocks=n_blocks-2, rnn_type=rnn_type)
        model.add(b)
        b2 = Bidirectional(n=neurons, output_type='encoding', src_layers=b.get_last_layers(), rnn_type=rnn_type,
                           name='bi_'+rnn_type+'_lastlayer')
        model.add(b2)
        model.add(Recurrent(n=neurons, output_type='arbitrarylength', src_layers=b2.get_last_layers(),
                            rnn_type=rnn_type, max_output_length=max_output_length))
        model.add(OutputLayer())
    elif n_blocks >= 2:
        model = Sequential(conn=conn, model_table=model_table)
        b2 = Bidirectional(n=neurons, output_type='encoding', rnn_type=rnn_type, name='bi_'+rnn_type+'_layer_')
        model.add(b2)
        model.add(Recurrent(n=neurons, output_type='arbitrarylength', src_layers=b2.get_last_layers(),
                            rnn_type=rnn_type, max_output_length=max_output_length))
        model.add(OutputLayer())
    else:
        raise DLPyError('The number of blocks for a text generation model should be at least 2.')

    return model


def SequenceLabeling(conn, model_table='sequence_labeling_model', neurons=10, n_blocks=3, rnn_type='gru'):
    '''
    Generates a sequence labeling model.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    neurons : int, optional
        Specifies the number of neurons to be in each layer.
        Default: 10
    n_blocks : int, optional
        Specifies the number of bidirectional blocks to be added to the model.
        Default: 3
    rnn_type : string, optional
        Specifies the type of the rnn layer.
        Default: RNN
        Valid Values: RNN, LSTM, GRU
        
    Returns
    -------
    :class:`Sequential`

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if n_blocks >= 1:
        model = Sequential(conn=conn, model_table=model_table)
        model.add(Bidirectional(n=neurons, n_blocks=n_blocks, rnn_type=rnn_type, name='bi_'+rnn_type+'_layer_'))
        model.add(OutputLayer())
    else:
        raise DLPyError('The number of blocks for a sequence labeling model should be at least 1.')

    return model


def SpeechRecognition(conn, model_table='acoustic_model', neurons=10, n_blocks=3, rnn_type='gru'):
    '''
    Generates a speech recognition model.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    neurons : int, optional
        Specifies the number of neurons to be in each layer.
        Default: 10
    n_blocks : int, optional
        Specifies the number of bidirectional blocks to be added to the model.
        Default: 3
    rnn_type : string, optional
        Specifies the type of the rnn layer.
        Default: RNN
        Valid Values: RNN, LSTM, GRU

    Returns
    -------
    :class:`Sequential`

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if n_blocks >= 1:
        model = Sequential(conn=conn, model_table=model_table)
        model.add(Bidirectional(n=neurons, n_blocks=n_blocks, rnn_type=rnn_type, name='bi_'+rnn_type+'_layer_'))
        model.add(OutputLayer(error='CTC'))
    else:
        raise DLPyError('The number of blocks for an acoustic model should be at least 1.')

    return model


def LeNet5(conn, model_table='LENET5', n_classes=10, n_channels=1, width=28, height=28, scale=1.0 / 255,
           random_flip='none', random_crop='none', offsets=0):
    '''
    Generates a deep learning model with the LeNet5 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 1
    width : int, optional
        Specifies the width of the input layer.
        Default: 28
    height : int, optional
        Specifies the height of the input layer.
        Default: 28
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model
        will automatically detect the number of classes based on the
        training set.
        Default: 10
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'v', 'hv', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data
        is used. Images are cropped to the values that are specified in the
        width and height parameters. Only the images with one or both
        dimensions that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: 0

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    model.add(Conv2d(n_filters=6, width=5, height=5, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=16, width=5, height=5, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=120))
    model.add(Dense(n=84))

    model.add(OutputLayer(n=n_classes))

    return model


def VGG11(conn, model_table='VGG11', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the VGG11 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1409.1556.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))

    model.add(OutputLayer(n=n_classes))

    return model


def VGG13(conn, model_table='VGG13', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the VGG13 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1409.1556.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))

    model.add(OutputLayer(n=n_classes))

    return model


def VGG16(conn, model_table='VGG16', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
          pre_trained_weights=False, pre_trained_weights_file=None, include_top=False):
    '''
    Generates a deep learning model with the VGG16 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights trained on the ImageNet data set.
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pre-trained weights.
        Must be a fully qualified file name of SAS-compatible file (e.g., *.caffemodel.h5)
        Note: Required when pre_trained_weights=True.
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers (i.e., the FC layers)
        Default: False

    Returns
    -------
    :class:`Sequential`
        If `pre_trained_weights` is False
    :class:`Model`
        If `pre_trained_weights` is True

    References
    ----------
    https://arxiv.org/pdf/1409.1556.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                             random_flip=random_flip, random_crop=random_crop))

        model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Dense(n=4096, dropout=0.5, name='fc6'))
        model.add(Dense(n=4096, dropout=0.5, name='fc7'))

        model.add(OutputLayer(n=n_classes, name='fc8'))

        return model

    else:
        # TODO: I need to re-factor loading / downloading pre-trained models.
        # something like pytorch style

        if pre_trained_weights_file is None:
            raise DLPyError('\nThe pre-trained weights file is not specified.\n'
                            'Please follow the steps below to attach the pre-trained weights:\n'
                            '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. specify the pre_trained_weights_file using the fully qualified server side path.')

        model_cas = model_vgg16.VGG16_Model(s=conn, model_table=model_table, n_channels=n_channels,
                                            width=width, height=height, random_crop=random_crop, offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)
            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:
            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_trained_weights_file)

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<19'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True, **model.model_weights.to_table_params()))
            model._retrieve_('deeplearn.removelayer', model=model_table, name='fc8')
            model._retrieve_('deeplearn.addlayer', model=model_table, name='fc8',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['fc7'])
            model = Model.from_table(conn.CASTable(model_table))

            return model


def VGG19(conn, model_table='VGG19', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
          pre_trained_weights=False, pre_trained_weights_file=None, include_top=False):
    '''
    Generates a deep learning model with the VGG19 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights trained on the ImageNet data set.
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pre-trained weights.
        Must be a fully qualified file name of SAS-compatible file (e.g., *.caffemodel.h5)
        Note: Required when pre_trained_weights=True.
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers (i.e., the FC layers).
        Default: False

    Returns
    -------
    :class:`Sequential`
        If `pre_trained_weights` is False
    :class:`Model`
        If `pre_trained_weights` is True

    References
    ----------
    https://arxiv.org/pdf/1409.1556.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                             random_flip=random_flip, random_crop=random_crop))

        model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Dense(n=4096, dropout=0.5))
        model.add(Dense(n=4096, dropout=0.5))

        model.add(OutputLayer(n=n_classes))

        return model

    else:
        if pre_trained_weights_file is None:
            raise DLPyError('\nThe pre-trained weights file is not specified.\n'
                            'Please follow the steps below to attach the pre-trained weights:\n'
                            '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. specify the pre_trained_weights_file using the fully qualified server side path.')

        model_cas = model_vgg19.VGG19_Model(s=conn, model_table=model_table, n_channels=n_channels,
                                            width=width, height=height, random_crop=random_crop, offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)

            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:

            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_trained_weights_file)

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<22'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True, **model.model_weights.to_table_params()))
            model._retrieve_('deeplearn.removelayer', model=model_table, name='fc8')
            model._retrieve_('deeplearn.addlayer', model=model_table, name='fc8',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['fc7'])
            model = Model.from_table(conn.CASTable(model_table))

            return model


def ResNet18_SAS(conn, model_table='RESNET18_SAS', batch_norm_first=True, n_classes=1000, n_channels=3, width=224,
                 height=224, scale=1, random_flip='none', random_crop='none', random_mutation='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the ResNet18 architecture.

    Compared to Caffe ResNet18, the model prepends a batch normalization layer to the last global pooling layer.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    batch_norm_first : bool, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: True
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop, random_mutation=random_mutation))

    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
    n_filters_list = [(64, 64), (128, 128), (256, 256), (512, 512)]
    rep_nums_list = [2, 2, 2, 2]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters, strides=strides,
                                 batch_norm_first=batch_norm_first))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(GlobalAveragePooling2D())

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet18_Caffe(conn, model_table='RESNET18_CAFFE', batch_norm_first=False, n_classes=1000, n_channels=3, width=224,
                   height=224, scale=1, random_flip='none', random_crop='none', random_mutation='none', offsets=None):
    '''
    Generates a deep learning model with the ResNet18 architecture with convolution shortcut.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: False
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop, random_mutation=random_mutation))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
    n_filters_list = [(64, 64), (128, 128), (256, 256), (512, 512)]
    rep_nums_list = [2, 2, 2, 2]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if rep_num == 0:
                conv_short_cut = True
                if i == 0:
                    strides = 1
                else:
                    strides = 2
            else:
                conv_short_cut = False
                strides = 1
            model.add(ResBlock_Caffe(kernel_sizes=kernel_sizes, n_filters=n_filters, strides=strides,
                                     batch_norm_first=batch_norm_first, conv_short_cut=conv_short_cut))

    # Bottom Layers
    # pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(GlobalAveragePooling2D())

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet34_SAS(conn, model_table='RESNET34_SAS', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                 batch_norm_first=True, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the ResNet34 architecture.

    Compared to Caffe ResNet34, the model prepends a batch normalization layer to the last global pooling layer.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: True
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
    n_filters_list = [(64, 64), (128, 128), (256, 256), (512, 512)]
    rep_nums_list = [3, 4, 6, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet34_Caffe(conn, model_table='RESNET34_CAFFE',  n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                   batch_norm_first=False, random_flip='none', random_crop='none', offsets=None):
    '''
    Generates a deep learning model with the ResNet34 architecture with convolution shortcut.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: False
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    # Configuration of the residual blocks
    kernel_sizes_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
    n_filters_list = [(64, 64), (128, 128), (256, 256), (512, 512)]
    rep_nums_list = [3, 4, 6, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if rep_num == 0:
                conv_short_cut = True
                if i == 0:
                    strides = 1
                else:
                    strides = 2
            else:
                conv_short_cut = False
                strides = 1
            model.add(ResBlock_Caffe(kernel_sizes=kernel_sizes, n_filters=n_filters, strides=strides,
                                     batch_norm_first=batch_norm_first, conv_short_cut=conv_short_cut))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet50_SAS(conn, model_table='RESNET50_SAS', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                 batch_norm_first=True, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the ResNet50 architecture.

    Compared to Caffe ResNet50, the model prepends a batch normalization layer to the last global pooling layer.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: True
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(1, 3, 1)] * 4
    n_filters_list = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
    rep_nums_list = [3, 4, 6, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))

    model.add(BN(act='relu'))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet50_Caffe(conn, model_table='RESNET50_CAFFE',  n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                   batch_norm_first=False, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
                   pre_trained_weights=False, pre_trained_weights_file=None, include_top=False):
    '''
    Generates a deep learning model with the ResNet50 architecture with convolution shortcut.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: False
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights trained on the ImageNet data set.
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pre-trained weights.
        This option is required when pre_trained_weights=True.
        Must be a fully qualified file name of SAS-compatible file (e.g., *.caffemodel.h5)
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers
        (i.e., the last layer for classification).
        Default: False

    Returns
    -------
    :class:`Sequential`
        If `pre_trained_weights` is `False`
    :class:`Model`
        If `pre_trained_weights` is `True`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                             random_flip=random_flip, random_crop=random_crop))
        # Top layers
        model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
        model.add(BN(act='relu'))
        model.add(Pooling(width=3, stride=2))
        # Residual block configuration.
        kernel_sizes_list = [(1, 3, 1)] * 4
        n_filters_list = [(64, 64, 256), (128, 128, 512),
                          (256, 256, 1024), (512, 512, 2048)]
        rep_nums_list = [3, 4, 6, 3]

        for i in range(4):
            kernel_sizes = kernel_sizes_list[i]
            n_filters = n_filters_list[i]
            for rep_num in range(rep_nums_list[i]):
                if rep_num == 0:
                    conv_short_cut = True
                    if i == 0:
                        strides = 1
                    else:
                        strides = 2
                else:
                    conv_short_cut = False
                    strides = 1
                model.add(ResBlock_Caffe(kernel_sizes=kernel_sizes,
                                         n_filters=n_filters, strides=strides,
                                         batch_norm_first=batch_norm_first,
                                         conv_short_cut=conv_short_cut))

        # Bottom Layers
        pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
        model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

        model.add(OutputLayer(act='softmax', n=n_classes))

        return model

    else:
        if pre_trained_weights_file is None:
            raise DLPyError('\nThe pre-trained weights file is not specified.\n'
                            'Please follow the steps below to attach the pre-trained weights:\n'
                            '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. specify the pre_trained_weights_file using the fully qualified server side path.')

        model_cas = model_resnet50.ResNet50_Model(s=conn, model_table=model_table, n_channels=n_channels,
                                                  width=width, height=height, random_crop=random_crop, offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)

            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:
            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_trained_weights_file)
            model._retrieve_('deeplearn.removelayer', model=model_table, name='fc1000')
            model._retrieve_('deeplearn.addlayer', model=model_table, name='output',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['pool5'])

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<125'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True, **model.model_weights.to_table_params()))
            model = Model.from_table(conn.CASTable(model_table))
            return model


def ResNet101_SAS(conn, model_table='RESNET101_SAS',  n_classes=1000,  n_channels=3, width=224, height=224, scale=1,
                  batch_norm_first=True, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the ResNet101 architecture.

    Compared to Caffe ResNet101, the model prepends a batch normalization
    layer to the last global pooling layer.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: True
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(1, 3, 1)] * 4
    n_filters_list = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
    rep_nums_list = [3, 4, 23, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))
    model.add(BN(act='relu'))
    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet101_Caffe(conn, model_table='RESNET101_CAFFE', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                    batch_norm_first=False, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
                    pre_trained_weights=False, pre_trained_weights_file=None, include_top=False):
    '''
    Generates a deep learning model with the ResNet101 architecture with convolution shortcut.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: False
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights from ImageNet data set
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pre-trained weights.
        Must be a fully qualified file name of SAS-compatible file (e.g., *.caffemodel.h5)
        Note: Required when pre_trained_weights=True.
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers,
        i.e. the last layer for classification.
        Default: False.

    Returns
    -------
    :class:`Sequential`
        If `pre_trained_weights` is `False`
    :class:`Model`
        If `pre_trained_weights` is `True`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                             random_flip=random_flip, random_crop=random_crop))
        # Top layers
        model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
        model.add(BN(act='relu'))
        model.add(Pooling(width=3, stride=2))
        # Residual block configuration.
        kernel_sizes_list = [(1, 3, 1)] * 4
        n_filters_list = [(64, 64, 256), (128, 128, 512),
                          (256, 256, 1024), (512, 512, 2048)]
        rep_nums_list = [3, 4, 23, 3]

        for i in range(4):
            kernel_sizes = kernel_sizes_list[i]
            n_filters = n_filters_list[i]
            for rep_num in range(rep_nums_list[i]):
                if rep_num == 0:
                    conv_short_cut = True
                    if i == 0:
                        strides = 1
                    else:
                        strides = 2
                else:
                    conv_short_cut = False
                    strides = 1
                model.add(ResBlock_Caffe(kernel_sizes=kernel_sizes,
                                         n_filters=n_filters, strides=strides,
                                         batch_norm_first=batch_norm_first,
                                         conv_short_cut=conv_short_cut))

        # Bottom Layers
        pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
        model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

        model.add(OutputLayer(act='softmax', n=n_classes))

        return model

    else:
        if pre_trained_weights_file is None:
            raise DLPyError('\nThe pre-trained weights file is not specified.\n'
                            'Please follow the steps below to attach the pre-trained weights:\n'
                            '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. specify the pre_trained_weights_file using the fully qualified server side path.')
        model_cas = model_resnet101.ResNet101_Model( s=conn, model_table=model_table, n_channels=n_channels,
                                                     width=width, height=height, random_crop=random_crop,
                                                     offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)

            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:
            model = Model.from_table(conn.CASTable(model_table), display_note=False)
            model.load_weights(path=pre_trained_weights_file)
            model._retrieve_('deeplearn.removelayer', model=model_table, name='fc1000')
            model._retrieve_('deeplearn.addlayer', model=model_table, name='output',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['pool5'])

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<244'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True, **model.model_weights.to_table_params()))
            model = Model.from_table(conn.CASTable(model_table))
            return model


def ResNet152_SAS(conn, model_table='RESNET152_SAS',  n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                  batch_norm_first=True, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the SAS ResNet152 architecture.

    Compared to Caffe ResNet152, the model prepends a batch normalization
    layer to the last global pooling layer.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: True
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(1, 3, 1)] * 4
    n_filters_list = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
    rep_nums_list = [3, 8, 36, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))
    model.add(BN(act='relu'))
    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2,
                    height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet152_Caffe(conn, model_table='RESNET152_CAFFE',  n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                    batch_norm_first=False, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
                    pre_trained_weights=False, pre_trained_weights_file=None, include_top=False):
    '''
    Generates a deep learning model with the ResNet152 architecture with convolution shortcut

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: False
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights trained on the ImageNet data set.
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pre-trained weights.
        Must be a fully qualified file name of SAS-compatible file (e.g., *.caffemodel.h5)
        Note: Required when pre_trained_weights=True.
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers,
        i.e. the last layer for classification.
        Default: False

    Returns
    -------
    :class:`Sequential`
        If `pre_trained_weights` is `False`
    :class:`Model`
        If `pre_trained_weights` is `True`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                             scale=scale, offsets=offsets, random_flip=random_flip,
                             random_crop=random_crop))
        # Top layers
        model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
        model.add(BN(act='relu'))
        model.add(Pooling(width=3, stride=2))
        # Residual block configuration.
        kernel_sizes_list = [(1, 3, 1)] * 4
        n_filters_list = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
        rep_nums_list = [3, 8, 36, 3]

        for i in range(4):
            kernel_sizes = kernel_sizes_list[i]
            n_filters = n_filters_list[i]
            for rep_num in range(rep_nums_list[i]):
                if rep_num == 0:
                    conv_short_cut = True
                    if i == 0:
                        strides = 1
                    else:
                        strides = 2
                else:
                    conv_short_cut = False
                    strides = 1
                model.add(ResBlock_Caffe(kernel_sizes=kernel_sizes,
                                         n_filters=n_filters, strides=strides,
                                         batch_norm_first=batch_norm_first,
                                         conv_short_cut=conv_short_cut))

        # Bottom Layers
        pooling_size = (width // 2 // 2 // 2 // 2 // 2,
                        height // 2 // 2 // 2 // 2 // 2)
        model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

        model.add(OutputLayer(act='softmax', n=n_classes))

        return model
    else:
        if pre_trained_weights_file is None:
            raise ValueError('\nThe pre-trained weights file is not specified.\n'
                             'Please follow the steps below to attach the pre-trained weights:\n'
                             '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                             'and download the associated weight file.\n'
                             '2. upload the *.h5 file to '
                             'a server side directory which the CAS session has access to.\n'
                             '3. specify the pre_trained_weights_file using the fully qualified server side path.')
        model_cas = model_resnet152.ResNet152_Model( s=conn, model_table=model_table, n_channels=n_channels,
                                                     width=width, height=height, random_crop=random_crop,
                                                     offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)

            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:
            model = Model.from_table(conn.CASTable(model_table), display_note=False)
            model.load_weights(path=pre_trained_weights_file)
            model._retrieve_('deeplearn.removelayer', model=model_table, name='fc1000')
            model._retrieve_('deeplearn.addlayer', model=model_table, name='output',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['pool5'])

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<363'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True, **model.model_weights.to_table_params()))
            model = Model.from_table(conn.CASTable(model_table))
            return model


def ResNet_Wide(conn, model_table='WIDE_RESNET', batch_norm_first=True, number_of_blocks=1, k=4, n_classes=None,
                n_channels=3, width=32, height=32, scale=1, random_flip='none', random_crop='none',
                offsets=(103.939, 116.779, 123.68)):
    '''
    Generate a deep learning model with Wide ResNet architecture.

    Wide ResNet is just a ResNet with more feature maps in each convolutional layers.
    The width of ResNet is controlled by widening factor k.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: True
    number_of_blocks : int
        Specifies the number of blocks in a residual group. For example,
        this value is [2, 2, 2, 2] for the ResNet18 architecture and [3, 4, 6, 3]
        for the ResNet34 architecture. In this case, the number of blocks
        are the same for each group as in the ResNet18 architecture.
        Default: 1
    k : int
        Specifies the widening factor.
        Default: 4
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 32
    height : int, optional
        Specifies the height of the input layer.
        Default: 32
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1605.07146.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    in_filters = 16

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))
    # Top layers
    model.add(Conv2d(in_filters, 3, act='identity', include_bias=False, stride=1))
    model.add(BN(act='relu'))

    # Residual block configuration.
    n_filters_list = [(16 * k, 16 * k), (32 * k, 32 * k), (64 * k, 64 * k)]
    kernel_sizes_list = [(3, 3)] * len(n_filters_list)
    rep_nums_list = [number_of_blocks, number_of_blocks, number_of_blocks]

    for i in range(len(n_filters_list)):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))
    model.add(BN(act='relu'))
    # Bottom Layers
    pooling_size = (width // 2 // 2, height // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def MobileNetV1(conn, model_table='MobileNetV1', n_classes=1000, n_channels=3, width=224, height=224,
                random_flip='none', random_crop='none', random_mutation='none',
                norm_stds = [255 * 0.229, 255 * 0.224, 255 * 0.225], offsets = (255 * 0.485, 255 * 0.456, 255 * 0.406),
                alpha=1, depth_multiplier=1):
    '''
    Generate a deep learning model with MobileNetV1 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 32
    height : int, optional
        Specifies the height of the input layer.
        Default: 32
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    norm_stds : double or iter-of-doubles, optional

        Default: (255 * 0.229, 255 * 0.224, 255 * 0.225)
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    alpha : int, optional
    depth_multiplier : int, optional


    Returns
    -------
    :class:`Model`

    References
    ----------
    https://arxiv.org/pdf/1605.07146.pdf

    '''
    def _conv_block(inputs, filters, alpha, kernel = 3, stride = 1):
        """Adds an initial convolution layer (with batch normalization and relu6).
        # Arguments
            inputs: Input tensor of shape `(rows, cols, 3)`
                (with `channels_last` data format) or
                (3, rows, cols) (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 32.
                E.g. `(224, 224, 3)` would be one valid value.
            filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the convolution).
            alpha: controls the width of the network.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                     are used at each layer.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution
                along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
        # Input shape
            4D tensor with shape:
            `(samples, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(samples, rows, cols, channels)` if data_format='channels_last'.
        # Output shape
            4D tensor with shape:
            `(samples, filters, new_rows, new_cols)`
            if data_format='channels_first'
            or 4D tensor with shape:
            `(samples, new_rows, new_cols, filters)`
            if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.
        # Returns
            Output tensor of block.
        """
        filters = int(filters * alpha)
        x = Conv2d(filters, kernel, act = 'identity', include_bias = False, stride = stride, name = 'conv1')(inputs)
        x = BN(name = 'conv1_bn', act='relu')(x)
        return x, filters

    def _depthwise_conv_block(inputs, n_groups, pointwise_conv_filters, alpha,
                              depth_multiplier = 1, stride = 1, block_id = 1):
        """Adds a depthwise convolution block.
        A depthwise convolution block consists of a depthwise conv,
        batch normalization, relu6, pointwise convolution,
        batch normalization and relu6 activation.
        # Arguments
            inputs: Input tensor of shape `(rows, cols, channels)`
                (with `channels_last` data format) or
                (channels, rows, cols) (with `channels_first` data format).
            pointwise_conv_filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the pointwise convolution).
            alpha: controls the width of the network.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                     are used at each layer.
            depth_multiplier: The number of depthwise convolution output channels
                for each input channel.
                The total number of depthwise convolution output
                channels will be equal to `filters_in * depth_multiplier`.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution
                along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
            block_id: Integer, a unique identification designating
                the block number.
        # Input shape
            4D tensor with shape:
            `(batch, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, rows, cols, channels)` if data_format='channels_last'.
        # Output shape
            4D tensor with shape:
            `(batch, filters, new_rows, new_cols)`
            if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, new_rows, new_cols, filters)`
            if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.
        # Returns
            Output tensor of block.
        """
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)

        x = GroupConv2d(n_groups, n_groups, 3, stride = stride, act = 'identity',
                        include_bias = False, name = 'conv_dw_%d' % block_id)(inputs)
        x = BN(name = 'conv_dw_%d_bn' % block_id, act = 'relu')(x)

        x = Conv2d(pointwise_conv_filters, 1, act = 'identity', include_bias = False, stride = 1,
                   name = 'conv_pw_%d' % block_id)(x)
        x = BN(name = 'conv_pw_%d_bn' % block_id, act = 'relu')(x)
        return x, pointwise_conv_filters

    inp = Input(n_channels=n_channels, width=width, height=height, name='data',
                norm_stds = norm_stds, offsets = offsets,
                random_flip=random_flip, random_crop=random_crop, random_mutation=random_mutation)
    x, depth = _conv_block(inp, 32, alpha, stride = 2)
    x, depth = _depthwise_conv_block(x, depth, 64, alpha, depth_multiplier, block_id = 1)

    x, depth = _depthwise_conv_block(x, depth, 128, alpha, depth_multiplier,
                                     stride = 2, block_id = 2)
    x, depth = _depthwise_conv_block(x, depth, 128, alpha, depth_multiplier, block_id = 3)

    x, depth = _depthwise_conv_block(x, depth, 256, alpha, depth_multiplier,
                                     stride = 2, block_id = 4)
    x, depth = _depthwise_conv_block(x, depth, 256, alpha, depth_multiplier, block_id = 5)

    x, depth = _depthwise_conv_block(x, depth, 512, alpha, depth_multiplier,
                                     stride = 2, block_id = 6)
    x, depth = _depthwise_conv_block(x, depth, 512, alpha, depth_multiplier, block_id = 7)
    x, depth = _depthwise_conv_block(x, depth, 512, alpha, depth_multiplier, block_id = 8)
    x, depth = _depthwise_conv_block(x, depth, 512, alpha, depth_multiplier, block_id = 9)
    x, depth = _depthwise_conv_block(x, depth, 512, alpha, depth_multiplier, block_id = 10)
    x, depth = _depthwise_conv_block(x, depth, 512, alpha, depth_multiplier, block_id = 11)

    x, depth = _depthwise_conv_block(x, depth, 1024, alpha, depth_multiplier,
                                     stride = 2, block_id = 12)
    x, depth = _depthwise_conv_block(x, depth, 1024, alpha, depth_multiplier, block_id = 13)

    x = GlobalAveragePooling2D(name = "Global_avg_pool")(x)
    x = OutputLayer(n=n_classes)(x)

    model = Model(conn, inp, x, model_table)
    model.compile()

    return model


def MobileNetV2(conn, model_table='MobileNetV2', n_classes=1000, n_channels=3, width=224, height=224,
                norm_stds=[255 * 0.229, 255 * 0.224, 255 * 0.225], offsets=(255*0.485, 255*0.456, 255*0.406),
                random_flip='none', random_crop='none', random_mutation='none', alpha=1):
    def _make_divisible(v, divisor, min_value = None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _inverted_res_block(inputs, in_channels, expansion, stride, alpha, filters, block_id):
        # in_channels = backend.int_shape(inputs)[channel_axis]
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
        x = inputs
        prefix = 'block_{}_'.format(block_id)
        n_groups = in_channels

        if block_id:
            # Expand
            n_groups = expansion * in_channels
            x = Conv2d(expansion * in_channels, 1, include_bias = False, act = 'identity',
                       name = prefix + 'expand')(x)
            x = BN(name = prefix + 'expand_BN', act = 'identity')(x)
        else:
            prefix = 'expanded_conv_'

        # Depthwise
        x = GroupConv2d(n_groups, n_groups, 3, stride = stride, act = 'identity',
                        include_bias = False, name = prefix + 'depthwise')(x)
        x = BN(name = prefix + 'depthwise_BN', act = 'relu')(x)

        # Project
        x = Conv2d(pointwise_filters, 1, include_bias = False, act = 'identity', name = prefix + 'project')(x)
        x = BN(name = prefix + 'project_BN', act = 'identity')(x)

        if in_channels == pointwise_filters and stride == 1:
            return Res(name = prefix + 'add')([inputs, x]), pointwise_filters
        return x, pointwise_filters

    inp = Input(n_channels = n_channels, width = width, height = height, name = 'data',
                norm_stds = norm_stds, offsets = offsets,
                random_flip = random_flip, random_crop = random_crop, random_mutation = random_mutation)
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2d(first_block_filters, 3, stride = 2, include_bias = False, name = 'Conv1', act = 'identity')(inp)
    x = BN(name = 'bn_Conv1', act='relu')(x)

    x, n_channels = _inverted_res_block(x, first_block_filters, filters = 16, alpha = alpha, stride = 1,
                                        expansion = 1, block_id = 0)

    x, n_channels = _inverted_res_block(x, n_channels, filters = 24, alpha = alpha, stride = 2,
                                        expansion = 6, block_id = 1)
    x, n_channels = _inverted_res_block(x, n_channels, filters = 24, alpha = alpha, stride = 1,
                                        expansion = 6, block_id = 2)

    x, n_channels = _inverted_res_block(x, n_channels, filters = 32, alpha = alpha, stride = 2,
                                        expansion = 6, block_id = 3)
    x, n_channels = _inverted_res_block(x, n_channels, filters = 32, alpha = alpha, stride = 1,
                                        expansion = 6, block_id = 4)
    x, n_channels = _inverted_res_block(x, n_channels, filters = 32, alpha = alpha, stride = 1,
                                        expansion = 6, block_id = 5)

    x, n_channels = _inverted_res_block(x, n_channels, filters = 64, alpha = alpha, stride = 2,
                                        expansion = 6, block_id = 6)
    x, n_channels = _inverted_res_block(x, n_channels, filters = 64, alpha = alpha, stride = 1,
                                        expansion = 6, block_id = 7)
    x, n_channels = _inverted_res_block(x, n_channels, filters = 64, alpha = alpha, stride = 1,
                                        expansion = 6, block_id = 8)
    x, n_channels = _inverted_res_block(x, n_channels, filters = 64, alpha = alpha, stride = 1,
                                        expansion = 6, block_id = 9)

    x, n_channels = _inverted_res_block(x, n_channels, filters = 96, alpha = alpha, stride = 1,
                                        expansion = 6, block_id = 10)
    x, n_channels = _inverted_res_block(x, n_channels, filters = 96, alpha = alpha, stride = 1,
                                        expansion = 6, block_id = 11)
    x, n_channels = _inverted_res_block(x, n_channels, filters = 96, alpha = alpha, stride = 1,
                                        expansion = 6, block_id = 12)

    x, n_channels = _inverted_res_block(x, n_channels, filters = 160, alpha = alpha, stride = 2,
                                        expansion = 6, block_id = 13)
    x, n_channels = _inverted_res_block(x, n_channels, filters = 160, alpha = alpha, stride = 1,
                                        expansion = 6, block_id = 14)
    x, n_channels = _inverted_res_block(x, n_channels, filters = 160, alpha = alpha, stride = 1,
                                        expansion = 6, block_id = 15)

    x, n_channels = _inverted_res_block(x, n_channels, filters = 320, alpha = alpha, stride = 1,
                                        expansion = 6, block_id = 16)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = Conv2d(last_block_filters, 1, include_bias = False, name = 'Conv_1', act = 'identity')(x)
    x = BN(name = 'Conv_1_bn', act = 'relu')(x)

    x = GlobalAveragePooling2D(name = "Global_avg_pool")(x)
    x = OutputLayer(n = n_classes)(x)

    model = Model(conn, inp, x, model_table)
    model.compile()

    return model


def ShuffleNetV1(conn, model_table='ShuffleNetV1', n_classes=1000, n_channels=3, width=224, height=224,
                 norm_stds=[255 * 0.229, 255 * 0.224, 255 * 0.225], offsets=(255*0.485, 255*0.456, 255*0.406),
                 random_flip='none', random_crop='none', random_mutation='none', scale_factor=1.0,
                 num_shuffle_units = [3, 7, 3], bottleneck_ratio=0.25, groups=3, block_act='identity'):
    import numpy as np

    def _block(x, channel_map, bottleneck_ratio, repeat = 1, groups = 1, stage = 1):
        """
        creates a bottleneck block containing `repeat + 1` shuffle units
        Parameters
        ----------
        x:
            Input tensor of with `channels_last` data format
        channel_map: list
            list containing the number of output channels for a stage
        repeat: int(1)
            number of repetitions for a shuffle unit with stride 1
        groups: int(1)
            number of groups per channel
        bottleneck_ratio: float
            bottleneck ratio implies the ratio of bottleneck channels to output channels.
            For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
            the width of the bottleneck feature map.
        stage: int(1)
            stage number
        Returns
        -------
        """
        x = _shuffle_unit(x, in_channels = channel_map[stage - 2],
                          out_channels = channel_map[stage - 1], strides = 2,
                          groups = groups, bottleneck_ratio = bottleneck_ratio,
                          stage = stage, block = 1)

        for i in range(1, repeat + 1):
            x = _shuffle_unit(x, in_channels = channel_map[stage - 1],
                              out_channels = channel_map[stage - 1], strides = 1,
                              groups = groups, bottleneck_ratio = bottleneck_ratio,
                              stage = stage, block = (i + 1))

        return x

    def _shuffle_unit(inputs, in_channels, out_channels, groups, bottleneck_ratio, strides = 2, stage = 1, block = 1):
        """
        creates a shuffleunit
        Parameters
        ----------
        inputs:
            Input tensor of with `channels_last` data format
        in_channels:
            number of input channels
        out_channels:
            number of output channels
        strides:
            An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
        groups: int(1)
            number of groups per channel
        bottleneck_ratio: float
            bottleneck ratio implies the ratio of bottleneck channels to output channels.
            For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
            the width of the bottleneck feature map.
        stage: int(1)
            stage number
        block: int(1)
            block number
        Returns
        -------
        """

        prefix = 'stage%d/block%d' % (stage, block)

        # if strides >= 2:
        # out_channels -= in_channels

        # default: 1/4 of the output channel of a ShuffleNet Unit
        bottleneck_channels = int(out_channels * bottleneck_ratio)
        groups = (1 if stage == 2 and block == 1 else groups)

        # x = _group_conv(inputs, in_channels, out_channels = bottleneck_channels,
        #                 groups = (1 if stage == 2 and block == 1 else groups),
        #                 name = '%s/1x1_gconv_1' % prefix)

        x = GroupConv2d(bottleneck_channels, n_groups = (1 if stage == 2 and block == 1 else groups), act = 'identity',
                        width = 1, height = 1, stride = 1, include_bias = False,
                        name = '%s/1x1_gconv_1' % prefix)(inputs)

        x = BN(act = 'relu', name = '%s/bn_gconv_1' % prefix)(x)

        x = ChannelShuffle(n_groups = groups, name = '%s/channel_shuffle' % prefix)(x)
        # depthwise convolutioin
        x = GroupConv2d(x.shape[-1], n_groups = x.shape[-1], width = 3, height = 3, include_bias = False,
                        stride = strides, act = 'identity',
                        name = '%s/1x1_dwconv_1' % prefix)(x)
        x = BN(act = block_act, name = '%s/bn_dwconv_1' % prefix)(x)

        out_channels = out_channels if strides == 1 else out_channels - in_channels
        x = GroupConv2d(out_channels, n_groups = groups, width = 1, height = 1, stride=1, act = 'identity',
                        include_bias = False, name = '%s/1x1_gconv_2' % prefix)(x)

        x = BN(act = block_act, name = '%s/bn_gconv_2' % prefix)(x)

        if strides < 2:
            ret = Res(act = 'relu', name = '%s/add' % prefix)([x, inputs])
        else:
            avg = Pooling(width = 3, height = 3, stride = 2, pool = 'mean', name = '%s/avg_pool' % prefix)(inputs)
            ret = Concat(act = 'relu', name = '%s/concat' % prefix)([x, avg])

        return ret

    # def _group_conv(x, in_channels, out_channels, groups, kernel = 1, stride = 1, name = ''):
    #     """
    #     grouped convolution
    #     Parameters
    #     ----------
    #     x:
    #         Input tensor of with `channels_last` data format
    #     in_channels:
    #         number of input channels
    #     out_channels:
    #         number of output channels
    #     groups:
    #         number of groups per channel
    #     kernel: int(1)
    #         An integer or tuple/list of 2 integers, specifying the
    #         width and height of the 2D convolution window.
    #         Can be a single integer to specify the same value for
    #         all spatial dimensions.
    #     stride: int(1)
    #         An integer or tuple/list of 2 integers,
    #         specifying the strides of the convolution along the width and height.
    #         Can be a single integer to specify the same value for all spatial dimensions.
    #     name: str
    #         A string to specifies the layer name
    #     Returns
    #     -------
    #     """
    #     if groups == 1:
    #         return Conv2d(out_channels, kernel, include_bias = False, stride = stride, name = name)(x)
    #
    #     return GroupConv2d(out_channels, kernel, n_groups = groups, stride=stride,
    #                        include_bias = False, name = name)(x)

    # model_table = "ShuffleNet_%.2gX_g%d_br_%.2g_%s" % (scale_factor, groups, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    out_dim_stage_two = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}
    exp = np.insert(np.arange(0, len(num_shuffle_units), dtype = np.float32), 0, 0)
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= out_dim_stage_two[groups]  # calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    inp = Input(n_channels = n_channels, width = width, height = height, name = 'data',
                norm_stds = norm_stds, offsets = offsets,
                random_flip = random_flip, random_crop = random_crop, random_mutation = random_mutation)

    # create shufflenet architecture
    x = Conv2d(out_channels_in_stage[0], 3, include_bias=False, stride=2, act="identity", name="conv1")(inp)
    x = BN(act = 'relu', name = 'bn1')(x)
    x = Pooling(width = 3, height = 3, stride=2, name="maxpool1")(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(0, len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = _block(x, out_channels_in_stage, repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   groups=groups, stage=stage + 2)

    x = GlobalAveragePooling2D(name="Global_avg_pool")(x)
    x = OutputLayer(n = n_classes)(x)

    model = Model(conn, inputs=inp, outputs=x, model_table = model_table)
    model.compile()

    return model


# def ShuffleNetV2():
'''https://github.com/opconty/keras-shufflenetV2'''
#     def channel_split(x, name = ''):
#         # equipartition
#         in_channles = x.shape.as_list()[-1]
#         ip = in_channles // 2
#         c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name = '%s/sp%d_slice' % (name, 0))(x)
#         c = Lambda(lambda z: z[:, :, :, ip:], name = '%s/sp%d_slice' % (name, 1))(x)
#         return c_hat, c
#
#     def channel_shuffle(x):
#         height, width, channels = x.shape.as_list()[1:]
#         channels_per_split = channels // 2
#         x = K.reshape(x, [-1, height, width, 2, channels_per_split])
#         x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
#         x = K.reshape(x, [-1, height, width, channels])
#         return x
#
#     def shuffle_unit(inputs, out_channels, bottleneck_ratio, strides = 2, stage = 1, block = 1):
#
#         prefix = 'stage{}/block{}'.format(stage, block)
#         bottleneck_channels = int(out_channels * bottleneck_ratio)
#         if strides < 2:
#             c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
#             inputs = c
#
#         x = Conv2d(bottleneck_channels, 1, stride = 1, act = 'identity', name = '{}/1x1conv_1'.format(prefix))(inputs)
#         x = BN(act = 'relu', name = '{}/bn_1x1conv_1'.format(prefix))(x)
#         x = GroupConv2d(x.shape[2], x.shape[2], kernel_size = 3, stride = strides,
#                             name = '{}/3x3dwconv'.format(prefix))(x)
#         x = BN(act = 'relu', name = '{}/bn_3x3dwconv'.format(prefix))(x)
#         x = Conv2d(bottleneck_channels, 1, stride = 1, act = 'identity', name = '{}/1x1conv_2'.format(prefix))(x)
#         x = BN(act = 'relu', name = '{}/bn_1x1conv_2'.format(prefix))(x)
#
#         if strides < 2:
#             ret = Concat(name = '{}/concat_1'.format(prefix))([x, c_hat])
#         else:
#             s2 = GroupConv2d(inputs.shape[2], inputs.shape[2], 3, stride = 2,
#                              name = '{}/3x3dwconv_2'.format(prefix))(inputs)
#             s2 = BN(act = 'relu', name = '{}/bn_3x3dwconv_2'.format(prefix))(s2)
#             s2 = Conv2d(bottleneck_channels, 1, stride = 1, act = 'identity',
#                         name = '{}/1x1_conv_3'.format(prefix))(s2)
#             s2 = BN(act = 'relu', name = '{}/bn_1x1conv_3'.format(prefix))(s2)
#             ret = Concat(name = '{}/concat_2'.format(prefix))([x, s2])
#
#         ret = ChannelShuffle(n_groups=2, name = '{}/channel_shuffle'.format(prefix))(ret)
#
#         return ret
#
#     def block(x, channel_map, bottleneck_ratio, repeat = 1, stage = 1):
#         x = shuffle_unit(x, out_channels = channel_map[stage - 1],
#                          strides = 2, bottleneck_ratio = bottleneck_ratio, stage = stage, block = 1)
#
#         for i in range(1, repeat + 1):
#             x = shuffle_unit(x, out_channels = channel_map[stage - 1], strides = 1,
#                              bottleneck_ratio = bottleneck_ratio, stage = stage, block = (1 + i))
#
#         return x


def DenseNet(conn, blocks, model_table='DenseNet', n_classes=1000, n_channels=3, width=224, height=224,
             norm_stds=[255 * 0.229, 255 * 0.224, 255 * 0.225], offsets=(255*0.485, 255*0.456, 255*0.406),
             random_flip='none', random_crop='none', random_mutation='none'):

    def dense_block(x, blocks, name):
        """A dense block.
        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.
        # Returns
            output tensor for the block.
        """
        for i in range(blocks):
            x = conv_block(x, 32, name = name + '_block' + str(i + 1))
        return x

    def transition_block(x, reduction, name):
        """A transition block.
        # Arguments
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.
        # Returns
            output tensor for the block.
        """
        x = BN(name = name + '_bn', act = 'relu')(x)
        x = Conv2d(x.shape[2] * reduction, 1, act = 'identity', include_bias = False, name = name + '_conv')(x)
        x = Pooling(width = 2, height = 2, stride = 2, pool = 'mean', name = name + '_pool')(x)
        return x

    def conv_block(x, growth_rate, name):
        """A building block for a dense block.
        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.
        # Returns
            Output tensor for the block.
        """
        x1 = BN(name = name + '_0_bn', act = 'relu')(x)
        x1 = Conv2d(4 * growth_rate, 1, act = 'identity', include_bias = False, name = name + '_1_conv')(x1)
        x1 = BN(name = name + '_1_bn', act = 'relu')(x1)
        x1 = Conv2d(growth_rate, 3, act = 'identity', include_bias = False, name = name + '_2_conv')(x1)
        x = Concat(name = name + '_concat')([x, x1])
        return x

    inp = Input(n_channels = n_channels, width = width, height = height, name = 'data',
                norm_stds = norm_stds, offsets = offsets,
                random_flip = random_flip, random_crop = random_crop, random_mutation = random_mutation)

    x = Conv2d(64, 7, stride=2, act = 'identity', include_bias=False, name='conv1/conv')(inp)
    x = BN(name='conv1/bn', act='relu')(x)
    x = Pooling(3, stride=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = BN(name='bn', act = 'relu')(x)

    x = GlobalAveragePooling2D(name = "Global_avg_pool")(x)
    x = OutputLayer(n = n_classes, act='softmax', name='output_layer')(x)

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(conn, inp, x, model_table='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(conn, inp, x, model_table='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(conn, inp, x, model_table='densenet201')
    else:
        model = Model(conn, inp, x, model_table=model_table)

    model.compile()

    return model


def Darknet_Reference(conn, model_table='Darknet_Reference', n_classes=1000, act='leaky',
                      n_channels=3, width=224, height=224,
                      norm_stds=[255 * 0.229, 255 * 0.224, 255 * 0.225], offsets=(255*0.485, 255*0.456, 255*0.406),
                      random_flip='H', random_crop='UNIQUE'):

    '''
    Generates a deep learning model with the Darknet_Reference architecture.

    The head of the model except the last convolutional layer is same as
    the head of Tiny Yolov2. Darknet Reference is pre-trained model for
    ImageNet classification.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    act : string
        Specifies the type of the activation function for the batch
        normalization layers and the final convolution layer.
        Default: 'leaky'
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3.
    width : int, optional
        Specifies the width of the input layer.
        Default: 224.
    height : int, optional
        Specifies the height of the input layer.
        Default: 224.
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'h'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'unique'

    Returns
    -------
    :class:`Sequential`

    '''

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, norm_stds=norm_stds, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    # conv1 224
    model.add(Conv2d(16, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv2 112
    model.add(Conv2d(32, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv3 56
    model.add(Conv2d(64, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv4 28
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv5 14
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv6 7
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=1, pool='max'))
    # conv7 7
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv8 7
    model.add(Conv2d(1000, width=1, act=act, include_bias=True, stride=1))

    model.add(Pooling(width=7, height=7, pool='mean'))
    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def Darknet(conn, model_table='Darknet', n_classes=1000, act='leaky', n_channels=3, width=224, height=224,
            norm_stds = [255 * 0.229, 255 * 0.224, 255 * 0.225], offsets=(255*0.485, 255*0.456, 255*0.406),
            random_flip='H', random_crop='UNIQUE'):
    '''
    Generate a deep learning model with the Darknet architecture.

    The head of the model except the last convolutional layer is
    same as the head of Yolov2. Darknet is pre-trained model for
    ImageNet classification.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model
        will automatically detect the number of classes based on the
        training set.
        Default: 1000
    act : string
        Specifies the type of the activation function for the batch
        normalization layers and the final convolution layer.
        Default: 'leaky'
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the
        input layer.
        Default: 3.
    width : int, optional
        Specifies the width of the input layer.
        Default: 224.
    height : int, optional
        Specifies the height of the input layer.
        Default: 224.
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel
        intensity values.
        Default: 1. / 255
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'h'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'unique'

    Returns
    -------
    :class:`Sequential`

    '''

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         norm_stds=norm_stds, random_flip=random_flip, offsets=offsets,
                         random_crop=random_crop))
    # conv1 224 416
    model.add(Conv2d(32, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    # conv2 112 208
    model.add(Conv2d(64, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    # conv3 56 104
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv4 56 104
    model.add(Conv2d(64, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv5 56 104
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    # conv6 28 52
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv7 28 52
    model.add(Conv2d(128, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv8 28 52
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    # conv9 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv10 14 26
    model.add(Conv2d(256, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv11 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv12 14 26
    model.add(Conv2d(256, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv13 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))  # route
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    # conv14 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv15 7 13
    model.add(Conv2d(512, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv16 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv17 7 13
    model.add(Conv2d(512, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv18 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv19 7 13
    model.add(Conv2d(1000, width=1, act=act, include_bias=True, stride=1))
    # model.add(BN(act = actx))

    model.add(Pooling(width=7, height=7, pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def YoloV2(conn, anchors, model_table='Tiny-Yolov2', n_channels=3, width=416, height=416, scale=1.0 / 255,
           random_mutation='NONE', act='leaky', act_detection='AUTO', softmax_for_class_prob=True,
           coord_type='YOLO', max_label_per_image=30, max_boxes=30,
           n_classes=20, predictions_per_grid=5, do_sqrt=True, grid_number=13,
           coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
           detection_threshold=None, iou_threshold=None, random_boxes=False, match_anchor_size=None,
           num_to_force_coord=None):
    '''
    Generates a deep learning model with the Yolov2 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    anchors : list
        Specifies the anchor box values.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 416
    height : int, optional
        Specifies the height of the input layer.
        Default: 416
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    act : string, optional
        Specifies the activation function for the batch normalization layers.
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 5
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 13
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information. Default: False
    match_anchor_size : bool, optional
        Whether to force the predicted box match the anchor boxes in sizes for all predictions
    num_to_force_coord : int, optional
        The number of leading chunk of images in training when the algorithm forces predicted objects
        in each grid to be equal to the anchor box sizes, and located at the grid center

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1612.08242.pdf

    '''

    if len(anchors) != 2 * predictions_per_grid:
        raise DLPyError('The size of the anchor list in the detection layer for YOLOv2 should be equal to '
                        'twice the number of predictions_per_grid.')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, random_mutation=random_mutation,
                         scale=scale))

    # conv1 224 416
    model.add(Conv2d(32, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv2 112 208
    model.add(Conv2d(64, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv3 56 104
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv4 56 104
    model.add(Conv2d(64, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv5 56 104
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv6 28 52
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv7 28 52
    model.add(Conv2d(128, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv8 28 52
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv9 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv10 14 26
    model.add(Conv2d(256, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv11 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv12 14 26
    model.add(Conv2d(256, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv13 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv14 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv15 7 13
    model.add(Conv2d(512, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv16 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv17 7 13
    model.add(Conv2d(512, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv18 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))

    model.add(
        Conv2d((n_classes + 5) * predictions_per_grid, width=1, act='identity', include_bias=False, stride=1))

    model.add(Detection(act = act_detection, detection_model_type = 'yolov2', anchors = anchors,
                        softmax_for_class_prob = softmax_for_class_prob, coord_type = coord_type,
                        class_number = n_classes, grid_number = grid_number,
                        predictions_per_grid = predictions_per_grid, do_sqrt = do_sqrt, coord_scale = coord_scale,
                        object_scale = object_scale, prediction_not_a_object_scale = prediction_not_a_object_scale,
                        class_scale = class_scale, detection_threshold = detection_threshold,
                        iou_threshold = iou_threshold, random_boxes = random_boxes,
                        max_label_per_image = max_label_per_image, max_boxes = max_boxes,
                        match_anchor_size = match_anchor_size, num_to_force_coord = num_to_force_coord))

    return model


def YoloV2_MultiSize(conn, anchors, model_table='Tiny-Yolov2', n_channels=3, width=416, height=416,
                     norm_stds=[255 * 0.229, 255 * 0.224, 255 * 0.225], offsets=(255*0.485, 255*0.456, 255*0.406),
                     random_mutation='NONE', act='leaky', act_detection='AUTO', softmax_for_class_prob=True,
                     coord_type='YOLO', max_label_per_image=30, max_boxes=30,
                     n_classes=20, predictions_per_grid=5, do_sqrt=True, grid_number=13,
                     coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
                     detection_threshold=None, iou_threshold=None, random_boxes=False, match_anchor_size=None,
                     num_to_force_coord=None):
    '''
    Generates a deep learning model with the Yolov2 architecture.

    The model is same as Yolov2 proposed in original paper. In addition to
    Yolov2, the model adds a passthrough layer that brings feature from an
    earlier layer to lower resolution layer.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    anchors : list
        Specifies the anchor box values.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 416
    height : int, optional
        Specifies the height of the input layer.
        Default: 416
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the
        input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    act : string, optional
        Specifies the activation function for the batch normalization layers.
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 5
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 13
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information. Default: False
    match_anchor_size : bool, optional
        Whether to force the predicted box match the anchor boxes in sizes for all predictions
    num_to_force_coord : int, optional
        The number of leading chunk of images in training when the algorithm forces predicted objects
        in each grid to be equal to the anchor box sizes, and located at the grid center

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1612.08242.pdf

    '''

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, random_mutation=random_mutation,
                         norm_stds = norm_stds, offsets = offsets))

    # conv1 224 416
    model.add(Conv2d(32, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv2 112 208
    model.add(Conv2d(64, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv3 56 104
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv4 56 104
    model.add(Conv2d(64, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv5 56 104
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv6 28 52
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv7 28 52
    model.add(Conv2d(128, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv8 28 52
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv9 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv10 14 26
    model.add(Conv2d(256, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv11 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv12 14 26
    model.add(Conv2d(256, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv13 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    pointLayer1 = BN(act=act, name='BN5_13')
    model.add(pointLayer1)
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv14 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv15 7 13
    model.add(Conv2d(512, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv16 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv17 7 13
    model.add(Conv2d(512, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv18 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))

    # conv19 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act, name='BN6_19'))
    # conv20 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    pointLayer2 = BN(act=act, name='BN6_20')
    model.add(pointLayer2)

    # conv21 7 26 * 26 * 512 -> 26 * 26 * 64
    model.add(Conv2d(64, width=1, act='identity', include_bias=False, stride=1, src_layers=[pointLayer1]))
    model.add(BN(act=act))
    # reshape 26 * 26 * 64 -> 13 * 13 * 256
    pointLayer3 = Reshape(act='identity', width=13, height=13, depth=256, name='reshape1')
    model.add(pointLayer3)

    # concat
    model.add(Concat(act='identity', src_layers=[pointLayer2, pointLayer3]))

    # conv22 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))

    model.add(
        Conv2d((n_classes + 5) * predictions_per_grid, width=1, act='identity', include_bias=False, stride=1))

    model.add(Detection(act = act_detection, detection_model_type = 'yolov2', anchors = anchors,
                        softmax_for_class_prob = softmax_for_class_prob, coord_type = coord_type,
                        class_number = n_classes, grid_number = grid_number,
                        predictions_per_grid = predictions_per_grid, do_sqrt = do_sqrt, coord_scale = coord_scale,
                        object_scale = object_scale, prediction_not_a_object_scale = prediction_not_a_object_scale,
                        class_scale = class_scale, detection_threshold = detection_threshold,
                        iou_threshold = iou_threshold, random_boxes = random_boxes,
                        max_label_per_image = max_label_per_image, max_boxes = max_boxes,
                        match_anchor_size = match_anchor_size, num_to_force_coord = num_to_force_coord))

    return model


def Tiny_YoloV2(conn, anchors, model_table='Tiny-Yolov2', n_channels=3, width=416, height=416, scale=1,
                norm_stds=[255 * 0.229, 255 * 0.224, 255 * 0.225], offsets=(255*0.485, 255*0.456, 255*0.406),
                random_mutation='NONE', act='leaky', act_detection='AUTO', softmax_for_class_prob=True,
                coord_type='YOLO', max_label_per_image=30, max_boxes=30,
                n_classes=20, predictions_per_grid=5, do_sqrt=True, grid_number=13,
                coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
                detection_threshold=None, iou_threshold=None, random_boxes=False, match_anchor_size=None,
                num_to_force_coord=None):
    '''
    Generate a deep learning model with the Tiny Yolov2 architecture.

    Tiny Yolov2 is a very small model of Yolov2, so that it includes fewer
    numbers of convolutional layer and batch normalization layer.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    anchors : list
        Specifies the anchor box values.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 416
    height : int, optional
        Specifies the height of the input layer.
        Default: 416
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the
        input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    act : string, optional
        Specifies the activation function for the batch normalization layers.
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 5
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 13
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information. Default: False
    match_anchor_size : bool, optional
        Whether to force the predicted box match the anchor boxes in sizes for all predictions
    num_to_force_coord : int, optional
        The number of leading chunk of images in training when the algorithm forces predicted objects
        in each grid to be equal to the anchor box sizes, and located at the grid center

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1612.08242.pdf

    '''

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, random_mutation=random_mutation,
                         norm_stds = norm_stds, offsets = offsets))
    # conv1 416 448
    model.add(Conv2d(n_filters=16, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv2 208 224
    model.add(Conv2d(n_filters=32, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv3 104 112
    model.add(Conv2d(n_filters=64, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv4 52 56
    model.add(Conv2d(n_filters=128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv5 26 28
    model.add(Conv2d(n_filters=256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv6 13 14
    model.add(Conv2d(n_filters=512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=1, pool='max'))
    # conv7 13
    model.add(Conv2d(n_filters=1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv8 13
    model.add(Conv2d(n_filters=512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))

    model.add(Conv2d((n_classes + 5) * predictions_per_grid, width=1, act='identity', include_bias=False, stride=1))

    model.add(Detection(act=act_detection, detection_model_type='yolov2', anchors=anchors,
                        softmax_for_class_prob=softmax_for_class_prob, coord_type=coord_type,
                        class_number=n_classes, grid_number=grid_number,
                        predictions_per_grid=predictions_per_grid, do_sqrt=do_sqrt, coord_scale=coord_scale,
                        object_scale=object_scale, prediction_not_a_object_scale=prediction_not_a_object_scale,
                        class_scale=class_scale, detection_threshold=detection_threshold,
                        iou_threshold=iou_threshold, random_boxes=random_boxes,
                        max_label_per_image=max_label_per_image, max_boxes=max_boxes,
                        match_anchor_size=match_anchor_size, num_to_force_coord=num_to_force_coord))
    return model


def YoloV1(conn, model_table='Yolov1', n_channels=3, width=448, height=448, scale=1.0 / 255,
           random_mutation='NONE', act='leaky', dropout=0, act_detection='AUTO', softmax_for_class_prob=True,
           coord_type='YOLO', max_label_per_image=30, max_boxes=30,
           n_classes=20, predictions_per_grid=2, do_sqrt=True, grid_number=7,
           coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
           detection_threshold=None, iou_threshold=None, random_boxes=False):
    '''
    Generates a deep learning model with the Yolo V1 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 448
    height : int, optional
        Specifies the height of the input layer.
        Default: 448
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in
        the input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    act: String, optional
        Specifies the activation function to be used in the convolutional layer
        layers and the final convolution layer.
        Default: 'leaky'
    dropout: double, optional
        Specifies the drop out rate.
        Default: 0
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 2
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 7
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information. Default: False

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1506.02640.pdf

    '''

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, random_mutation=random_mutation,
                         scale=scale))
    # conv1 448
    model.add(Conv2d(32, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv2 224
    model.add(Conv2d(64, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv3 112
    model.add(Conv2d(128, width=3, act=act, include_bias=False, stride=1))
    # conv4 112
    model.add(Conv2d(64, width=1, act=act, include_bias=False, stride=1))
    # conv5 112
    model.add(Conv2d(128, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv6 56
    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1))
    # conv7 56
    model.add(Conv2d(128, width=1, act=act, include_bias=False, stride=1))
    # conv8 56
    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv9 28
    model.add(Conv2d(512, width=3, act=act, include_bias=False, stride=1))
    # conv10 28
    model.add(Conv2d(256, width=1, act=act, include_bias=False, stride=1))
    # conv11 28
    model.add(Conv2d(512, width=3, act=act, include_bias=False, stride=1))
    # conv12 28
    model.add(Conv2d(256, width=1, act=act, include_bias=False, stride=1))
    # conv13 28
    model.add(Conv2d(512, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv14 14
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv15 14
    model.add(Conv2d(512, width=1, act=act, include_bias=False, stride=1))
    # conv16 14
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv17 14
    model.add(Conv2d(512, width=1, act=act, include_bias=False, stride=1))
    # conv18 14
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))

    # conv19 14
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv20 7
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=2))
    # conv21 7
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv22 7
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv23 7
    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1, dropout=dropout))
    model.add(Dense(n=(n_classes + (5 * predictions_per_grid)) * grid_number * grid_number, act='identity'))

    model.add(Detection(act = act_detection, detection_model_type = 'yolov1',
                        softmax_for_class_prob = softmax_for_class_prob, coord_type = coord_type,
                        class_number = n_classes, grid_number = grid_number,
                        predictions_per_grid = predictions_per_grid, do_sqrt = do_sqrt, coord_scale = coord_scale,
                        object_scale = object_scale, prediction_not_a_object_scale = prediction_not_a_object_scale,
                        class_scale = class_scale, detection_threshold = detection_threshold,
                        iou_threshold = iou_threshold, random_boxes = random_boxes,
                        max_label_per_image = max_label_per_image, max_boxes = max_boxes))

    return model


def Tiny_YoloV1(conn, model_table='Tiny-Yolov1', n_channels=3, width=448, height=448, scale=1.0 / 255,
                random_mutation='NONE', act='leaky', dropout=0, act_detection='AUTO', softmax_for_class_prob=True,
                coord_type='YOLO', max_label_per_image=30, max_boxes=30,
                n_classes=20, predictions_per_grid=2, do_sqrt=True, grid_number=7,
                coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
                detection_threshold=None, iou_threshold=None, random_boxes=False):
    '''
    Generates a deep learning model with the Tiny Yolov1 architecture.

    Tiny Yolov1 is a very small model of Yolov1, so that it includes
    fewer numbers of convolutional layer.

        Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 448
    height : int, optional
        Specifies the height of the input layer.
        Default: 448
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in
        the input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    act: String, optional
        Specifies the activation function to be used in the convolutional layer
        layers and the final convolution layer.
        Default: 'leaky'
    dropout: double, optional
        Specifies the drop out rate.
        Default: 0
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 2
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 7
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information. Default: False

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1506.02640.pdf

    '''

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, random_mutation=random_mutation,
                         scale=scale))

    model.add(Conv2d(16, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(32, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(64, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(128, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(512, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1, dropout=dropout))

    model.add(Dense(n=(n_classes + (5 * predictions_per_grid)) * grid_number * grid_number, act='identity'))

    model.add(Detection(act = act_detection, detection_model_type = 'yolov1',
                        softmax_for_class_prob = softmax_for_class_prob, coord_type = coord_type,
                        class_number = n_classes, grid_number = grid_number,
                        predictions_per_grid = predictions_per_grid, do_sqrt = do_sqrt, coord_scale = coord_scale,
                        object_scale = object_scale, prediction_not_a_object_scale = prediction_not_a_object_scale,
                        class_scale = class_scale, detection_threshold = detection_threshold,
                        iou_threshold = iou_threshold, random_boxes = random_boxes,
                        max_label_per_image = max_label_per_image, max_boxes = max_boxes))

    return model


def InceptionV3(conn, model_table='InceptionV3',
                n_classes=1000, n_channels=3, width=299, height=299, scale=1,
                random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
                pre_trained_weights=False, pre_trained_weights_file=None, include_top=False):
    '''
    Generates a deep learning model with the Inceptionv3 architecture with batch normalization layers.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model in.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 299
    height : int, optional
        Specifies the height of the input layer.
        Default: 299
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights from ImageNet data set
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pretained weights.
        Must be a fully qualified file name of SAS-compatible file (*.caffemodel.h5)
        Note: Required when pre_train_weight=True.
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers,
        i.e. the FC layers
        Default: False

    Returns
    -------
    :class:`Sequential`
        If `pre_train_weight` is `False`
    :class:`Model`
        If `pre_train_weight` is `True`

    References
    ----------
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width,
                             height=height, scale=scale, offsets=offsets,
                             random_flip=random_flip, random_crop=random_crop))

        # 299 x 299 x 3
        model.add(Conv2d(n_filters=32, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0))
        model.add(BN(act='relu'))
        # 149 x 149 x 32
        model.add(Conv2d(n_filters=32, width=3, height=3, stride=1,
                         act='identity', include_bias=False, padding=0))
        model.add(BN(act='relu'))
        # 147 x 147 x 32
        model.add(Conv2d(n_filters=64, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        # 147 x 147 x 64
        model.add(Pooling(width=3, height=3, stride=2, pool='max', padding=0))

        # 73 x 73 x 64
        model.add(Conv2d(n_filters=80, width=1, height=1, stride=1,
                         act='identity', include_bias=False, padding=0))
        model.add(BN(act='relu'))
        # 73 x 73 x 80
        model.add(Conv2d(n_filters=192, width=3, height=3, stride=1,
                         act='identity', include_bias=False, padding=0))
        model.add(BN(act='relu'))
        # 71 x 71 x 192
        pool2 = Pooling(width=3, height=3, stride=2, pool='max', padding=0)
        model.add(pool2)


        # mixed 0: output 35 x 35 x 256

        # branch1x1
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[pool2]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch5x5
        model.add(Conv2d(n_filters=48, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[pool2]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=64, width=5, height=5, stride=1,
                         act='identity', include_bias=False))
        branch5x5 = BN(act='relu')
        model.add(branch5x5)

        # branch3x3dbl
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[pool2]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        branch3x3dbl = BN(act='relu')
        model.add(branch3x3dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[pool2]))
        model.add(Conv2d(n_filters=32, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed0 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch5x5, branch3x3dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 1: output 35 x 35 x 288

        # branch1x1
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch5x5
        model.add(Conv2d(n_filters=48, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=64, width=5, height=5, stride=1,
                         act='identity', include_bias=False))
        branch5x5 = BN(act='relu')
        model.add(branch5x5)

        # branch3x3dbl
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        branch3x3dbl = BN(act='relu')
        model.add(branch3x3dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[concat]))
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed1 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch5x5, branch3x3dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 2: output 35 x 35 x 288

        # branch1x1
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch5x5
        model.add(Conv2d(n_filters=48, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=64, width=5, height=5, stride=1,
                         act='identity', include_bias=False))
        branch5x5 = BN(act='relu')
        model.add(branch5x5)

        # branch3x3dbl
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        branch3x3dbl = BN(act='relu')
        model.add(branch3x3dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[concat]))
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed2 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch5x5, branch3x3dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 3: output 17 x 17 x 768

        # branch3x3
        model.add(Conv2d(n_filters=384, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0,
                         src_layers=[concat]))
        branch3x3 = BN(act='relu')
        model.add(branch3x3)

        # branch3x3dbl
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0))
        branch3x3dbl = BN(act='relu')
        model.add(branch3x3dbl)

        # branch_pool
        branch_pool = Pooling(width=3, height=3, stride=2, pool='max',
                              padding=0, src_layers=[concat])
        model.add(branch_pool)

        # mixed3 concat
        concat = Concat(act='identity',
                        src_layers=[branch3x3, branch3x3dbl, branch_pool])
        model.add(concat)


        # mixed 4: output 17 x 17 x 768

        # branch1x1
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch7x7
        model.add(Conv2d(n_filters=128, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=128, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        branch7x7 = BN(act='relu')
        model.add(branch7x7)

        # branch7x7dbl
        model.add(Conv2d(n_filters=128, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=128, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=128, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=128, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        branch7x7dbl = BN(act='relu')
        model.add(branch7x7dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[concat]))
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed4 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch7x7, branch7x7dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 5, 6: output 17 x 17 x 768
        for i in range(2):
            # branch1x1
            model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            branch1x1 = BN(act='relu')
            model.add(branch1x1)

            # branch7x7
            model.add(Conv2d(n_filters=160, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=160, width=7, height=1, stride=1,
                             act='identity', include_bias=False))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                             act='identity', include_bias=False))
            branch7x7 = BN(act='relu')
            model.add(branch7x7)

            # branch7x7dbl
            model.add(Conv2d(n_filters=160, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=160, width=1, height=7, stride=1,
                             act='identity', include_bias=False))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=160, width=7, height=1, stride=1,
                             act='identity', include_bias=False))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=160, width=1, height=7, stride=1,
                             act='identity', include_bias=False))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                             act='identity', include_bias=False))
            branch7x7dbl = BN(act='relu')
            model.add(branch7x7dbl)

            # branch_pool
            model.add(Pooling(width=3, height=3, stride=1, pool='average',
                              src_layers=[concat]))
            model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                             act='identity', include_bias=False))
            branch_pool = BN(act='relu')
            model.add(branch_pool)

            # concat
            concat = Concat(act='identity',
                            src_layers=[branch1x1, branch7x7, branch7x7dbl,
                                        branch_pool])
            model.add(concat)


        # mixed 7: output 17 x 17 x 768

        # branch1x1
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch7x7
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        branch7x7 = BN(act='relu')
        model.add(branch7x7)

        # branch7x7dbl
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        branch7x7dbl = BN(act='relu')
        model.add(branch7x7dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[concat]))
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed7 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch7x7, branch7x7dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 8: output 8 x 8 x 1280

        # branch3x3
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=320, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0))
        branch3x3 = BN(act='relu')
        model.add(branch3x3)

        # branch7x7x3
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0))
        branch7x7x3 = BN(act='relu')
        model.add(branch7x7x3)

        # branch_pool
        branch_pool = Pooling(width=3, height=3, stride=2, pool='max',
                              padding=0, src_layers=[concat])
        model.add(branch_pool)

        # mixed8 concat
        concat = Concat(act='identity',
                        src_layers=[branch3x3, branch7x7x3, branch_pool])
        model.add(concat)


        # mixed 9, 10:  output 8 x 8 x 2048
        for i in range(2):
            # branch1x1
            model.add(Conv2d(n_filters=320, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            branch1x1 = BN(act='relu')
            model.add(branch1x1)

            # branch3x3
            model.add(Conv2d(n_filters=384, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            branch3x3 = BN(act='relu')
            model.add(branch3x3)

            model.add(Conv2d(n_filters=384, width=3, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[branch3x3]))
            branch3x3_1 = BN(act='relu')
            model.add(branch3x3_1)

            model.add(Conv2d(n_filters=384, width=1, height=3, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[branch3x3]))
            branch3x3_2 = BN(act='relu')
            model.add(branch3x3_2)

            branch3x3 = Concat(act='identity',
                               src_layers=[branch3x3_1, branch3x3_2])
            model.add(branch3x3)

            # branch3x3dbl
            model.add(Conv2d(n_filters=448, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=384, width=3, height=3, stride=1,
                             act='identity', include_bias=False))
            branch3x3dbl = BN(act='relu')
            model.add(branch3x3dbl)

            model.add(Conv2d(n_filters=384, width=3, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[branch3x3dbl]))
            branch3x3dbl_1 = BN(act='relu')
            model.add(branch3x3dbl_1)

            model.add(Conv2d(n_filters=384, width=1, height=3, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[branch3x3dbl]))
            branch3x3dbl_2 = BN(act='relu')
            model.add(branch3x3dbl_2)

            branch3x3dbl = Concat(act='identity',
                                  src_layers=[branch3x3dbl_1, branch3x3dbl_2])
            model.add(branch3x3dbl)

            # branch_pool
            model.add(Pooling(width=3, height=3, stride=1, pool='average',
                              src_layers=[concat]))
            model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                             act='identity', include_bias=False))
            branch_pool = BN(act='relu')
            model.add(branch_pool)

            # concat
            concat = Concat(act='identity',
                            src_layers=[branch1x1, branch3x3,
                                        branch3x3dbl, branch_pool])
            model.add(concat)


        # calculate dimensions for global average pooling
        w = max((width - 75) // 32 + 1, 1)
        h = max((height - 75) // 32 + 1, 1)

        # global average pooling
        model.add(Pooling(width=w, height=h, stride=1, pool='average',
                          padding=0, src_layers=[concat]))

        # output layer
        model.add(OutputLayer(n=n_classes))

        return model

    else:
        if pre_trained_weights_file is None:
            raise ValueError('\nThe pre-trained weights file is not specified.\n'
                             'Please follow the steps below to attach the '
                             'pre-trained weights:\n'
                             '1. go to the website '
                             'https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                             'and download the associated weight file.\n'
                             '2. upload the *.h5 file to '
                             'a server side directory which the CAS '
                             'session has access to.\n'
                             '3. specify the pre_train_weight_file using '
                             'the fully qualified server side path.')
        print('NOTE: Scale is set to 1/127.5, and offsets 1 to '
              'match Keras preprocessing.')
        model_cas = model_inceptionv3.InceptionV3_Model(
            s=conn, model_table=model_table, n_channels=n_channels,
            width=width, height=height, random_crop=random_crop,
            offsets=[1, 1, 1])

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, '
                              'n_classes will be set to 1000.', RuntimeWarning)
            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:
            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_trained_weights_file)

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<218'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True,
                                         **model.model_weights.to_table_params()))
            model._retrieve_('deeplearn.removelayer', model=model_table,
                             name='predictions')
            model._retrieve_('deeplearn.addlayer', model=model_table,
                             name='predictions',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['avg_pool'])
            model = Model.from_table(conn.CASTable(model_table))

            return model


def UNet(conn, n_channels=3, width=512, height=512, scale=1.0 / 255, n_classes = 2, init = None):
    inputs = Input(n_channels = n_channels, width = width, height = height, scale = scale, name = 'InputLayer_1')
    conv1 = Conv2d(64, 3, act = 'relu', init = init)(inputs)
    conv1 = Conv2d(64, 3, act = 'relu', init = init)(conv1)
    pool1 = Pooling(2)(conv1)

    conv2 = Conv2d(128, 3, act = 'relu', init = init)(pool1)
    conv2 = Conv2d(128, 3, act = 'relu', init = init)(conv2)
    pool2 = Pooling(2)(conv2)

    conv3 = Conv2d(256, 3, act = 'relu', init = init)(pool2)
    conv3 = Conv2d(256, 3, act = 'relu', init = init)(conv3)
    pool3 = Pooling(2)(conv3)

    conv4 = Conv2d(512, 3, act = 'relu', init = init)(pool3)
    conv4 = Conv2d(512, 3, act = 'relu', init = init)(conv4)
    pool4 = Pooling(2)(conv4)

    conv5 = Conv2d(1024, 3, act = 'relu', init = init)(pool4)
    conv5 = Conv2d(1024, 3, act = 'relu', init = init)(conv5)

    tconv6 = Conv2DTranspose(512, 3, stride = 2, act = 'relu', padding = 1, output_size = conv4._op.output_size,
                             init = init)(conv5)  # 64
    merge6 = Concat()([conv4, tconv6])
    conv6 = Conv2d(512, 3, act = 'relu', init = init)(merge6)
    conv6 = Conv2d(512, 3, act = 'relu', init = init)(conv6)

    tconv7 = Conv2DTranspose(256, 3, stride = 2, act = 'relu', padding = 1, output_size = conv3._op.output_size,
                             init = init)(conv6)  # 128
    merge7 = Concat()([conv3, tconv7])
    conv7 = Conv2d(256, 3, act = 'relu', init = init)(merge7)
    conv7 = Conv2d(256, 3, act = 'relu', init = init)(conv7)

    tconv8 = Conv2DTranspose(128, stride = 2, act = 'relu', padding = 1, output_size = conv2._op.output_size,
                             init = init)(conv7)  # 256
    merge8 = Concat()([conv2, tconv8])
    conv8 = Conv2d(128, 3, act = 'relu', init = init)(merge8)
    conv8 = Conv2d(128, 3, act = 'relu', init = init)(conv8)

    tconv9 = Conv2DTranspose(64, stride = 2, act = 'relu', padding = 1, output_size = conv1._op.output_size,
                             init = init)(conv8)  # 512
    merge9 = Concat()([conv1, tconv9])
    conv9 = Conv2d(64, 3, act = 'relu', init = init)(merge9)
    conv9 = Conv2d(64, 3, act = 'relu', init = init)(conv9)

    conv9 = Conv2d(n_classes, 3, act = 'identity', init = init)(conv9)

    seg1 = Segmentation(name = 'Segmentation_1')(conv9)
    model = Model(conn, inputs = inputs, outputs = seg1)
    model.compile()
    return model


def Nest_Net(conn, n_channels=1, width=512, height=512, scale=1.0 / 255, n_classes = 2, deep_supervision=True):
    def standard_unit(input_tensor, stage, nb_filter, kernel_size = 3):
        x = Conv2d(nb_filter, kernel_size, act = 'relu', name = 'conv' + stage + '_1')(input_tensor)
        x = Conv2d(nb_filter, kernel_size, act = 'relu', name = 'conv' + stage + '_2')(x)
        return x

    nb_filter = [32, 64, 128, 256, 512]

    inputs = Input(n_channels = n_channels, width = width, height = height, scale = scale, name = 'InputLayer_1')

    conv1_1 = standard_unit(inputs, stage='11', nb_filter=nb_filter[0])
    pool1 = Pooling(width = 2, height = 2, stride=2, name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = Pooling(width = 2, height = 2, stride=2, name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], 3, stride=2, act = 'relu', name='up12', padding=1,
                            output_size = conv1_1._op.output_size)(conv2_1)
    conv1_2 = Concat(name='merge12')([up1_2, conv1_1])
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = Pooling(width = 2, height = 2, stride=2, name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], 3, stride=2, act = 'relu', name='up22', padding=1,
                            output_size = conv2_1._op.output_size)(conv3_1)
    conv2_2 = Concat(name='merge22')([up2_2, conv2_1])
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], 3, stride=2, act = 'relu', name='up13', padding=1,
                            output_size = conv1_1._op.output_size)(conv2_2)
    conv1_3 = Concat(name='merge13')([up1_3, conv1_1, conv1_2])
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = Pooling(width = 2, height = 2, stride=2, name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], 3, stride=2, act = 'relu', name='up32', padding=1,
                            output_size = conv3_1._op.output_size)(conv4_1)
    conv3_2 = Concat(name='merge32')([up3_2, conv3_1])
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], 3, stride=2, act = 'relu', name='up23', padding=1,
                            output_size = conv2_1._op.output_size)(conv3_2)
    conv2_3 = Concat(name='merge23')([up2_3, conv2_1, conv2_2])
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], 3, stride=2, act = 'relu', name='up14', padding=1,
                            output_size = conv1_1._op.output_size)(conv2_3)
    conv1_4 = Concat(name='merge14')([up1_4, conv1_1, conv1_2, conv1_3])
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], 3, stride=2, act = 'relu', name='up42', padding=1,
                            output_size = conv4_1._op.output_size)(conv5_1)
    conv4_2 = Concat(name='merge42')([up4_2, conv4_1])
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], 3, stride=2, act = 'relu', name='up33', padding=1,
                            output_size = conv3_1._op.output_size)(conv4_2)
    conv3_3 = Concat(name='merge33')([up3_3, conv3_1, conv3_2])
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], 3, stride=2, act = 'relu', name='up24', padding=1,
                            output_size = conv2_1._op.output_size)(conv3_3)
    conv2_4 = Concat(name='merge24')([up2_4, conv2_1, conv2_2, conv2_3])
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], 3, stride=2, act = 'relu', name='up15', padding=1,
                            output_size = conv1_1._op.output_size)(conv2_4)
    conv1_5 = Concat(name='merge15')([up1_5, conv1_1, conv1_2, conv1_3, conv1_4])
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2d(n_classes, 1, act='identity', name='output_1')(conv1_2)
    nestnet_output_2 = Conv2d(n_classes, 1, act='identity', name='output_2')(conv1_3)
    nestnet_output_3 = Conv2d(n_classes, 1, act='identity', name='output_3')(conv1_4)
    nestnet_output_4 = Conv2d(n_classes, 1, act='identity', name='output_4')(conv1_5)

    seg1 = Segmentation(name = 'Segmentation_1')(nestnet_output_1)
    seg2 = Segmentation(name = 'Segmentation_2')(nestnet_output_2)
    seg3 = Segmentation(name = 'Segmentation_3')(nestnet_output_3)
    seg4 = Segmentation(name = 'Segmentation_4')(nestnet_output_4)

    if deep_supervision:
        model = Model(conn, inputs=inputs, outputs=[seg1, seg2, seg3, seg4])
    else:
        model = Model(conn, inputs=inputs, outputs=[seg4])

    model.compile()

    return model


def Faster_RCNN(conn, n_channels=3, width=1000, height=496, scale=1,
                offsets=[102.9801,115.9465,122.7717], random_mutation = 'none',
                n_classes=20, anchor_num_to_sample = 256, anchor_scale = [8, 16, 32], anchor_ratio = [0.5, 1, 2],
                base_anchor_size = 16, coord_type = 'coco', max_label_per_image = 200, proposed_roi_num_train = 2000,
                proposed_roi_num_score = 300, roi_train_sample_num = 128,
                roi_pooling_width = 7, roi_pooling_height = 7,
                nms_iou_threshold = 0.3, detection_threshold = 0.5, max_objec_num = 50):
    num_anchors = len(anchor_ratio) * len(anchor_scale)
    inp = Input(n_channels = n_channels, width = width, height = height, scale = scale, offsets = offsets,
                random_mutation = random_mutation)

    conv1_1 = Conv2d(n_filters = 64, width = 3, height = 3, stride = 1, name='conv1_1')(inp)
    conv1_2 = Conv2d(n_filters = 64, width = 3, height = 3, stride = 1, name='conv1_2')(conv1_1)
    pool1 = Pooling(width = 2, height = 2, stride = 2, pool = 'max', name='pool1')(conv1_2)

    conv2_1 = Conv2d(n_filters = 128, width = 3, height = 3, stride = 1, name = 'conv2_1')(pool1)
    conv2_2 = Conv2d(n_filters = 128, width = 3, height = 3, stride = 1, name = 'conv2_2')(conv2_1)
    pool2 = Pooling(width = 2, height = 2, stride = 2, pool = 'max')(conv2_2)

    conv3_1 = Conv2d(n_filters = 256, width = 3, height = 3, stride = 1, name = 'conv3_1')(pool2)
    conv3_2 = Conv2d(n_filters = 256, width = 3, height = 3, stride = 1, name = 'conv3_2')(conv3_1)
    conv3_3 = Conv2d(n_filters = 256, width = 3, height = 3, stride = 1, name = 'conv3_3')(conv3_2)
    pool3 = Pooling(width = 2, height = 2, stride = 2, pool = 'max')(conv3_3)

    conv4_1 = Conv2d(n_filters = 512, width = 3, height = 3, stride = 1, name = 'conv4_1')(pool3)
    conv4_2 = Conv2d(n_filters = 512, width = 3, height = 3, stride = 1, name = 'conv4_2')(conv4_1)
    conv4_3 = Conv2d(n_filters = 512, width = 3, height = 3, stride = 1, name = 'conv4_3')(conv4_2)
    pool4 = Pooling(width = 2, height = 2, stride = 2, pool = 'max')(conv4_3)

    conv5_1 = Conv2d(n_filters = 512, width = 3, height = 3, stride = 1, name = 'conv5_1')(pool4)
    conv5_2 = Conv2d(n_filters = 512, width = 3, height = 3, stride = 1, name = 'conv5_2')(conv5_1)
    conv5_3 = Conv2d(n_filters = 512, width = 3, height = 3, stride = 1, name = 'conv5_3')(conv5_2)

    rpn_conv = Conv2d(width = 3, n_filters = 512, name = 'rpn_conv_3x3')(conv5_3)
    rpn_score = Conv2d(act = 'identity', width = 1, n_filters = ((n_classes + 1 + 4) * num_anchors),
                       name = 'rpn_score')(rpn_conv)

    rp1 = RegionProposal(max_label_per_image = max_label_per_image, base_anchor_size = base_anchor_size,
                         coord_type = coord_type, name = 'rois', anchor_num_to_sample = anchor_num_to_sample,
                         anchor_scale = anchor_scale, anchor_ratio = anchor_ratio,
                         proposed_roi_num_train = proposed_roi_num_train,
                         proposed_roi_num_score = proposed_roi_num_score,
                         roi_train_sample_num = roi_train_sample_num
                         )(rpn_score)
    roipool1 = ROIPooling(output_height=roi_pooling_height, output_width=roi_pooling_width,
                          spatial_scale=conv5_3.shape[0]/width,
                          name = 'roi_pooling')([conv5_3 + rp1])

    fc6 = Dense(n = 4096, act = 'relu', name = 'fc6')(roipool1)
    fc7 = Dense(n = 4096, act = 'relu', name = 'fc7')(fc6)
    cls1 = Dense(n = n_classes+1, act = 'identity', name = 'cls_score')(fc7)
    reg1 = Dense(n = (n_classes+1)*4, act = 'identity', name = 'bbox_pred')(fc7)
    fr1 = FastRCNN(nms_iou_threshold = nms_iou_threshold, max_label_per_image = max_label_per_image,
                   max_objec_num = max_objec_num,  detection_threshold = detection_threshold,
                   class_number = n_classes, name = 'fastrcnn')([cls1, reg1, rp1])
    faster_rcnn = Model(conn, inp, fr1)
    faster_rcnn.compile()
    return faster_rcnn

