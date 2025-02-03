"""
    Created by Jiazhou Wang on 2017/8/5
"""

import logging

from keras.regularizers import l2
from keras.layers import Add
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import Input
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import Activation

from lib.resnet50_fcn import resnet50
from lib.vgg16_fcn import vgg16
from lib.vgg19_fcn import vgg19

logger = logging.getLogger('fcn_model.set_model')


def set_model(base_net='vgg16', weight_decay=0, dropout_rate=0,
              **kwargs):

    weight_decay = float(weight_decay)
    dropout_rate = float(dropout_rate)
    assert isinstance(base_net, str)
    assert isinstance(weight_decay, float)
    assert isinstance(dropout_rate, float)
    if base_net == 'dense_net' or base_net =='stacked_net':
        if 'growth_rate' in kwargs:
            growth_rate = int(kwargs['growth_rate'])
        else:
            growth_rate = 32
            logger.warning('use default growth_rate parameter: 32')
        if 'nb_filter' in kwargs:
            nb_filter = int(kwargs['nb_filter'])
        else:
            nb_filter = 64
            logger.warning('use default nb_filter parameter  : 64')
        if 'reduction' in kwargs:
            reduction = float(kwargs['reduction'])
        else:
            reduction = 0
            logger.warning('use default reduction parameter  : 0')
        assert isinstance(growth_rate, int)
        assert isinstance(nb_filter, int)
        assert isinstance(reduction, float)

    input_img_shape = [512, 512, 5]
    input_img = Input(shape=input_img_shape, name="input_img")

    if base_net == 'vgg16':
        stage1_out, stage2_out, stage3_out, stage4_out, stage5_out = \
            vgg16(input_img, weight_decay, dropout_rate)
    if base_net == 'vgg19':
        stage1_out, stage2_out, stage3_out, stage4_out, stage5_out = \
            vgg19(input_img, weight_decay, dropout_rate)
    if base_net == 'resnet50':
        stage1_out, stage2_out, stage3_out, stage4_out, stage5_out = \
            resnet50(input_img, weight_decay, dropout_rate)

    stage5_out_shape = stage5_out.shape
    stage4_out_shape = stage4_out.shape
    stage3_out_shape = stage3_out.shape
    stage2_out_shape = stage2_out.shape
    stage1_out_shape = stage1_out.shape

    logger.info('stage5_out_shape: ' + str(stage5_out_shape))
    logger.info('stage4_out_shape: ' + str(stage4_out_shape))
    logger.info('stage3_out_shape: ' + str(stage3_out_shape))
    logger.info('stage2_out_shape: ' + str(stage2_out_shape))
    logger.info('stage1_out_shape: ' + str(stage1_out_shape))

    stage5_ch = int(stage5_out_shape[3])
    stage4_ch = int(stage4_out_shape[3])
    stage3_ch = int(stage3_out_shape[3])
    stage2_ch = int(stage2_out_shape[3])
    stage1_ch = int(stage1_out_shape[3])

    upscore4 = Conv2DTranspose(filters=stage4_ch,
                               kernel_size=(2, 2),
                               strides=(2, 2),
                               padding='same',
                               name='upscore4', use_bias=False,
                               kernel_initializer='he_uniform',
                               kernel_regularizer=l2(weight_decay))(stage5_out)
    upscore4 = BatchNormalization(name='upscore4_bn')(upscore4)
    upscore4 = Activation('relu')(upscore4)

    fuse4 = Add(name='fuse4_add')([upscore4, stage4_out])
    fuse4 = Conv2D(stage4_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='fuse4_conv1', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse4)
    fuse4 = BatchNormalization(name='fuse4_bn1')(fuse4)
    fuse4 = Activation('relu')(fuse4)
    fuse4 = Conv2D(stage4_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='fuse4_conv2', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse4)
    fuse4 = BatchNormalization(name='fuse4_bn2')(fuse4)
    fuse4 = Activation('relu')(fuse4)

    upscore3 = Conv2DTranspose(filters=stage3_ch,
                               kernel_size=(2, 2),
                               strides=(2, 2),
                               padding='same',
                               name='upscore3', use_bias=False,
                               kernel_initializer='he_uniform',
                               kernel_regularizer=l2(weight_decay))(fuse4)
    upscore3 = BatchNormalization(name='upscore3_bn')(upscore3)
    upscore3 = Activation('relu')(upscore3)
    fuse3 = Add(name='fuse3_add')([upscore3, stage3_out])
    fuse3 = Conv2D(stage3_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='fuse3_conv1', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse3)
    fuse3 = BatchNormalization(name='fuse3_bn1')(fuse3)
    fuse3 = Activation('relu')(fuse3)
    fuse3 = Conv2D(stage3_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='fuse3_conv2', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse3)
    fuse3 = BatchNormalization(name='fuse3_bn2')(fuse3)
    fuse3 = Activation('relu')(fuse3)

    upscore2 = Conv2DTranspose(filters=stage2_ch,
                               kernel_size=(2, 2),
                               strides=(2, 2),
                               padding='same',
                               name='upscore2', use_bias=False,
                               kernel_initializer='he_uniform',
                               kernel_regularizer=l2(weight_decay))(fuse3)
    upscore2 = BatchNormalization(name='upscore2_bn')(upscore2)
    upscore2 = Activation('relu')(upscore2)
    fuse2 = Add(name='fuse2_add')([upscore2, stage2_out])
    fuse2 = Conv2D(stage2_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='fuse2_conv1', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse2)
    fuse2 = BatchNormalization(name='fuse2_bn1')(fuse2)
    fuse2 = Activation('relu')(fuse2)
    fuse2 = Conv2D(stage2_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='fuse2_conv2', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse2)
    fuse2 = BatchNormalization(name='fuse2_bn2')(fuse2)
    fuse2 = Activation('relu')(fuse2)

    upscore1 = Conv2DTranspose(filters=stage1_ch,
                               kernel_size=(2, 2),
                               strides=(2, 2),
                               padding='same',
                               name='upscore1', use_bias=False,
                               kernel_initializer='he_uniform',
                               kernel_regularizer=l2(weight_decay))(fuse2)
    upscore1 = BatchNormalization(name='upscore1_bn')(upscore1)
    upscore1 = Activation('relu')(upscore1)
    fuse1 = Add(name='fuse1_add')([upscore1, stage1_out])
    fuse1 = Conv2D(stage1_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='fuse1_conv1', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse1)
    fuse1 = BatchNormalization(name='fuse1_bn1')(fuse1)
    fuse1 = Activation('relu')(fuse1)
    fuse1 = Conv2D(stage1_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='fuse1_conv2', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse1)
    fuse1 = BatchNormalization(name='fuse1_bn2')(fuse1)
    fuse1 = Activation('relu')(fuse1)

    upscore0 = UpSampling2D(size=(2, 2), name='upscore0')(fuse1)
    fuse0 = Concatenate(name='fuse0_concatenate')([upscore0, input_img])

    fuse0 = Conv2D(stage1_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='mask_conv1', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse0)
    fuse0 = BatchNormalization(name='fuse0_bn1')(fuse0)
    fuse0 = Activation('relu')(fuse0)
    fuse0 = Conv2D(stage1_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='mask_conv2', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse0)
    fuse0 = BatchNormalization(name='fuse0_bn2')(fuse0)
    fuse0 = Activation('relu')(fuse0)

    output_mask = Conv2D(8, (1, 1),
                         strides=(1, 1),
                         padding='same',
                         name='output_mask_conv',
                         kernel_initializer='he_uniform',
                         kernel_regularizer=l2(weight_decay))(fuse0)
    output_mask = Activation('sigmoid', name='output_mask')(output_mask)

    # dose_input = Concatenate()([input_img, output_mask])

    stage1_out, stage2_out, stage3_out, stage4_out, stage5_out = vgg16(output_mask, weight_decay, dropout_rate, prefix='dose_')

    stage5_out_shape = stage5_out.shape
    stage4_out_shape = stage4_out.shape
    stage3_out_shape = stage3_out.shape
    stage2_out_shape = stage2_out.shape
    stage1_out_shape = stage1_out.shape

    logger.info('stage5_out_shape: ' + str(stage5_out_shape))
    logger.info('stage4_out_shape: ' + str(stage4_out_shape))
    logger.info('stage3_out_shape: ' + str(stage3_out_shape))
    logger.info('stage2_out_shape: ' + str(stage2_out_shape))
    logger.info('stage1_out_shape: ' + str(stage1_out_shape))

    stage5_ch = int(stage5_out_shape[3])
    stage4_ch = int(stage4_out_shape[3])
    stage3_ch = int(stage3_out_shape[3])
    stage2_ch = int(stage2_out_shape[3])
    stage1_ch = int(stage1_out_shape[3])

    upscore4 = Conv2DTranspose(filters=stage4_ch,
                               kernel_size=(2, 2),
                               strides=(2, 2),
                               padding='same',
                               name='dose_upscore4', use_bias=False,
                               kernel_initializer='he_uniform',
                               kernel_regularizer=l2(weight_decay))(stage5_out)
    upscore4 = BatchNormalization(name='dose_upscore4_bn')(upscore4)
    upscore4 = Activation('relu')(upscore4)

    fuse4 = Add(name='dose_fuse4_add')([upscore4, stage4_out])
    fuse4 = Conv2D(stage4_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='dose_fuse4_conv1', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse4)
    fuse4 = BatchNormalization(name='dose_fuse4_bn1')(fuse4)
    fuse4 = Activation('relu')(fuse4)
    fuse4 = Conv2D(stage4_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='dose_fuse4_conv2', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse4)
    fuse4 = BatchNormalization(name='dose_fuse4_bn2')(fuse4)
    fuse4 = Activation('relu')(fuse4)

    upscore3 = Conv2DTranspose(filters=stage3_ch,
                               kernel_size=(2, 2),
                               strides=(2, 2),
                               padding='same',
                               name='dose_upscore3', use_bias=False,
                               kernel_initializer='he_uniform',
                               kernel_regularizer=l2(weight_decay))(fuse4)
    upscore3 = BatchNormalization(name='dose_upscore3_bn')(upscore3)
    upscore3 = Activation('relu')(upscore3)
    fuse3 = Add(name='dose_fuse3_add')([upscore3, stage3_out])
    fuse3 = Conv2D(stage3_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='dose_fuse3_conv1', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse3)
    fuse3 = BatchNormalization(name='dose_fuse3_bn1')(fuse3)
    fuse3 = Activation('relu')(fuse3)
    fuse3 = Conv2D(stage3_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='dose_fuse3_conv2', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse3)
    fuse3 = BatchNormalization(name='dose_fuse3_bn2')(fuse3)
    fuse3 = Activation('relu')(fuse3)

    upscore2 = Conv2DTranspose(filters=stage2_ch,
                               kernel_size=(2, 2),
                               strides=(2, 2),
                               padding='same',
                               name='dose_upscore2', use_bias=False,
                               kernel_initializer='he_uniform',
                               kernel_regularizer=l2(weight_decay))(fuse3)
    upscore2 = BatchNormalization(name='dose_upscore2_bn')(upscore2)
    upscore2 = Activation('relu')(upscore2)
    fuse2 = Add(name='dose_fuse2_add')([upscore2, stage2_out])
    fuse2 = Conv2D(stage2_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='dose_fuse2_conv1', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse2)
    fuse2 = BatchNormalization(name='dose_fuse2_bn1')(fuse2)
    fuse2 = Activation('relu')(fuse2)
    fuse2 = Conv2D(stage2_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='dose_fuse2_conv2', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse2)
    fuse2 = BatchNormalization(name='dose_fuse2_bn2')(fuse2)
    fuse2 = Activation('relu')(fuse2)

    upscore1 = Conv2DTranspose(filters=stage1_ch,
                               kernel_size=(2, 2),
                               strides=(2, 2),
                               padding='same',
                               name='dose_upscore1', use_bias=False,
                               kernel_initializer='he_uniform',
                               kernel_regularizer=l2(weight_decay))(fuse2)
    upscore1 = BatchNormalization(name='dose_upscore1_bn')(upscore1)
    upscore1 = Activation('relu')(upscore1)
    fuse1 = Add(name='dose_fuse1_add')([upscore1, stage1_out])
    fuse1 = Conv2D(stage1_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='dose_fuse1_conv1', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse1)
    fuse1 = BatchNormalization(name='dose_fuse1_bn1')(fuse1)
    fuse1 = Activation('relu')(fuse1)
    fuse1 = Conv2D(stage1_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='dose_fuse1_conv2', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse1)
    fuse1 = BatchNormalization(name='dose_fuse1_bn2')(fuse1)
    fuse1 = Activation('relu')(fuse1)

    upscore0 = UpSampling2D(size=(2, 2), name='dose_upscore0')(fuse1)
    fuse0 = Concatenate(name='dose_fuse0_concatenate')([upscore0, input_img])

    fuse0 = Conv2D(stage1_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='dose_mask_conv1', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse0)
    fuse0 = BatchNormalization(name='dose_fuse0_bn1')(fuse0)
    fuse0 = Activation('relu')(fuse0)
    fuse0 = Conv2D(stage1_ch, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   name='dose_mask_conv2', use_bias=False,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(fuse0)
    fuse0 = BatchNormalization(name='dose_fuse0_bn2')(fuse0)
    fuse0 = Activation('relu')(fuse0)

    output_dose = Conv2D(1, (1, 1),
                         strides=(1, 1),
                         padding='same',
                         name='output_dose_conv',
                         kernel_initializer='he_uniform',
                         kernel_regularizer=l2(weight_decay))(fuse0)
    output_dose = Activation('relu', name='output_dose')(output_dose)

    model = Model(inputs=input_img, outputs=[output_dose, output_mask])

    return model
