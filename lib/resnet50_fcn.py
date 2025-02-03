"""
    Created by Jiazhou Wang on 2017/8/5
"""

from keras.regularizers import l2
from keras.layers import Activation
from keras.layers import Add
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization

bn_axis = 3


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   weight_decay=0, trainable_setting=True):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters=filters1, kernel_size=(1, 1),
               name=conv_name_base + '2a',
               kernel_regularizer=l2(weight_decay),
               trainable=trainable_setting)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters2, kernel_size=kernel_size,
               padding='same', name=conv_name_base + '2b',
               kernel_regularizer=l2(weight_decay),
               trainable=trainable_setting)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters3, kernel_size=(1, 1),
               name=conv_name_base + '2c',
               kernel_regularizer=l2(weight_decay),
               trainable=trainable_setting)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2),  weight_decay=0, trainable_setting=True):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters=filters1, kernel_size=(1, 1),
               strides=strides, name=conv_name_base + '2a',
               kernel_regularizer=l2(weight_decay),
               trainable=trainable_setting)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters2, kernel_size=kernel_size,
               padding='same', name=conv_name_base + '2b',
               kernel_regularizer=l2(weight_decay),
               trainable=trainable_setting)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters3, kernel_size=(1, 1),
               name=conv_name_base + '2c',
               kernel_regularizer=l2(weight_decay),
               trainable=trainable_setting)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters=filters3, kernel_size=(1, 1),
                      strides=strides, name=conv_name_base + '1',
                      kernel_regularizer=l2(weight_decay),
                      trainable=trainable_setting)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50(img_input, weight_decay=0, drop_rate=0, trainable_setting=True,):
    # stage1_out.shape (?, 256, 256, 64)
    # stage2_out.shape (?, 128, 128, 128)
    # stage3_out.shape (?, 64, 64, 256)
    # stage4_out.shape (?, 32, 32, 512)
    # stage5_out.shape (?, 16, 16, 1024)

    x = Conv2D(filters=64, kernel_size=(3, 3),
               padding='same', name='stage1_conv1', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(img_input)
    x = BatchNormalization(name='stage1_bn1')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3),
               padding='same', name='stage1_conv2', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(name='stage1_bn2')(x)
    x = Activation('relu')(x)

    if drop_rate == 0:
        stage1_out = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='stage1_pooling')(x)
    else:
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='stage1_pooling')(x)
        stage1_out = Dropout(drop_rate)(x)

    x = conv_block(stage1_out, kernel_size=3, filters=[64, 64, 128], stage=2, block='a',
                   weight_decay=weight_decay, trainable_setting=trainable_setting)
    x = identity_block(x, kernel_size=3, filters=[64, 64, 128], stage=2, block='b',
                       weight_decay=weight_decay, trainable_setting=trainable_setting)
    x = identity_block(x, kernel_size=3, filters=[64, 64, 128], stage=2, block='c',
                       weight_decay=weight_decay, trainable_setting=trainable_setting)
    stage2_out = x

    x = conv_block(stage2_out, kernel_size=3, filters=[128, 128, 256], stage=3, block='a',
                   weight_decay=weight_decay, trainable_setting=trainable_setting)
    x = identity_block(x, kernel_size=3, filters=[128, 128, 256], stage=3, block='b',
                       weight_decay=weight_decay, trainable_setting=trainable_setting)
    x = identity_block(x, kernel_size=3, filters=[128, 128, 256], stage=3, block='c',
                       weight_decay=weight_decay, trainable_setting=trainable_setting)
    x = identity_block(x, kernel_size=3, filters=[128, 128, 256], stage=3, block='d',
                       weight_decay=weight_decay, trainable_setting=trainable_setting)
    stage3_out = x

    x = conv_block(stage3_out, kernel_size=3, filters=[256, 256, 512], stage=4, block='a',
                   weight_decay=weight_decay, trainable_setting=trainable_setting)
    x = identity_block(x, kernel_size=3, filters=[256, 256, 512], stage=4, block='b',
                       weight_decay=weight_decay, trainable_setting=trainable_setting)
    x = identity_block(x, kernel_size=3, filters=[256, 256, 512], stage=4, block='c',
                       weight_decay=weight_decay, trainable_setting=trainable_setting)
    x = identity_block(x, kernel_size=3, filters=[256, 256, 512], stage=4, block='d',
                       weight_decay=weight_decay, trainable_setting=trainable_setting)
    x = identity_block(x, kernel_size=3, filters=[256, 256, 512], stage=4, block='e',
                       weight_decay=weight_decay, trainable_setting=trainable_setting)
    x = identity_block(x, kernel_size=3, filters=[256, 256, 512], stage=4, block='f',
                       weight_decay=weight_decay, trainable_setting=trainable_setting)
    stage4_out = x

    x = conv_block(stage4_out, kernel_size=3, filters=[512, 512, 1024], stage=5, block='a',
                   weight_decay=weight_decay, trainable_setting=trainable_setting)
    x = identity_block(x, kernel_size=3, filters=[512, 512, 1024], stage=5, block='b',
                       weight_decay=weight_decay, trainable_setting=trainable_setting)
    x = identity_block(x, kernel_size=3, filters=[512, 512, 1024], stage=5, block='c',
                       weight_decay=weight_decay, trainable_setting=trainable_setting)
    stage5_out = x

    return stage1_out, stage2_out, stage3_out, stage4_out, stage5_out
