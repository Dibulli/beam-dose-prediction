"""
    Created by Jiazhou Wang on 2017/8/5
"""

from keras.regularizers import l2
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation


# stage1_out.shape (?, 256, 256, 64)
# stage2_out.shape (?, 128, 128, 128)
# stage3_out.shape (?, 64, 64, 256)
# stage4_out.shape (?, 32, 32, 512)
# stage5_out.shape (?, 16, 16, 1024)


def vgg19(img_input, weight_decay=0, drop_rate=0):
    # Block 1
    x = Conv2D(filters=64, kernel_size=(3, 3),
               padding='same', name='block1_conv1', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(img_input)
    x = BatchNormalization(name='block1_bn1')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3),
               padding='same', name='block1_conv2', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(name='block1_bn2')(x)
    x = Activation('relu')(x)
    if drop_rate == 0:
        stage1_out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    else:
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        stage1_out = Dropout(drop_rate)(x)

    # Block 2
    x = Conv2D(filters=128, kernel_size=(3, 3),
               padding='same', name='block2_conv1', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(stage1_out)
    x = BatchNormalization(name='block2_bn1')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3),
               padding='same', name='block2_conv2', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(name='block2_bn2')(x)
    x = Activation('relu')(x)
    if drop_rate == 0:
        stage2_out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    else:
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        stage2_out = Dropout(drop_rate)(x)

    # Block 3
    x = Conv2D(filters=256, kernel_size=(3, 3),
               padding='same', name='block3_conv1', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(stage2_out)
    x = BatchNormalization(name='block3_bn1')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),
               padding='same', name='block3_conv2', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(name='block3_bn2')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),
               padding='same', name='block3_conv3', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(name='block3_bn3')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),
               padding='same', name='block3_conv4', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(name='block3_bn4')(x)
    x = Activation('relu')(x)
    if drop_rate == 0:
        stage3_out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    else:
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        stage3_out = Dropout(drop_rate)(x)

    # Block 4
    x = Conv2D(filters=512, kernel_size=(3, 3),
               padding='same', name='block4_conv1', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(stage3_out)
    x = BatchNormalization(name='block4_bn1')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               padding='same', name='block4_conv2', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(name='block4_bn2')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               padding='same', name='block4_conv3', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(name='block4_bn3')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               padding='same', name='block4_conv4', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(name='block4_bn4')(x)
    x = Activation('relu')(x)
    if drop_rate == 0:
        stage4_out = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    else:
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        stage4_out = Dropout(drop_rate)(x)

    # Block 5
    x = Conv2D(filters=1024, kernel_size=(3, 3),
               padding='same', name='block5_conv1', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(stage4_out)
    x = BatchNormalization(name='block5_bn1')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=1024, kernel_size=(3, 3),
               padding='same', name='block5_conv2', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(name='block5_bn2')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=1024, kernel_size=(3, 3),
               padding='same', name='block5_conv3', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(name='block5_bn3')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=1024, kernel_size=(3, 3),
               padding='same', name='block5_conv4', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(name='block5_bn4')(x)
    x = Activation('relu')(x)
    if drop_rate == 0:
        stage5_out = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    else:
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        stage5_out = Dropout(drop_rate)(x)

    return stage1_out, stage2_out, stage3_out, stage4_out, stage5_out
