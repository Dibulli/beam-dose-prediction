import numpy as np
import keras
import torch
from lib.network.wnet import Wnet as UNet

pytorch_network = UNet(n_channels=5, add_channels=1, n_classes=8)
from lib.set_model import set_model

keras_network = set_model()
from lib import util
from collections import OrderedDict
import h5py


VAR_AFFIX = ':0' if keras.backend.backend() == 'tensorflow' else ''

KERAS_GAMMA_KEY = 'gamma' + VAR_AFFIX
KERAS_KERNEL_KEY = 'kernel' + VAR_AFFIX
KERAS_ALPHA_KEY = 'alpha' + VAR_AFFIX
KERAS_BIAS_KEY = 'bias' + VAR_AFFIX
KERAS_BETA_KEY = 'beta' + VAR_AFFIX
KERAS_MOVING_MEAN_KEY = 'moving_mean' + VAR_AFFIX
KERAS_MOVING_VARIANCE_KEY = 'moving_variance' + VAR_AFFIX
KERAS_EPSILON = 1e-3
PYTORCH_EPSILON = 1e-5


def check_for_missing_layers(keras_names, pytorch_layer_names, verbose):

    if verbose:
        print("Layer names in PyTorch state_dict", pytorch_layer_names)
        print("Layer names in Keras HDF5", keras_names)
        print(len(pytorch_layer_names), len(keras_names))

    if not all(x in keras_names for x in pytorch_layer_names):
        missing_layers = list(set(pytorch_layer_names) - set(keras_names))
        raise Exception("Missing layer(s) in Keras HDF5 that are present" +
                        " in state_dict: {}".format(missing_layers))


def keras_to_pytorch(keras_model, pytorch_model,
                     flip_filters=None, verbose=True):

    # If not specifically set, determine whether to flip filters automatically
    # for the right backend.
    if flip_filters is None:
        flip_filters = not keras.backend.backend() == 'tensorflow'

    keras_model.save('D:\\work_space\\Philips_internship\\model\\model.hd5')
    input_state_dict = pytorch_model.state_dict()
    pytorch_layer_names = util.state_dict_layer_names(input_state_dict)

    with h5py.File('D:\\work_space\\Philips_internship\\model\\model.hd5', 'r') as f:
        model_weights = f['model_weights']
        layer_names = list(map(str, model_weights.keys()))
        check_for_missing_layers(layer_names, pytorch_layer_names, verbose)
        state_dict = OrderedDict()

        for layer in pytorch_layer_names:

            params = util.dig_to_params(model_weights[layer])

            weight_key = layer + '.weight'
            bias_key = layer + '.bias'
            running_mean_key = layer + '.running_mean'
            running_var_key = layer + '.running_var'

            # Load weights (or other learned parameters)
            if weight_key in input_state_dict:
                if KERAS_GAMMA_KEY in params:
                    weights = params[KERAS_GAMMA_KEY][:]
                elif KERAS_KERNEL_KEY in params:
                    weights = params[KERAS_KERNEL_KEY][:]
                else:
                    weights = np.squeeze(params[KERAS_ALPHA_KEY][:])

                weights = convert_weights(weights,
                                          to_keras=True,
                                          flip_filters=flip_filters)

                state_dict[weight_key] = torch.from_numpy(weights)

            # Load bias
            if bias_key in input_state_dict:
                if running_var_key in input_state_dict:
                    bias = params[KERAS_BETA_KEY][:]
                else:
                    bias = params[KERAS_BIAS_KEY][:]
                state_dict[bias_key] = torch.from_numpy(
                    bias.transpose())

            # Load batch normalization running mean
            if running_mean_key in input_state_dict:
                running_mean = params[KERAS_MOVING_MEAN_KEY][:]
                state_dict[running_mean_key] = torch.from_numpy(
                    running_mean.transpose())

            # Load batch normalization running variance
            if running_var_key in input_state_dict:
                running_var = params[KERAS_MOVING_VARIANCE_KEY][:]
                # account for difference in epsilon used
                running_var += KERAS_EPSILON - PYTORCH_EPSILON
                state_dict[running_var_key] = torch.from_numpy(
                    running_var.transpose())

    pytorch_model.load_state_dict(state_dict)
    return pytorch_model


def convert_weights(weights, to_keras, flip_filters, flip_channels=False):

    if len(weights.shape) == 3:  # 1D conv
        weights = weights.transpose()

        if flip_channels:
            weights = weights[::-1]

        if flip_filters:
            weights = weights[..., ::-1].copy()

    if len(weights.shape) == 4:  # 2D conv
        if to_keras:  # D1 D2 F F
            weights = weights.transpose(3, 2, 0, 1)
        else:
            weights = weights.transpose(2, 3, 1, 0)

        if flip_channels:
            weights = weights[::-1, ::-1]
        if flip_filters:
            weights = weights[..., ::-1, ::-1].copy()

    elif len(weights.shape) == 5:  # 3D conv
        if to_keras:  # D1 D2 D3 F F
            weights = weights.transpose(4, 3, 0, 1, 2)
        else:
            weights = weights.transpose(2, 3, 4, 1, 0)

        if flip_channels:
            weights = weights[::-1, ::-1, ::-1]

        if flip_filters:
            weights = weights[..., ::-1, ::-1, ::-1].copy()
    else:
        weights = weights.transpose()

    return weights


def main(keras_network, pytorch_network):
    # transfer keras model to pytorch
    # keras_network.summary()
    # keras_network.load_weights("D:\\work_space\\Philips_internship\\model\\model.hd5")
    pytorch_network = keras_to_pytorch(keras_network, pytorch_network)
    torch.save(pytorch_network.state_dict(), "D:\\work_space\\Philips_internship\\model\\auto_seg.pth")


main(keras_network, pytorch_network)
