# [source]: N. Khedkar (project partner) - 4_training/model_v2.py, renamed dilated_U_net.py
# [source]: Yu and Koltun (2016) - dilated convolutions, arXiv:1511.07122
# [source]: Oktay et al. (2018) - attention gates, arXiv:1804.03999
# [source]: Abraham and Khan (2019) - attention gates with the focal Tversky loss for small targets, arXiv:1810.07842

# dilated_U_net
# a 3 level U-Net with a dilated bottleneck and an attention gate on every skip
# connection. 3 poolings keep a 1 km crater, about 10 px, resolvable at the
# bottleneck, and the dilated convolutions widen the receptive field for large
# craters without a 4th pooling. same buildModel(params) interface as the other
# model files.
# parameters:
#         none, depth and dilation rates are fixed below and the rest comes from params
# outputs:
#         a buildModel function


import keras
from keras.layers import (Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate,
                          Dropout, BatchNormalization, Activation, add, multiply)
from keras.regularizers import l2


# pooling levels and bottleneck dilation rates, fixed for every run
DEPTH = 3
DILATION_RATES = (1, 2, 4)


# _conv_block
# two convolutions, each followed by batch normalisation and relu.
# parameters:
#         x: input tensor
#         f: filters in both convolutions
#         FL: kernel size
#         init: kernel initialiser
#         reg: kernel regulariser
# outputs:
#         tensor, same height and width as x
def _conv_block(x, f, FL, init, reg):
    for _ in range(2):
        x = Conv2D(f, FL, padding='same', kernel_initializer=init,
                   kernel_regularizer=reg, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x


# _dilated_bottleneck
# three stacked convolutions at dilation rates 1, 2 and 4, so the bottleneck
# sees large craters without pooling away the small ones.
# parameters:
#         x: input tensor
#         f: filters in every convolution
#         FL: kernel size
#         init: kernel initialiser
#         reg: kernel regulariser
# outputs:
#         tensor, same height and width as x
def _dilated_bottleneck(x, f, FL, init, reg):
    for rate in DILATION_RATES:
        x = Conv2D(f, FL, padding='same', dilation_rate=rate,
                   kernel_initializer=init, kernel_regularizer=reg,
                   use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x


# _attention_gate
# additive attention gate. the decoder map weights the encoder skip map pixel by
# pixel, so background is suppressed before the two are concatenated.
# parameters:
#         skip: encoder feature map
#         gating: decoder feature map at the same resolution
#         inter: filters in the intermediate 1x1 convolutions
#         init: kernel initialiser
#         reg: kernel regulariser
# outputs:
#         tensor, the skip map multiplied by its attention coefficients in [0, 1]
def _attention_gate(skip, gating, inter, init, reg):
    theta = Conv2D(inter, 1, kernel_initializer=init, kernel_regularizer=reg)(skip)
    phi = Conv2D(inter, 1, kernel_initializer=init, kernel_regularizer=reg)(gating)
    act = Activation('relu')(add([theta, phi]))
    psi = Conv2D(1, 1, activation='sigmoid', kernel_initializer=init,
                 kernel_regularizer=reg)(act)
    return multiply([skip, psi])


# buildModel
# builds the Dilated U-Net, with a sigmoid output for the rim mask.
# parameters:
#         params: dict read for dim, input_channels, n_filters, FL, init, lmbda, dropout
# outputs:
#         keras model, input (dim, dim, input_channels), output (dim, dim, 1)
def buildModel(params):
    dim = params['dim']
    ch = params['input_channels']
    base = params['n_filters']
    FL = params['FL']
    init = params['init']
    reg = l2(params.get('lmbda', 1e-6))
    drop = params['dropout']

    inp = keras.Input(shape=(dim, dim, ch))

    # Encoder, DEPTH blocks of two convolutions then a 2x2 max pool
    skips = []
    x = inp
    for d in range(DEPTH):
        x = _conv_block(x, base * 2 ** d, FL, init, reg)
        skips.append(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Dilated bottleneck
    x = _dilated_bottleneck(x, base * 2 ** DEPTH, FL, init, reg)

    # Decoder, each skip gated by attention before the merge
    for d in reversed(range(DEPTH)):
        f = base * 2 ** d
        x = Conv2DTranspose(f, 2, strides=2, padding='same')(x)
        gated_skip = _attention_gate(skips[d], x, f, init, reg)
        x = Concatenate()([x, gated_skip])
        x = Dropout(drop)(x)
        x = _conv_block(x, f, FL, init, reg)

    # Output layer
    out = Conv2D(1, 1, activation='sigmoid')(x)

    return keras.Model(inp, out, name='dilated_U_net')
