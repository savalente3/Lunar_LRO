# [source]: N. Khedkar (project partner) - 4_training/model_v2.py
# [source]: Yu and Koltun (2016) - dilated convolutions, arXiv:1511.07122
# [source]: Oktay et al. (2018) - attention U-Net, arXiv:1804.03999

# model_v2
# a U-Net aimed at the sub 2 km craters U_Net_v1 misses. three or four poolings
# instead of four, an optional dilated bottleneck that widens the receptive field
# without pooling, and optional attention gates on the skips. same
# buildModel(params) interface as the other model files.
# parameters:
#         v2_depth: pooling levels, 3 or 4
#         v2_dilation: dilated bottleneck instead of a plain one
#         v2_attention: attention gates on the skip connections
# outputs:
#         a buildModel function


import keras
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout, BatchNormalization, Activation, add, multiply
from keras.regularizers import l2


# convBlock
# two 3x3 convolutions, each followed by batch normalisation and relu.
# parameters:
#         x: input tensor
#         n_filters: filters in both convolutions
#         params: dict read for FL, init, lmbda
# outputs:
#         tensor, same height and width as x
def convBlock(x, n_filters, params):

    for _ in range(2):
        x = Conv2D(
            n_filters,
            params['FL'],
            kernel_initializer=params['init'],
            kernel_regularizer=l2(params['lmbda']),
            padding='same',
            use_bias=False
        )(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x


# dilatedBottleneck
# stacked convolutions at rates 1, 2 and 4. widens the receptive field enough to
# see large crater context without pooling away the small craters.
# parameters:
#         x: input tensor
#         n_filters: filters in each convolution
#         params: dict read for FL, init, lmbda
# outputs:
#         tensor, same height and width as x
def dilatedBottleneck(x, n_filters, params):

    for rate in (1, 2, 4):
        x = Conv2D(
            n_filters,
            params['FL'],
            dilation_rate=rate,
            kernel_initializer=params['init'],
            kernel_regularizer=l2(params['lmbda']),
            padding='same',
            use_bias=False
        )(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x


# attentionGate
# weights a skip connection by what the decoder is looking for, so crater-like
# regions survive the concatenate and background is damped.
# parameters:
#         skip: encoder tensor being gated
#         gating: decoder tensor supplying the query
#         n_filters: filters in the intermediate projection
#         params: dict read for init, lmbda
# outputs:
#         tensor, skip scaled by the learned attention
def attentionGate(skip, gating, n_filters, params):

    theta = Conv2D(n_filters, 1, kernel_initializer=params['init'], kernel_regularizer=l2(params['lmbda']))(skip)
    phi = Conv2D(n_filters, 1, kernel_initializer=params['init'], kernel_regularizer=l2(params['lmbda']))(gating)

    joined = Activation('relu')(add([theta, phi]))

    psi = Conv2D(1, 1, activation='sigmoid', kernel_initializer=params['init'], kernel_regularizer=l2(params['lmbda']))(joined)

    return multiply([skip, psi])


# buildModel
# builds the v2 U-Net, filters doubling down the encoder and halving back up the
# decoder, with a sigmoid output for the rim mask.
# parameters:
#         params: dict read for dim, input_channels, n_filters, FL, init, lmbda,
#                 dropout, v2_depth, v2_dilation, v2_attention
# outputs:
#         keras model, input (dim, dim, input_channels), output (dim, dim, 1)
def buildModel(params):

    img_input = keras.Input(shape=(params['dim'], params['dim'], params['input_channels']))

    # Encoder
    skips = []
    x = img_input

    for level in range(params['v2_depth']):
        x = convBlock(x, params['n_filters'] * 2 ** level, params)
        skips.append(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Bottleneck
    if params['v2_dilation']:
        x = dilatedBottleneck(x, params['n_filters'] * 2 ** params['v2_depth'], params)
    else:
        x = convBlock(x, params['n_filters'] * 2 ** params['v2_depth'], params)

    # Decoder
    for level in reversed(range(params['v2_depth'])):

        n_filters = params['n_filters'] * 2 ** level

        x = Conv2DTranspose(n_filters, kernel_size=2, strides=2, padding='same')(x)

        skip = skips[level]

        if params['v2_attention']:
            skip = attentionGate(skip, x, n_filters, params)

        x = Concatenate()([x, skip])
        x = Dropout(params['dropout'])(x)
        x = convBlock(x, n_filters, params)

    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(x)

    return keras.Model(img_input, output, name='U-Net-v2')
