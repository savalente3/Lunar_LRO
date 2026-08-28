"""
Baseline - DeepMoon's architecture (Silburt et al. 2019).

"""

import keras
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout
from keras.regularizers import l2


def buildModel(params):

    img_input = keras.Input(shape=(params['dim'], params['dim'], params['input_channels']))

    # Encoder1
    a1 = Conv2D(
        params['n_filters'],
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(img_input)

    a1 = Conv2D(
        params['n_filters'],
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(a1)

    a1P = MaxPooling2D((2, 2), strides=(2, 2))(a1)

    # Encoder2
    a2 = Conv2D(
        params['n_filters'] * 2,
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(a1P)

    a2 = Conv2D(
        params['n_filters'] * 2,
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(a2)

    a2P = MaxPooling2D((2, 2), strides=(2, 2))(a2)

    # Encoder3
    a3 = Conv2D(
        params['n_filters'] * 4,
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(a2P)

    a3 = Conv2D(
        params['n_filters'] * 4,
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(a3)

    a3P = MaxPooling2D((2, 2), strides=(2, 2))(a3)

    # Connecting - stays at 448, no doubling. DeepMoon 2.3
    u = Conv2D(
        params['n_filters'] * 4,
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(a3P)

    u = Conv2D(
        params['n_filters'] * 4,
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(u)

    # Decoder1 - 224
    d1CT = Conv2DTranspose(params['n_filters']*2, kernel_size=2, strides=2, padding='same')(u)
    d1c = Concatenate()([d1CT, a3])
    x1 = Dropout(params['dropout'])(d1c)

    d1 = Conv2D(
        params['n_filters'] * 2,
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(x1)

    d1 = Conv2D(
        params['n_filters'] * 2,
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(d1)

    # Decoder2 - 112
    d2CT = Conv2DTranspose(params['n_filters'], kernel_size=2, strides=2, padding='same')(d1)
    d2c = Concatenate()([d2CT, a2])
    x2 = Dropout(params['dropout'])(d2c)

    d2 = Conv2D(
        params['n_filters'],
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(x2)

    d2 = Conv2D(
        params['n_filters'],
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(d2)

    # Decoder3 - 112, no reduction. DeepMoon 2.3
    d3CT = Conv2DTranspose(params['n_filters'], kernel_size=2, strides=2, padding='same')(d2)
    d3c = Concatenate()([d3CT, a1])
    x3 = Dropout(params['dropout'])(d3c)

    d3 = Conv2D(
        params['n_filters'],
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(x3)

    d3 = Conv2D(
        params['n_filters'],
        params['FL'],
        activation='relu',
        kernel_initializer=params['init'],
        kernel_regularizer=l2(params['lmbda']),
        padding='same'
    )(d3)

    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(d3)

    return keras.Model(img_input, output, name='DeepMoon-baseline')
