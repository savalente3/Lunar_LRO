# [source]: N. Khedkar (project partner) - 4_training/model_deepmoon.py
# [source]: Silburt et al. (2019) - paper 2.3 and 2.7
# [example source]: https://github.com/silburt/DeepMoon - unet_model.py

# model_deepmoon
# the DeepMoon network as published: 3 contracting blocks, a connecting path and
# 3 expansive blocks, upsampling rather than transposed convolution, and dropout
# after each merge. same buildModel(params) interface as the other model files,
# so train.py picks it up through params['model'].
# parameters:
#         none, the filter counts are fixed by the paper and the rest comes from params
# outputs:
#         a buildModel function


import keras
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from keras.regularizers import l2


# paper 2.3 filter counts, fixed by the DeepMoon design and not scaled by params.
# the paper text reads 122 for the expansive path, the released code uses 112.
F1, F2, F3 = 112, 224, 448
F_CONNECT = 448
E5, E6, E7 = 224, 112, 112


# convBlock
# the two padded 3x3 convolutions every DeepMoon block is built from.
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
            activation='relu',
            kernel_initializer=params['init'],
            kernel_regularizer=l2(params['lmbda']),
            padding='same'
        )(x)

    return x


# buildModel
# builds the DeepMoon network, with a sigmoid output for the rim mask.
# parameters:
#         params: dict read for dim, input_channels, FL, init, lmbda, dropout
# outputs:
#         keras model, input (dim, dim, input_channels), output (dim, dim, 1)
def buildModel(params):

    img_input = keras.Input(shape=(params['dim'], params['dim'], params['input_channels']))

    # Encoder1
    c1 = convBlock(img_input, F1, params)
    c1P = MaxPooling2D((2, 2), strides=(2, 2))(c1)

    # Encoder2
    c2 = convBlock(c1P, F2, params)
    c2P = MaxPooling2D((2, 2), strides=(2, 2))(c2)

    # Encoder3
    c3 = convBlock(c2P, F3, params)
    c3P = MaxPooling2D((2, 2), strides=(2, 2))(c3)

    # Connecting path
    u = convBlock(c3P, F_CONNECT, params)

    # Decoder1
    d1U = UpSampling2D((2, 2))(u)
    d1c = Concatenate()([d1U, c3])
    x1 = Dropout(params['dropout'])(d1c)
    d1 = convBlock(x1, E5, params)

    # Decoder2
    d2U = UpSampling2D((2, 2))(d1)
    d2c = Concatenate()([d2U, c2])
    x2 = Dropout(params['dropout'])(d2c)
    d2 = convBlock(x2, E6, params)

    # Decoder3
    d3U = UpSampling2D((2, 2))(d2)
    d3c = Concatenate()([d3U, c1])
    x3 = Dropout(params['dropout'])(d3c)
    d3 = convBlock(x3, E7, params)

    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(d3)

    return keras.Model(img_input, output, name='DeepMoon-paper')
