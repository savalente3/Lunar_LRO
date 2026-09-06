"""
Model V2 - small-crater-optimised U-Net.

Targets the sub-2 km crater detection gap. Standard segmentation U-Nets
(model_v1 (Sofia Valente), and the DeepMoon baseline of Silburt et al. 2019)
apply three to four pooling stages, so a ~10 px feature (a 1 km crater at
100 m/px WAC resolution) is reduced to roughly one pixel or less at the
bottleneck and its detail is lost. Model V2 preserves small-crater resolution
through two mechanisms, both exposed as parameters so the architecture can be
selected empirically rather than fixed by assumption:

  params['v2_depth']     : number of pooling levels (3 or 4). Fewer poolings
                           keep small craters resolvable; more poolings widen
                           the receptive field at the cost of fine detail.
                           Default 3.
  params['v2_dilation']  : if True, the bottleneck uses stacked dilated
                           convolutions (rates 1, 2, 4) to enlarge the
                           receptive field WITHOUT further pooling, so global
                           context is gained while spatial resolution is kept.
                           Default True.
                           [ref] Yu & Koltun (2016), "Multi-Scale Context
                           Aggregation by Dilated Convolutions", ICLR;
                           arXiv:1511.07122.
  params['v2_attention'] : if True, additive attention gates are applied to the
                           skip connections so the decoder emphasises
                           crater-like features and suppresses background.
                           Default False.
                           [ref] Oktay et al. (2018), "Attention U-Net:
                           Learning Where to Look for the Pancreas";
                           arXiv:1804.03999.

The model exposes the same buildModel(params) interface as the other
architectures in this project (model_v1 (Sofia Valente) and the DeepMoon
baseline) and reads the same shared parameters - 'dim', 'input_channels',
'n_filters', 'FL', 'init', 'lmbda', 'dropout' - so that the training pipeline,
loss and data are held constant and only the network differs between runs.

Usage:
    from model_v2 import buildModel
    model = buildModel(params)
"""

import keras
from keras.layers import (Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate,
                          Dropout, BatchNormalization, Activation, multiply)
from keras.regularizers import l2


def _conv_block(x, f, FL, init, reg):
    """Two 3x3 convolutions, each followed by batch normalisation and ReLU.
    Batch normalisation stabilises training at the higher effective resolution
    that Model V2 preserves."""
    for _ in range(2):
        x = Conv2D(f, FL, padding='same', kernel_initializer=init,
                   kernel_regularizer=reg, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x


def _dilated_bottleneck(x, f, FL, init, reg):
    """Stacked dilated convolutions (rates 1, 2, 4). Enlarges the receptive
    field to capture large-crater context without pooling away the spatial
    resolution that small craters require."""
    for rate in (1, 2, 4):
        x = Conv2D(f, FL, padding='same', dilation_rate=rate,
                   kernel_initializer=init, kernel_regularizer=reg,
                   use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x


def _attention_gate(skip, gating, inter, init, reg):
    """Additive attention gate (Oktay et al. 2018). Learns to weight the skip
    connection so the decoder focuses on salient (crater-like) regions and
    suppresses irrelevant background before concatenation."""
    theta = Conv2D(inter, 1, kernel_initializer=init, kernel_regularizer=reg)(skip)
    phi = Conv2D(inter, 1, kernel_initializer=init, kernel_regularizer=reg)(gating)
    add = Activation('relu')(keras.layers.add([theta, phi]))
    psi = Conv2D(1, 1, activation='sigmoid', kernel_initializer=init,
                 kernel_regularizer=reg)(add)
    return multiply([skip, psi])


def buildModel(params):
    dim = params['dim']
    ch = params['input_channels']
    base = params['n_filters']
    FL = params['FL']
    init = params['init']
    reg = l2(params.get('lmbda', 1e-6))
    drop = params['dropout']

    depth = params.get('v2_depth', 3)
    use_dilation = params.get('v2_dilation', True)
    use_attention = params.get('v2_attention', False)

    inp = keras.Input(shape=(dim, dim, ch))

    # encoder: `depth` blocks, each two convs then a 2x2 max-pool
    skips = []
    x = inp
    for d in range(depth):
        x = _conv_block(x, base * 2 ** d, FL, init, reg)
        skips.append(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # bottleneck: dilated (resolution-preserving) or a plain conv block
    if use_dilation:
        x = _dilated_bottleneck(x, base * 2 ** depth, FL, init, reg)
    else:
        x = _conv_block(x, base * 2 ** depth, FL, init, reg)

    # decoder: mirror the encoder, optionally gate each skip with attention
    for d in reversed(range(depth)):
        f = base * 2 ** d
        x = Conv2DTranspose(f, 2, strides=2, padding='same')(x)
        skip = skips[d]
        if use_attention:
            skip = _attention_gate(skip, x, f, init, reg)
        x = Concatenate()([x, skip])
        x = Dropout(drop)(x)
        x = _conv_block(x, f, FL, init, reg)

    # output: 1x1 convolution, sigmoid, single channel (per-pixel rim probability)
    out = Conv2D(1, 1, activation='sigmoid')(x)

    tag = f"d{depth}{'_dil' if use_dilation else ''}{'_att' if use_attention else ''}"
    return keras.Model(inp, out, name=f'U-Net-v2_{tag}')
