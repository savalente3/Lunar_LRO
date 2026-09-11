"""
dilated_U_net - attention-gated U-Net with a dilated bottleneck for small-crater
detection (the Dilated U-Net in the report; originally model_v2.py).

Targets the sub-2 km crater detection gap left by the baseline models. The
Deep U-Net (deep_U_net, Sofia Valente) and the Baseline U-Net (Silburt et al.
2019) are recall-limited on small craters: they rarely mislabel what they find
but miss a large fraction of the craters that are present. Model V2 combines
three mechanisms, each addressing that recall limitation from a different angle
and each supported by the segmentation literature.

The depth is 3. A 1 km crater is about 10 px at the 100 m/px WAC resolution, so
each pooling stage roughly halves its extent; three poolings keep small craters
resolvable at the bottleneck, whereas the four-level encoder of deep_U_net reduces
them to sub-pixel. The bottleneck uses dilated convolutions (rates 1, 2, 4) to
widen the receptive field for large-crater context without the extra pooling
that would destroy small-crater resolution (Yu & Koltun 2016, "Multi-Scale
Context Aggregation by Dilated Convolutions", ICLR, arXiv:1511.07122). Attention
gates are applied to every skip connection so the decoder emphasises crater-like
features and suppresses irrelevant terrain before concatenation; attention gates
were introduced for exactly this problem - localising small, variable, sparse
targets in a large background - and were shown to raise sensitivity to small
structures at negligible parameter cost (Oktay et al. 2018, "Attention U-Net:
Learning Where to Look for the Pancreas", arXiv:1804.03999; Schlemper et al.
2019, "Attention gated networks: Learning to leverage salient regions in medical
images", Medical Image Analysis).

The model is trained with the focal Tversky loss (defined in losses). The
pairing of attention gates with focal Tversky loss is taken directly from
Abraham & Khan (2019, "A Novel Focal Tversky Loss Function with Improved
Attention U-Net for Lesion Segmentation", IEEE ISBI, arXiv:1810.07842), who
combine the two specifically for small-lesion segmentation under class
imbalance: the attention gates help the network find the small targets, and the
focal Tversky loss penalises missing them. That is the same problem structure as
sub-2 km crater detection under the roughly 1:45 rim-to-background imbalance,
which is why this combination is used here.

The model exposes the same buildModel(params) interface as the other
architectures in this project (deep_U_net (Sofia Valente) and
baseline) and reads the same shared parameters, so the training pipeline, data
and loss are held constant and only the network differs between runs.

Usage:
    from dilated_U_net import buildModel
    model = buildModel(params)
"""

import keras
from keras.layers import (Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate,
                          Dropout, BatchNormalization, Activation, add, multiply)
from keras.regularizers import l2


# fixed architecture choices (see module docstring for the rationale)
DEPTH = 3                 # three pooling levels - keeps small craters resolvable
DILATION_RATES = (1, 2, 4)  # dilated bottleneck - receptive field without pooling


def _conv_block(x, f, FL, init, reg):
    """Two 3x3 convolutions, each followed by batch normalisation and ReLU."""
    for _ in range(2):
        x = Conv2D(f, FL, padding='same', kernel_initializer=init,
                   kernel_regularizer=reg, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x


def _dilated_bottleneck(x, f, FL, init, reg):
    """Stacked dilated convolutions (rates 1, 2, 4). Widens the receptive field
    to capture large-crater context without pooling away the resolution that
    small craters require (Yu & Koltun 2016)."""
    for rate in DILATION_RATES:
        x = Conv2D(f, FL, padding='same', dilation_rate=rate,
                   kernel_initializer=init, kernel_regularizer=reg,
                   use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x


def _attention_gate(skip, gating, inter, init, reg):
    """Additive attention gate (Oktay et al. 2018).

    Uses the coarser decoder feature map (`gating`) to weight the finer encoder
    skip feature map (`skip`), so the decoder attends to salient crater-like
    regions and suppresses irrelevant background before concatenation. The gate
    computes a per-pixel attention coefficient in [0, 1] and multiplies it into
    the skip connection.
    """
    theta = Conv2D(inter, 1, kernel_initializer=init, kernel_regularizer=reg)(skip)
    phi = Conv2D(inter, 1, kernel_initializer=init, kernel_regularizer=reg)(gating)
    act = Activation('relu')(add([theta, phi]))
    psi = Conv2D(1, 1, activation='sigmoid', kernel_initializer=init,
                 kernel_regularizer=reg)(act)
    return multiply([skip, psi])


def buildModel(params):
    dim = params['dim']
    ch = params['input_channels']
    base = params['n_filters']
    FL = params['FL']
    init = params['init']
    reg = l2(params.get('lmbda', 1e-6))
    drop = params['dropout']

    inp = keras.Input(shape=(dim, dim, ch))

    # encoder: DEPTH blocks, each two convs then a 2x2 max-pool
    skips = []
    x = inp
    for d in range(DEPTH):
        x = _conv_block(x, base * 2 ** d, FL, init, reg)
        skips.append(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # dilated bottleneck - resolution-preserving context
    x = _dilated_bottleneck(x, base * 2 ** DEPTH, FL, init, reg)

    # decoder: mirror the encoder, gate each skip with attention before merging
    for d in reversed(range(DEPTH)):
        f = base * 2 ** d
        x = Conv2DTranspose(f, 2, strides=2, padding='same')(x)
        gated_skip = _attention_gate(skips[d], x, f, init, reg)
        x = Concatenate()([x, gated_skip])
        x = Dropout(drop)(x)
        x = _conv_block(x, f, FL, init, reg)

    # output: 1x1 convolution, sigmoid, single channel (per-pixel rim probability)
    out = Conv2D(1, 1, activation='sigmoid')(x)

    return keras.Model(inp, out, name='dilated_U_net')