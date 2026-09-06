"""
DeepMoon baseline — faithful reimplementation of Silburt et al. (2019),
"Lunar Crater Identification via Deep Learning", Icarus 317, arXiv:1803.02192.

Every architectural choice below is cited to the paper (§2.3) or the reference
implementation at github.com/silburt/DeepMoon. Built to the same
buildModel(params) interface as model_v1.py and model_baseline.py so it slots
into train.py unchanged (params['model'] == 'DeepMoon-paper').

ARCHITECTURE (paper §2.3, verbatim):
  "The contracting and expansive paths each contain 3 convolutional blocks. A
   block in the contracting path consists of two convolutional layers followed
   by a max-pooling layer with a 2x2 pool size. A block in the expansive path
   consists of a 2x2 upsampling layer, a concatenation with the corresponding
   block from the contracting path (i.e. a merge layer), a dropout layer, and
   two convolutional layers. The connecting path consists of two convolutional
   layers. Lastly, the final output layer is a 1x1 convolutional layer with a
   sigmoid activation and a single filter."

  Filter counts (paper §2.3):
    contracting blocks 1, 2, 3 -> 112, 224, 448
    connecting path            -> 448
    expansive blocks 5, 6, 7   -> 224, 112, 112

  NOTE ON 112 vs 122: the paper text reads "224, 122, 122" for the expansive
  path. 122 is almost certainly a typographical error for 112 — the DeepMoon
  reference repository (silburt/DeepMoon, unet_model.py) uses 112, the value
  appears nowhere else, and 122 is not a filter count anyone selects
  deliberately. We use 112, matching the released code. This is the only point
  where paper text and reference code disagree.

  Other choices (paper §2.3 and §2.7):
    - all convs 3x3, padded ('same'), ReLU                       (§2.3)
    - upsampling via 2x2 UpSampling2D, NOT transposed conv       (§2.3: "2x2
      upsampling layer"; the repo uses UpSampling2D)
    - dropout AFTER each merge, BEFORE the two convs             (§2.3)
    - L2 weight regularisation 1e-5                              (§2.7: "weight
      regularization = 1e-5")
    - dropout 15%                                                (§2.7)
    - he_normal init is a modern default; the paper does not
      specify an initialiser, so this is left to params

  Trained (paper §2.7) with: Adam, lr 1e-4, batch 8, binary cross-entropy,
  depth 3. Those live in train.py's params, not here.
"""

import keras
from keras.layers import (Conv2D, MaxPooling2D, UpSampling2D, Concatenate,
                          Dropout)
from keras.regularizers import l2


# paper filter counts, fixed by the DeepMoon design (not scaled by params)
F1, F2, F3 = 112, 224, 448          # contracting blocks 1,2,3 (§2.3)
F_CONNECT = 448                      # connecting path (§2.3)
E5, E6, E7 = 224, 112, 112          # expansive blocks 5,6,7 (§2.3; 112 not 122)


def buildModel(params):
    FL = params['FL']                # 3x3 (paper §2.7)
    init = params['init']
    reg = l2(params.get('lmbda', 1e-5))   # paper §2.7: 1e-5
    drop = params['dropout']         # paper §2.7: 0.15

    def conv(x, f):
        # two 3x3 padded convs + ReLU (paper §2.3: "two convolutional layers")
        for _ in range(2):
            x = Conv2D(f, FL, activation='relu', padding='same',
                       kernel_initializer=init, kernel_regularizer=reg)(x)
        return x

    inp = keras.Input(shape=(params['dim'], params['dim'],
                             params['input_channels']))

    # ---- contracting path: 3 blocks, each = 2 convs then 2x2 maxpool ----
    c1 = conv(inp, F1)
    p1 = MaxPooling2D((2, 2), strides=(2, 2))(c1)

    c2 = conv(p1, F2)
    p2 = MaxPooling2D((2, 2), strides=(2, 2))(c2)

    c3 = conv(p2, F3)
    p3 = MaxPooling2D((2, 2), strides=(2, 2))(c3)

    # ---- connecting path: 2 convs, 448 filters ----
    u = conv(p3, F_CONNECT)

    # ---- expansive path: 3 blocks, each = upsample, merge, dropout, 2 convs ----
    # block 5 (merge with c3)
    u = UpSampling2D((2, 2))(u)
    u = Concatenate()([u, c3])
    u = Dropout(drop)(u)
    u = conv(u, E5)

    # block 6 (merge with c2)
    u = UpSampling2D((2, 2))(u)
    u = Concatenate()([u, c2])
    u = Dropout(drop)(u)
    u = conv(u, E6)

    # block 7 (merge with c1)
    u = UpSampling2D((2, 2))(u)
    u = Concatenate()([u, c1])
    u = Dropout(drop)(u)
    u = conv(u, E7)

    # ---- output: 1x1 conv, sigmoid, single filter (paper §2.3) ----
    out = Conv2D(1, 1, activation='sigmoid')(u)

    return keras.Model(inp, out, name='DeepMoon-paper')
