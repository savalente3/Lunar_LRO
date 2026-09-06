"""
Recall-oriented loss functions for small-crater segmentation.

MOTIVATION
----------
Evaluation of the baseline model_v1 (Sofia Valente) shows that detection
performance is limited by recall rather than precision: per-band crater
precision is high (0.79 at 1-2 km, 0.94 at 2-5 km, 0.96 at 5-10 km) while
recall is substantially lower (0.62, 0.59, 0.46 respectively). The model
rarely mislabels the craters it reports, but misses a large fraction of the
craters that are present. Improving on this baseline therefore requires raising
recall without collapsing precision.

Binary focal cross-entropy (the baseline loss) reweights examples by
classification confidence but is symmetric with respect to false positives and
false negatives, so it does not provide a direct control for trading precision
against recall. The Tversky loss introduces separate penalties for false
positives and false negatives, giving an explicit recall control, which makes
it the principled choice for a recall-limited segmentation task.

LOSSES
------
tversky_loss(alpha, beta):
    Generalises the Dice coefficient with independent weights for false
    positives (alpha) and false negatives (beta), where alpha + beta = 1.
    Setting beta > alpha penalises missed rim pixels more heavily than false
    alarms, biasing the model toward higher recall. alpha = beta = 0.5 recovers
    the Dice loss.
    [ref] Salehi, Erdogmus & Gholipour (2017), "Tversky loss function for image
    segmentation using 3D fully convolutional deep networks", MICCAI Workshop
    on Machine Learning in Medical Imaging; arXiv:1706.05721.
    [ref] Tversky (1977), "Features of similarity", Psychological Review 84(4),
    the origin of the Tversky index.

focal_tversky_loss(alpha, beta, gamma):
    Applies a focal exponent to the Tversky loss so that hard, low-overlap
    regions - characteristically the small and faint craters that a
    cross-entropy model tends to miss - dominate the gradient. gamma > 1
    increases this focus; gamma = 1 reduces to the plain Tversky loss.
    [ref] Abraham & Khan (2019), "A Novel Focal Tversky Loss Function with
    Improved Attention U-Net for Lesion Segmentation", IEEE International
    Symposium on Biomedical Imaging (ISBI); arXiv:1810.07842.

The losses are implemented with keras.ops so they are backend-agnostic and
consistent with the rest of the training pipeline. build_loss(params) selects
the loss from params so that training remains declarative and the exact loss
and its parameters are recorded for each run.

Usage:
    from losses_v2 import build_loss
    model.compile(optimizer=..., loss=build_loss(params))
"""

import keras
from keras import ops


def tversky_loss(alpha=0.3, beta=0.7, smooth=1.0):
    """Tversky loss. beta > alpha favours recall by penalising false negatives
    (missed rim pixels) more heavily than false positives."""
    def loss(y_true, y_pred):
        yt = ops.reshape(y_true, (-1,))
        yp = ops.reshape(y_pred, (-1,))

        tp = ops.sum(yt * yp)
        fp = ops.sum((1 - yt) * yp)
        fn = ops.sum(yt * (1 - yp))

        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1 - tversky
    return loss


def focal_tversky_loss(alpha=0.3, beta=0.7, gamma=1.333, smooth=1.0):
    """Focal Tversky loss. gamma > 1 concentrates the gradient on hard,
    low-overlap craters (small or faint), where recall is weakest."""
    def loss(y_true, y_pred):
        yt = ops.reshape(y_true, (-1,))
        yp = ops.reshape(y_pred, (-1,))

        tp = ops.sum(yt * yp)
        fp = ops.sum((1 - yt) * yp)
        fn = ops.sum(yt * (1 - yp))

        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return ops.power(1 - tversky, gamma)
    return loss


def build_loss(params):
    """Select the loss from params.

    params['loss'] in:
      'binary_focal_crossentropy'  -> baseline focal cross-entropy
      'tversky'                    -> recall-oriented; uses tversky_alpha/beta
      'focal_tversky'              -> recall with hard-example focus; adds
                                      tversky_gamma
    """
    name = params.get('loss', 'binary_focal_crossentropy')

    if name == 'tversky':
        return tversky_loss(
            alpha=params.get('tversky_alpha', 0.3),
            beta=params.get('tversky_beta', 0.7),
        )

    if name == 'focal_tversky':
        return focal_tversky_loss(
            alpha=params.get('tversky_alpha', 0.3),
            beta=params.get('tversky_beta', 0.7),
            gamma=params.get('tversky_gamma', 1.333),
        )

    # default: focal cross-entropy, so the baseline configuration is reproducible
    return keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=params.get('focal_class_balancing', True),
        alpha=params.get('focal_alpha', 0.75),
        gamma=params.get('focal_gamma', 2.0),
    )
