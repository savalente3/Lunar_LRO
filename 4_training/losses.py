# [source]: N. Khedkar (project partner) - 4_training/losses_v2.py, renamed losses.py
# [source]: Salehi, Erdogmus and Gholipour (2017) - Tversky loss, arXiv:1706.05721
# [source]: Abraham and Khan (2019) - focal Tversky loss, arXiv:1810.07842

# losses
# recall oriented losses. focal cross entropy weights examples by confidence but
# treats a false positive and a false negative the same, so it gives no handle on
# the recall limit. Tversky penalises the two separately, which does.
# parameters:
#         loss: 'binary_focal_crossentropy' | 'tversky' | 'focal_tversky'
#         tversky_alpha: weight on false positives
#         tversky_beta: weight on false negatives, beta > alpha favours recall
#         tversky_gamma: focal exponent, 1 reduces focal_tversky to tversky
# outputs:
#         a buildLoss function returning a keras loss


import keras
from keras import ops


# tverskyLoss
# Dice generalised with separate weights for false positives and false negatives.
# alpha = beta = 0.5 is Dice.
# parameters:
#         alpha: weight on false positives, default 0.3
#         beta: weight on false negatives, default 0.7
#         smooth: added to both sides so an empty patch does not divide by zero
# outputs:
#         loss function taking y_true, y_pred
def tverskyLoss(alpha=0.3, beta=0.7, smooth=1.0):

    def loss(y_true, y_pred):

        y_true = ops.reshape(y_true, (-1,))
        y_pred = ops.reshape(y_pred, (-1,))

        tp = ops.sum(y_true * y_pred)
        fp = ops.sum((1 - y_true) * y_pred)
        fn = ops.sum(y_true * (1 - y_pred))

        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

        return 1 - tversky

    return loss


# focalTverskyLoss
# Tversky raised to gamma, so the hard low overlap craters - the small and faint
# ones - take the larger share of the gradient.
# parameters:
#         alpha: weight on false positives, default 0.3
#         beta: weight on false negatives, default 0.7
#         gamma: focal exponent, default 1.333
#         smooth: added to both sides so an empty patch does not divide by zero
# outputs:
#         loss function taking y_true, y_pred
def focalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.333, smooth=1.0):

    def loss(y_true, y_pred):

        y_true = ops.reshape(y_true, (-1,))
        y_pred = ops.reshape(y_pred, (-1,))

        tp = ops.sum(y_true * y_pred)
        fp = ops.sum((1 - y_true) * y_pred)
        fn = ops.sum(y_true * (1 - y_pred))

        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

        return ops.power(1 - tversky, gamma)

    return loss


# buildLoss
# picks the loss named in params, so the run is declarative and mlflow records
# which loss and which weights were used.
# parameters:
#         params: dict read for loss and the loss's own parameters
# outputs:
#         keras loss
def buildLoss(params):

    match params['loss']:

        case 'tversky':
            return tverskyLoss(alpha=params['tversky_alpha'], beta=params['tversky_beta'])

        case 'focal_tversky':
            return focalTverskyLoss(alpha=params['tversky_alpha'], beta=params['tversky_beta'], gamma=params['tversky_gamma'])

        case 'binary_focal_crossentropy':
            return keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=params['focal_class_balancing'],
                alpha=params['focal_alpha'],
                gamma=params['focal_gamma'],
            )

        case _:
            return params['loss']
