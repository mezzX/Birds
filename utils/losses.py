import tensorflow as tf
from tensorflow import math
import config


def bce(pred, label):
    # Binary Cross Entropy
    if label == 0:
        pred = pred - config.EPSILON
        loss = math.log(1 - pred)

    else:
        pred = pred + config.EPSILON
        loss = math.log(pred)

    return -loss


def mse(pred, label):
    # Mean Squared Error
    loss = math.squared_difference(label, pred)

    return math.reduce_sum(loss) / len(pred)


def scce(cell, label):
    # Sparse Categorical Cross Entropy
    pred = cell[int(label) + 1]
    pred += config.EPSILON
    loss = math.log(pred)

    return -loss


def calc_losses(logits, targets):
    no_obj_loss = 0
    no_obj_num = 0
    obj_loss = 0
    obj_num = 0

    for i, img in enumerate(logits):
        for r, row in enumerate(img):
            for c, cell in enumerate(row):
                # Object Detection Loss
                obj_fnd_lbl = targets[i][r][c][0]
                obj_fnd_pred = cell[0]
                loss = bce(obj_fnd_pred, obj_fnd_lbl)

                # WHEN NO OBJ IS PRESENT
                if obj_fnd_lbl == 0:
                    no_obj_num += 1
                    no_obj_loss += loss

                # WHEN OBJ IS PRESENT
                else:
                    obj_num += 1
                    obj_loss += loss
                    # Class Loss
                    class_lbl = targets[i][r][c][1]
                    obj_loss += scce(cell, class_lbl)
                    # Bounding Box Loss
                    pts_lbl = targets[i][r][c][-4:]
                    pts_pred = cell[-4:]
                    obj_loss += mse(pts_pred, pts_lbl)

    avg_no_obj_loss = no_obj_loss / no_obj_num
    avg_obj_loss = obj_loss / obj_num
    total_loss = config.NEG_SAMPLE_SCALE * avg_no_obj_loss + avg_obj_loss

    return total_loss