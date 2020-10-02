import tensorflow as tf
import numpy as np
from utils.img_utils import resize_image, convert_outs, draw_shapes
from model import MezzNet
import config


def infer(img_path, ver='best'):
    model = MezzNet()
    model.load_weights('./weights/' + ver)
    img, _, _, _, _ = resize_image(img_path)
    img = np.asarray([img])
    outs = model(img)
    outs = np.squeeze(outs)
    shapes = nms(outs)

    draw_shapes(shapes, img_path)

    print('Output Image Saved in the outputs Directory')


def nms(outs):
    shapes = []

    for r, row in enumerate(outs):
        for c, cell in enumerate(row):
            prob = cell[0]
            if prob > config.NMS_THRESHOLD:
                pts = convert_outs(cell[1:], [r, c])
                shapes.append(pts)

    return shapes