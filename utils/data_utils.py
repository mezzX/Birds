import numpy as np
import csv
import cv2 as cv
from utils.img_utils import resize_image
import config


def parse_line(line):
    """
    Params
    ======
        line : A String containing the name of the file to be processed

    Outputs
    =======       
        img_path : The location to line

        bound : An array containing all the bounding shapes found in line.csv

        labels : An array containing all the class labels found in line.csv
    """

    if 'str' not in str(type(line)):
        line = line.decode

    img_path = config.DATA_PATH + '/images/' + line + '.jpg'
    label_path = config.DATA_PATH + '/labels/' + line + '.txt'
    bound = []
    labels = []

    with open(label_path, 'r') as f:
        data = f.readlines()
        for line in data:
            line = list(map(float, line.split(' ')))
            line = list(map(int, line))
            labels.append(line[0])
            bound.append(line[1:])

    return img_path, bound, labels


def assign_cell(pts, cell_size):
    coord = [pts[0], pts[1]]
    coord = np.asarray(coord, np.uint16)

    cell = coord // cell_size

    return cell.astype(np.uint8), coord


def resize_pts(points, ratio, pad_w, pad_h):
    pts = []
    for i, pt in enumerate(points):
        if i % 2 == 0:
                #width/x
                new_pt = round(ratio * pt) + pad_w
                pts.append(int(new_pt))
                
        else:
            #height/y
            new_pt = round(ratio * pt) + pad_h
            pts.append(int(new_pt))

    return pts


def resize_with_bounds(img, bound, class_n):
    padded_img, cell_size, resize_ratio, pad_w, pad_h = resize_image(img)

    # 1 + 1 + 4 = object detection probability + class label +
    #             (bb centre x + bb centre y + bb width + bb height) 
    y_true = np.zeros((config.GRID_SIZE,
                        config.GRID_SIZE,
                        1 + 1 + 4),
                        np.float32)

    for i, pts in enumerate(bound):
        # Set the object probability to 1 and add the class label
        label = [1, class_n[i]]
        # Scale the points to match the size from config.IMG_INPUT_SIZE
        pts = resize_pts(pts, resize_ratio, pad_w, pad_h)
        # Assign the bounding box to a cell
        cell, centre = assign_cell(pts, cell_size)
        new_cx = centre[0] / ((cell[0] + 1) * cell_size)
        new_cy = centre[1] / ((cell[1] + 1) * cell_size)
        new_w = pts[2] / config.IMG_INPUT_SIZE[0]
        new_h = pts[3] / config.IMG_INPUT_SIZE[1]
        padded_bound = [new_cx, new_cy, new_w, new_h]

        label.extend(padded_bound)
        label = np.asarray(label, np.float32)
        y_true[cell[0]][cell[1]] = label

    return padded_img, y_true


def parse_data(line):
    """
    Params
    ======
        line : A String containing the name of the file to be processed

    Outputs
    =======
        Two 3D np.arrays img_batch and label_batch
        
        img has the following shape:
            CONFIG.IMG_INPUT_SIZE * 3

        labels has the following shape:
            config.GRID_SIZE * config.GRID_SIZE * (config.NUM_CLASSES + 1 + 4)
    """

    img_path, bound, labels = parse_line(line)
    
    img, labels = resize_with_bounds(img_path, bound, labels)
  
    return img, labels


def get_batch_data(batch_line):
    """
    Params
    ======
        batch_line : An array containing items from train.txt
            Each item in the array is the file name for that exists in
            config.DATA_PATH/imgs and has a corresponding label that exist
            in config.DATA_PATH/labels

    Outputs
    =======
        Two 4D np.arrays img_batch and label_batch
        
        img_batch has the following shape:
            config.BATCH_SIZE * CONFIG.IMG_INPUT_SIZE * 3

        label_batch has the following shape:
            config.BATCH_SIZE * config.GRID_SIZE * config.GRID_SIZE * (config.NUM_CLASSES + 1 + config.MAX_NUM_PTS * 2 + 2)
    """
    
    img_batch, label_batch = [], []

    for line in batch_line:
        img, labels = parse_data(line)

        img_batch.append(img)
        label_batch.append(labels)

    img_batch = np.asarray(img_batch)
    label_batch = np.asarray(label_batch)

    return img_batch, label_batch