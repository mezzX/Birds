import cv2 as cv
import config
import numpy as np
from PIL import Image
from PIL import ImageDraw


CELL_SIZE = config.IMG_INPUT_SIZE[0] // config.GRID_SIZE

def conv_origin(pts, cell):
    cells = [(cell[0] + 1) * CELL_SIZE, (cell[1] + 1) * CELL_SIZE]
    pts = [round(pts[0] * cells[0]), round(pts[1] * cells[1])]
    pts = [int(pts[0]), int(pts[1])]

    
    return pts


def get_poly_points(origin, pts):
    new_pts = []
    for i, pt in enumerate(pts):
        if i % 2 == 0:
            #width
            new_pt = round(pt * config.IMG_INPUT_SIZE[0])
            new_pts.append(int(origin[0] + new_pt))
        
        else:
            #height
            new_pt = round(pt * config.IMG_INPUT_SIZE[1])
            new_pts.append(int(origin[1] + new_pt))
            
    return new_pts


def convert_outs(pts, cell):
    origin = pts[:2]
    pts = pts[2:]
    origin = conv_origin(origin, cell)
    pts = get_poly_points(origin, pts)
    
    return pts


def resize_image(img_path):
    img = cv.imread(img_path)
    height, width = img.shape[:2]

    resize_ratio = min(config.IMG_INPUT_SIZE[0] / width, config.IMG_INPUT_SIZE[1] / height)

    new_height = int(round(height * resize_ratio))
    new_width = int(round(width * resize_ratio))

    img = cv.resize(img, (new_width, new_height))
    padded_img = np.full((config.IMG_INPUT_SIZE[0],config.IMG_INPUT_SIZE[1],3),0,np.uint8)

    pad_w = int((config.IMG_INPUT_SIZE[0] - new_width) / 2)
    pad_h = int((config.IMG_INPUT_SIZE[1] - new_height) / 2)

    padded_img[pad_h:new_height+pad_h, pad_w:new_width+pad_w] = img

    padded_img = cv.cvtColor(padded_img, cv.COLOR_BGR2RGB).astype(np.float32)
    padded_img = padded_img / 255

    return padded_img, CELL_SIZE, resize_ratio, pad_w, pad_h


def scale_points(pts, size):
    ratio = min(config.IMG_INPUT_SIZE[0] / size[0], config.IMG_INPUT_SIZE[1] / size[1])
    n_width = int(round(ratio * size[0]))
    n_height = int(round(ratio * size[1]))
    
    pad_w = int((config.IMG_INPUT_SIZE[0] - n_width) / 2)
    pad_h = int((config.IMG_INPUT_SIZE[1] - n_height) / 2)
    
    new_pts = []
    for i, pt in enumerate(pts):
        if i % 2 == 0:
            #width
            pt = round((pt - pad_w) / ratio)
            new_pts.append(int(pt))
        
        else:
            #height
            pt = round((pt - pad_h) / ratio)
            new_pts.append(int(pt))
        
    return new_pts


def draw_shapes(shapes, img_path):
    img = Image.open(img_path).convert('RGBA')
    width, height = img.size
    img2 = ImageDraw.Draw(img)

    for points in shapes:
        points = scale_points(points, [width, height])
        img2.polygon(points, outline='red')

    img.save('outputs/out_' + img_path[-9:])