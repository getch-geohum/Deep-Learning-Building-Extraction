import os
import random
import numpy as np
from shapely.geometry import Polygon
from skimage import measure
from skimage.io import imread, imsave
from glob import glob
from tqdm import tqdm
from pathlib import Path
from skimage.transform import resize
from scipy.spatial import ConvexHull
import argparse


def getXYwh(points=None):

    x = points[:,0].tolist()
    y = points[:,1].tolist()

    x1, x2 = min(x), max(x)
    y1, y2 = min(y), max(y)
    c_x = x1+ (x2-x1)/2
    c_y = y1 + (y2-y1)/2
    w = x2-x1
    h = y2-y1

    return np.array([c_x, c_y, w, h])


def get_obb_from_points(points, calcconvexhull=True):
    
    if calcconvexhull:
        _ch = ConvexHull(points)
        points = _ch.points[_ch.vertices]

    cov_points = np.cov(points, y=None, rowvar=0, bias=1)
    v, vect = np.linalg.eig(cov_points)
    tvect = np.transpose(vect)

    points_rotated = np.dot(points, np.linalg.inv(tvect))
    mina = np.min(points_rotated, axis=0)
    maxa = np.max(points_rotated, axis=0)
    diff = (maxa - mina) * 0.5
    center = mina + diff

    corners = np.array([center + [-diff[0], -diff[1]], center + [diff[0], -diff[1]], center + [diff[0], diff[1]],
                        center + [-diff[0], diff[1]], center + [-diff[0], -diff[1]]])

    corners = np.dot(corners, tvect)
    

    return corners

def mask2cord(MASK, output=None):   # 'mask', 'bbox', 'obbox'
    assert len(MASK.shape) == 2, 'shape of the mask should be two dimensional. Please check it'

    contours = measure.find_contours(MASK, 0.5)
    H, W = MASK.shape[0], MASK.shape[1]

    cords_ = []

    if len(contours) >= 1:
        for cont in contours:
            for i in range(len(cont)):
                row, col = cont[i]
                cont[i] = (col, row)
            if len(cont) < 4:  # invalid geometries
                continue
            poly = Polygon(cont)
            poly = poly.simplify(1.0, preserve_topology=False)
            if poly.is_empty:
                continue
            if poly.geom_type == 'MultiPolygon':
                for spoly in list(poly.geoms): # # poly: This is a fix for Shapely 2...
                    if spoly.is_empty:
                        continue
                    if not spoly.is_valid:
                        continue

                    bound_cords = np.array(spoly.exterior.coords) # return a numpy array of shape [n, 2]

                    if output=='segment':
                        cords = np.clip(bound_cords/H, 0, 1).ravel().tolist()
                    elif output == "bbox":
                        cords = getXYwh(points=bound_cords)
                        cords = np.clip(cords/H, 0, 1).ravel().tolist()
                    elif output == "obbox":
                        cords = get_obb_from_points(bound_cords/H, calcconvexhull=True)
                        cords = cords.ravel().ravel().tolist()[:8]
                    else:
                        raise ValueError("output bbox type is not defined")

                    class_id = [0]
                    cord_txt = " ".join([str(a) for a in class_id + cords])

                    cords_.append(cord_txt)

            else:
                if not poly.is_valid:
                    continue

                bound_cords = np.array(poly.exterior.coords)  # return a numpy array of shape [n, 2]

                if output=='segment':
                    cords = np.clip(bound_cords/H, 0, 1).ravel().tolist()
                elif output == "bbox":
                    cords = getXYwh(points=bound_cords)
                    cords = np.clip(cords/H, 0, 1).ravel().tolist()
                elif output == "obbox":
                    cords = get_obb_from_points(bound_cords/H, calcconvexhull=True)
                    cords = cords.ravel().ravel().tolist()[:8]
                else:
                    raise ValueError("output bbox type is not defined")
                    
                class_id = [0]
                cord_txt = " ".join([str(a) for a in class_id + cords])
                cords_.append(cord_txt)

    return cords_


def file2text(file=None, out_dir=None, idx = None, upsample=True, output=None):

    mask = imread(file)
    
    if upsample:
        mask = resize(mask, (640, 640))
        mask[mask > 0] = 1
        mask = mask.astype(np.uint8)

        if idx == 0:
            print("Mask main {} and max {} ".format(mask.min(), mask.max()))

    assert len(mask.shape) == 2, "Mask is not a 2D image"
    str_cords = mask2cord(MASK=mask, output=output)

    name = os.path.splitext(os.path.split(file)[1])[0] + ".txt"
  
    if idx is not None:
        outpath = f"{out_dir}/{idx}_{name}"
    else:
        outpath = f"{out_dir}/{name}"

    try:
        with open(outpath, 'w+') as opf:
            for line in str_cords:
                opf.write(f"{line}\n")
            # opf.write("\n".join(str_cords))
    except:
        with open(outpath, 'w+') as opf:
            print('saving did not work')
            pass

def to_8bits(arr, vmin=0, vmax=255):
    arr_min, arr_max = arr.min(), arr.max()
    val = ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin
    return np.clip(val, 0, 255).astype(np.uint8)

def write_images(images1, labels=None, out_root=None, upsample=True, tobits=True,output=None):

    assert len(images1) == len(labels), "Images and labels are not of equal length"

    if labels is None:
        labels = [None] * len(images1)

    for idx, (im_path1, lb_path) in tqdm(enumerate(list(zip(images1, labels)))):
        # try:
        im1 = imread(im_path1)[:,:,:3]
        
        

        if upsample:
            im1 = resize(im1, (640, 640))
        if tobits:
            im1 = to_8bits(im1)
            
        im_name = f"{out_root}/images/{idx}_{os.path.splitext(os.path.split(im_path1)[1])[0]}.tif"
            
        imsave(im_name, im1)
        
        file2text(file=lb_path, out_dir=f"{out_root}/labels", idx=idx, output=output)
        # except:
        #     print("{}, {} failed".format(im_path1, lb_path))

def config_writer(save_dir):
    data_config = f"""
    path: {save_dir}/dataset
    train: train/images
    val: valid/images
    test: test/images
    names:
        0: dwellings
    nc: 1
    """

    with Path(f"{save_dir}/data.yaml").open("w") as f:
        f.write(data_config)


def prepare_yolo_data(data_dir, out_dir, ext='tif', output="segment", part="train", upsample=False, tobits=False, write_config=False):
    os.makedirs(f"{out_dir}/dataset/{part}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/dataset/{part}/labels", exist_ok=True)

    images = sorted(glob(f"{data_dir}/images/*.{ext}"))
    labels = sorted(glob(f"{data_dir}/labels/*.{ext}"))

    print(f"Started the training sample saving, with a total of {len(images)}")
    write_images(images1=images, labels=labels, out_root=f"{out_dir}/dataset/{part}", output=output, upsample=upsample, tobits=tobits)

    if write_config:
      config_writer(save_dir=out_dir)
