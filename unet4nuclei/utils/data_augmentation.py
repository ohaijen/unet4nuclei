import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp

def deform(image1, image2, points=10, distort=5.0):
    
    # create deformation grid 
    rows, cols = image1.shape[0], image1.shape[1]
    src_cols = np.linspace(0, cols, points)
    src_rows = np.linspace(0, rows, points)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add distortion to coordinates
    s = src[:, 1].shape
    dst_rows = src[:, 1] + np.random.normal(size=s)*np.random.uniform(0.0, distort, size=s)
    dst_cols = src[:, 0] + np.random.normal(size=s)*np.random.uniform(0.0, distort, size=s)
    
    dst = np.vstack([dst_cols, dst_rows]).T

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = rows 
    out_cols = cols
    out1 = warp(image1, tform, output_shape=(out_rows, out_cols), mode="symmetric")
    out2 = warp(image2, tform, output_shape=(out_rows, out_cols), mode="symmetric")
    
    return out1, out2