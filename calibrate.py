import numpy as np
import cv2
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def calibrate_camera():

    imgpaths = glob.glob('camera_cal/*.jpg')
    image = cv2.imread(imgpaths[0])
    imshape = image.shape[:2] # gets only the (height, width) to be used in the cv2.calibrateCamera()
    objpoints = [] #toạ độ không gian của điểm ảnh
    imgpoints = []  #toạ độ 2D của điểm ảnh

    nx = 9  # Number of inside corners on each row of the chessboard
    ny = 6  # Number of inside corners on each column of the chessboard

    objp = np.zeros([ny * nx, 3], dtype=np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Iterate over each calibration image and determine the objpoints and imgpoints
    for idx, imgpath in enumerate(imgpaths):
        print('idx ',str(idx))
        img = cv2.imread(imgpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            imgpoints.append(corners)
            objpoints.append(objp)
            cv2.imshow("img", img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imshape[::-1], None, None)
    return mtx, dist

if os.path.exists('camera_calib.p'):
    with open('camera_calib.p', mode='rb') as f:
        data = pickle.load(f)
        mtx, dist = data['mtx'], data['dist']
else:
    mtx, dist = calibrate_camera()
    with open('camera_calib.p', mode='wb') as f:
        pickle.dump({'mtx': mtx, 'dist': dist}, f)

if __name__ == "__main__":
    calibrate_camera()

