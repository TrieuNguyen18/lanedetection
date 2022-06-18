import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from combined import combined_thresh

def get_roi(img, vertices):
    vertices = np.array(vertices, ndmin=3, dtype=np.int32)
    if len(img.shape) == 3:
        fill_color = (255,) * 3
    else:
        fill_color = 255

    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, vertices, fill_color)
    return cv2.bitwise_and(img, mask)

def perspective_transform(img):
	"""
	Execute perspective transform
	"""
	img_size = (img.shape[1], img.shape[0])
	with open('camera_calib.p', 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']

	undist = cv2.undistort(img, mtx, dist, None, mtx)

	src = np.float32([[300, 720],[1100, 720],[595, 450],[685, 450]])
	dst = np.float32([[300, 720],[980, 720],[300, 0],[980, 0]])

	#src = np.float32([[235, 700],[1075, 700],[587, 455],[696, 455]])
	#dst = np.float32([[320, img_size[0]],[img_size[1] - 320, img_size[0]],[320, 0],[img_size[1] - 320, 0]])

	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)

	warped = cv2.warpPerspective(undist, m, img_size, flags=cv2.INTER_LINEAR)
	unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG

	vertices = np.array([[220, img_size[0]], [220, 0], [1080, 0], [1080, img_size[1]]])
	roi = get_roi(warped, vertices)

	return  warped, unwarped, m, m_inv


if __name__ == '__main__':
	#img = cv2.imread('test_images/test3.jpg')
	cap = cv2.VideoCapture("Product.mp4")
	with open('camera_calib.p', 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']
	while True:
		ret, img = cap.read()
		img = cv2.GaussianBlur(img, (5, 5), sigmaX=0)

		img = cv2.undistort(img, mtx, dist, None, mtx)
		img = combined_thresh(img)
		warped, unwarped, m, m_inv = perspective_transform(img)

		cv2.imshow("warp", warped)
		cv2.imshow("unwarp", unwarped)
		if cv2.waitKey(1) == ord('q'):
			break
cv2.destroyAllWindows()

