import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
	"""
	Takes an image, gradient orientation, and threshold min/max values
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Apply x or y gradient with the OpenCV Sobel() function
	# and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	# Return the result
	return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):

	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255
	gradmag = (gradmag/scale_factor).astype(np.uint8)
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	# Return the binary image
	return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Calculate the x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Take the absolute value of the gradient direction,
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Return the binary image
	return binary_output

def binary_threshold(img, low, high): #chuyển ảnh nhị phân
    if len(img.shape) == 2:
        output = np.zeros_like(img)
        mask = (img >= low) & (img <= high)

    elif len(img.shape) == 3:
        output = np.zeros_like(img[:, :, 0])
        mask = (img[:, :, 0] >= low[0]) & (img[:, :, 0] <= high[0]) \
               & (img[:, :, 1] >= low[1]) & (img[:, :, 1] <= high[1]) \
               & (img[:, :, 2] >= low[2]) & (img[:, :, 2] <= high[2])

    output[mask] = 1
    return output

def binary_thresh(img):

    ### LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    L_max, L_mean = np.max(L), np.mean(L)
    B = lab[:, :, 2]
    B_max, B_mean = np.max(B), np.mean(B)

    # YELLOW
    L_adapt_yellow = max(80, int(L_max * 0.45))
    B_adapt_yellow = max(int(B_max * 0.70), int(B_mean * 1.2))
    lab_low_yellow = np.array((L_adapt_yellow, 120, B_adapt_yellow))
    lab_high_yellow = np.array((255, 145, 255))

    lab_yellow = binary_threshold(lab, lab_low_yellow, lab_high_yellow)
    lab_bin = lab_yellow

    ### HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]
    H_max, H_mean = np.max(H), np.mean(H)
    S = hsv[:, :, 1]
    S_max, S_mean = np.max(S), np.mean(S)
    V = hsv[:, :, 2]
    V_max, V_mean = np.max(V), np.mean(V)

    # YELLOW
    S_adapt_yellow = max(int(S_max * 0.25), int(S_mean * 1.75))
    V_adapt_yellow = max(50, int(V_mean * 1.25))
    hsv_low_yellow = np.array((15, S_adapt_yellow, V_adapt_yellow))

    hsv_high_yellow = np.array((30, 255, 255))
    hsv_yellow = binary_threshold(hsv, hsv_low_yellow, hsv_high_yellow)

    # WHITE
    V_adapt_white = max(150, int(V_max * 0.8), int(V_mean * 1.25))
    hsv_low_white = np.array((0, 0, V_adapt_white))
    hsv_high_white = np.array((255, 40, 220))

    hsv_white = binary_threshold(hsv, hsv_low_white, hsv_high_white)

    hsv_bin = hsv_yellow | hsv_white

    ### HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    L = hls[:, :, 1]
    L_max, L_mean = np.max(L), np.mean(L)
    S = hls[:, :, 2]
    S_max, S_mean = np.max(S), np.mean(S)

    # YELLOW
    L_adapt_yellow = max(80, int(L_mean * 1.25))
    S_adapt_yellow = max(int(S_max * 0.25), int(S_mean * 1.75))
    hls_low_yellow = np.array((15, L_adapt_yellow, S_adapt_yellow))
    hls_high_yellow = np.array((30, 255, 255))

    hls_yellow = binary_threshold(hls, hls_low_yellow, hls_high_yellow)

    # WHITE
    L_adapt_white = max(160, int(L_max * 0.8), int(L_mean * 1.25))
    hls_low_white = np.array((0, L_adapt_white, 0))
    hls_high_white = np.array((255, 255, 255))

    hls_white = binary_threshold(hls, hls_low_white, hls_high_white)

    hsl_bin = hls_yellow | hls_white

    ### R color channel (WHITE)
    R = img[:, :, 0]
    R_max, R_mean = np.max(R), np.mean(R)

    R_low_white = min(max(150, int(R_max * 0.55), int(R_mean * 1.95)), 230)
    R_bin = binary_threshold(R, R_low_white, 255)

    ### Adaptive thresholding: Gaussian kernel
    # YELLOW

    adapt_yellow_S = cv2.adaptiveThreshold(hls[:, :, 2], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, -5)
    adapt_yellow_B = cv2.adaptiveThreshold(lab[:, :, 2], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, -5)
    adapt_yellow = adapt_yellow_S & adapt_yellow_B

    # WHITE
    adapt_white_R = cv2.adaptiveThreshold(img[:, :, 0], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, -27)
    adapt_white_L = cv2.adaptiveThreshold(hsv[:, :, 2], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, -27)
    adapt_white = adapt_white_R & adapt_white_L

    adap_bin = adapt_yellow | adapt_white

    ### Ensemble Voting
    combined = np.asarray(R_bin + lab_bin + hsl_bin + hsv_bin + adap_bin, dtype=np.uint8)

    combined[combined < 3] = 0
    combined[combined >= 3] = 1

    return combined

def combined_thresh(img):
    absx_bin = abs_sobel_thresh(img, orient='x', thresh_min=50, thresh_max=200)
    absy_bin = abs_sobel_thresh(img, orient='y', thresh_min=50, thresh_max=200)
    mag_bin = mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255))
    dir_bin = dir_threshold(img, sobel_kernel=15, thresh=(0.8, 1.2))
    img_bin = binary_thresh(img)

    combined = np.zeros_like(dir_bin)
    combined[((absx_bin == 1) & (absy_bin == 1)) | ((mag_bin == 1) & (dir_bin == 1)) | (img_bin == 1)] = 1

    return combined

if __name__ == '__main__':
    cap = cv2.VideoCapture("video_test.mp4")
    with open('camera_calib.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']
    while True:
        ret, img = cap.read()
        shape = img.shape[:2]
        img = cv2.pyrDown(img, dstsize=(shape[1] // 2, shape[0] // 2))
        ############


        img = cv2.undistort(img, mtx, dist, None, mtx)
        combined = combined_thresh(img)
        cv2.imshow("", combined)
        if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()