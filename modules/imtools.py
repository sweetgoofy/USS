import cv2
import numpy as np
from scipy.signal import medfilt, fftconvolve
from scipy.ndimage.filters import laplace, gaussian_filter
from scipy.optimize import leastsq
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def fit_sine(data):
    ''' Fit a least-square sine wave to data'''
    t = np.arange(len(data))
    guess_mean = np.mean(data)
    guess_phase = 0
    guess_freq = 1./len(data)
    guess_amp = np.percentile(data, 98)
    -np.percentile(data, 2)
    op_func = lambda x: x[3]*np.sin(2*np.pi*x[2]*t+x[1]) + x[0] - data
    est_mean, est_phase, est_freq, est_amp = leastsq(op_func, [guess_mean, guess_phase, guess_freq, guess_amp])[0]
    return est_amp*np.sin(est_freq*2*np.pi*t+est_phase)+est_mean

def process_img(url):
    ''' Extract water level from image'''
    raw = cv2.imread(url, 0)

    # Preprocess img to remove shot noise
    # and calculate gradients.
    img = cv2.medianBlur(raw, 31)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize= 3)

    def digitise(img):
        ''' Extract water level from gradient field'''
        ret = np.zeros(img.shape[1])
        max_indices = []

        # Generate smoothing kernels
        kernal_gauss = cv2.getGaussianKernel(img.shape[0], 20)
        kernal_gauss *= 1.0/np.max(kernal_gauss)

        # Estimate centres for smoothing kernals.
        for i in xrange(img.shape[1]):
            col = color[:,i]
            grad_avg = fftconvolve(col, [1./15 for i in xrange(15)],
                                   mode = 'same')
            max_idx = np.argmin(grad_avg)
            max_indices.append(max_idx)

        max_indices = medfilt(max_indices, 31)
        max_indices = gaussian_filter(max_indices, 55)
        cpy = max_indices
        max_indices = fit_sine(max_indices).astype('int32')

        # find edge
        for i in xrange(img.shape[1]):
            col = color[:,i]
            max_idx = max_indices[i]
            k = np.roll(kernal_gauss, max_idx - img.shape[0]/2)

            # Nullify data further from 3 times Gaussian std.
            if max_idx > img.shape[0]/2:
                k[:max_idx - 20 * 3] = 0
            else:
                k[max_idx + 20 * 3:] = 0
                
            col = col.astype('float64')*k[:,0]

            # Threshold for detecking water surface
            thresh = np.percentile(col[col<0],1)
            for j in xrange(len(col)-1, -1, -1):
                if col[j] <= thresh:
                    ret[i] = j
                    break

        # Smooth result for shot noise
        ret = medfilt(ret, 31)

        return img.shape[0] - ret

    # Now y-axis is positive up.
    ret = digitise(img)
    return ret
    
