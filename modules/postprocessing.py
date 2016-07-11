import numpy as np
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
import pickle
import copy
from scipy.optimize import leastsq

def scale(
    cam_data_url,
    x_scale,
    y_scale,
    x_res,
    y_res,
    invert = True):
    ''' Convert from camera to physical coordinates.
    scale: mm / px
    '''
    cam_data = (pickle.load(open(cam_data_url, 'rb'))).T
    if invert:
        cam_data = y_res - cam_data
    return (x_res * x_scale, cam_data * y_scale)

def calc_correl(cam_data,
                uss_data,
                fsamp_cam = 100,
                fsamp_uss = 50,
                centre_px = 821,
                check = False):
    ''' Correlate camera and uss data.'''
    assert (fsamp_cam % fsamp_uss == 0)
    stride = fsamp_cam / fsamp_uss
    cam_x, cam_y = cam_data

    # normalise camera data
    cam_y_mean = np.average(cam_y, axis = 1)
    cam_y_std = np.std(cam_y, axis = 1)
    for i in xrange(len(cam_y)):
        cam_y[i] -= cam_y_mean[i]
        cam_y[i] /= cam_y_std[i]

    # normalise uss data
    uss_y = uss_data[1:]
    uss_y_mean = np.average(uss_y, axis = 1)
    uss_y_std = np.std(uss_y, axis = 1)
    for i in xrange(len(uss_y)):
        uss_y[i] -= uss_y_mean[i]
        uss_y[i] /= uss_y_std[i]

    max_len = min(cam_y.shape[1]/stride,
                  len(uss_data[0]))
    if check:
        plt.plot(cam_y[820][::stride])
        plt.plot(np.roll(uss_y[0][:max_len],-13))
        plt.show()


    # Identify offset between camera and uss signals.
    c = fftconvolve(cam_y[centre_px-1][::stride][:max_len],
                    uss_y[0][:max_len][::-1],
                    mode = 'full')/max_len
    if check:
        plt.plot(np.arange(-max_len+1, max_len),c)
        plt.show()
    offset = np.argmax(c) - max_len + 1    
    ret = [np.dot(cam_y[:, ::stride][:,:max_len],
                  np.roll(uss_data_i[:max_len], offset))/max_len
           for uss_data_i in uss_y]

    if check:
        for i in xrange(len(ret)):
            plt.plot(ret[i], label = 'uss_data_%i' % (i+1))
        plt.legend()
        plt.show()

    return ret

def align(
        cam_data,
        uss_data,
        fsamp_cam = 100,
        fsamp_uss = 50,
        centre_px = 821,
        main_uss_index = 0,
        normalise = 0):
    ''' Correct misalignment between camera and uss data.
    fsamp_cam must be greater than fsamp_uss.
    '''

    # check dimensions
    assert len(cam_data.shape) == 2
    assert len(uss_data.shape) == 2
    
    # check length
    stride = fsamp_cam / fsamp_uss
    max_length = min(len(cam_data[0,::stride]),
                     len(uss_data[0]))

    # normalise
    def normalise(sig):
        ''' Normalise sig without modifying original'''
        if not normalise:
            sig_cpy = copy.copy(sig)
        else:
            sig_cpy = sig
        sig_avg = np.average(sig, axis = 1)
        sig_std = np.std(sig, axis = 1)
        for i in xrange(len(sig_cpy)):
            sig_cpy[i] -= sig_avg[i]
            sig_cpy[i] /= sig_std[i]

        return sig_cpy

    cam_data_norm = normalise(cam_data)
    uss_data_norm = normalise(uss_data)

    c = fftconvolve(cam_data_norm[centre_px - 1, ::stride][:max_length],
                    uss_data_norm[main_uss_index][:max_length][::-1])/max_length
    offset = np.argmax(c) - max_length + 1

    for sig in uss_data:
        sig = np.roll(sig, offset)

    return uss_data

def calc_weight(
        cam_data,
        uss_data,
        s_stride = 1,
        f_stride = 2,
        main_uss_index = 0,
        slope = 1,
        offset = 0):
    ''' Calculate least-square weighting vector W by 
    minimisation of (dot(cam_data.T, W) - uss_data)**2.
    '''
    
    cam_data = cam_data[::s_stride, ::f_stride]
    max_length = min(len(cam_data[0]),len(uss_data[0]))
    weight_v = [1 for i in xrange(len(cam_data))]

    # Objective function to optimise
    obj_func = lambda x: np.dot(cam_data[:,:max_length].T, np.array(x)) \
               - slope*uss_data[main_uss_index, :max_length] + offset
    weights_v = leastsq(obj_func, weight_v)[0]
    weights_v /= np.max(weights_v)
    return weights_v
