import numpy as np
from modules.imtools import *
from modules.helpers import *
from modules.postprocessing import *
from modules.plotting import *

SCN_W = 1366
SCN_H = 768
IMG_DIR = r"Desktop/Gangfu free surface test/Converted_images_full/Cam_Date=160706_Time=162425 converted/"
DUMP_URL = IMG_DIR + "results.pkl"
X_SCALE = 138./1920
Y_SCALE = 86./1200


def main():
    ret = []
    count = 0
    for url in get_urls(IMG_DIR):
        ret.append(process_img(url))
        count += 1
        print "processed %s" % url
    dump(np.array(ret), DUMP_URL)
    print "successfully dumped results to %s" % DUMP_URL



if __name__ == "__main__":
    

    main()  
    
    # process()

    # check()

    # test_1()
    # uss_data = parse(r'Desktop/Aachen/160706/uss3and2_15_21_00.txt', check = 0)
    # cam_data = scale(DUMP_URL, X_SCALE, Y_SCALE, 1920, 1200)
    # # c = calc_correl(cam_data, uss_data, check = True)
    # uss_aligned = align(cam_data[1], uss_data[1:])
    # weights = calc_weight(cam_data[1], uss_aligned, s_stride = 8, slope = 111.76, offset = 174.147)
    # h, b =np.histogram(weights, 100)
    # plot((b[:-1], h), 1, 'r')
    
