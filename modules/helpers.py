import pickle
import os
import numpy as np

def get_urls(url):
    ''' Pick out all bmp images at url.'''
    return [url+i for i in os.listdir(url) if i.split('.')[-1]=="bmp"]


def dump(obj, url):
    ''' Write obj as binary file at url.'''
    outFile = open(url, 'wb')
    pickle.dump(obj, outFile)
    outFile.close()

def parse(url, num_sensors = 2, check = 0):
    ''' Parse USS txt record.'''
    inFile = open(url, 'r')
    inFile.readline()           # digest header
    ret =  np.array([np.array(line.split(), dtype = 'float64')[:num_sensors+1]
                     for line in inFile]).T
    def plot(data):
        x = ret[0]
        for i in xrange(num_sensors):
            plt.plot(x, ret[i+1])
        plt.show()
    if check:
        plot(ret)

    return ret

def draw_line(
        img,
        data,
        color = (255,0,0),
        skip = 10,
        thickness = 1):
    ''' Draw water line on image.'''
    data = data.astype('int32')
    for i in xrange(0, img.shape[1]-1, skip):
        if i+ skip/2 > img.shape[1]-1:break
        cv2.line(img, (i, img.shape[0] - data[i]),
                 (i+skip/2,img.shape[0] - data[i+skip/2]),
                 color, thickness = thickness)
    return img

def check_frames(
        dump_url,
        img_dir,
        codec = ['m','p','4','v'],
        fps = 30,
        res = (1366, 768),
        color = False):
    ''' Helper function to check derived data
    by plotting water level directly on images.
    '''
    data = pickle.load(open(dump_url, 'rb'))
    urls = get_urls(img_dir)
    frames = []
    v_out = cv2.VideoWriter(img_dir
                            + 'result.mp4',
                            cv2.VideoWriter_fourcc(*codec),
                            fps,
                            res,
                            isColor = color)

    for i in xrange(len(urls)):
        url = urls[i]
        im = cv2.imread(url, 0)
        im = draw_line(cv2.cvtColor(im,
                                    cv2.COLOR_GRAY2BGR),
                       data[i],
                       skip = 100,
                       thickness = 5)
        v_out.write(cv2.resize(im, res))
        print "written frame %d to file" % i
    cv2.destroyAllWindows()
    v_out.release()         

