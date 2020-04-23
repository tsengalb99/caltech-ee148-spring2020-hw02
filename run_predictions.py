import os
import numpy as np
import json
from PIL import Image, ImageDraw
from multiprocessing import Pool
import multiprocessing as mp
import matplotlib.pyplot as plt

def compute_convolution(I, T, stride=1, padding=0):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''

    # pad I
    zs = I.shape
    zs = (zs[0] + 2 * padding, zs[1] + 2 * padding)
    padded = np.zeros(zs)
    padded[padding : I.shape[0] + padding, padding : I.shape[1] + padding] = I

    fc1 = T[:, :].flatten()

    heatmap = []
    for i in range(0, len(padded) - len(T), stride):
        tmp = []
        for j in range(0, len(padded[i]) - len(T[0]), stride):
            cur = padded[i:i + len(T), j:j + len(T[0])]
            c1 = np.dot(cur[:, :].flatten(), fc1)
            tmp.append(c1)
        heatmap.append(tmp)

    return np.array(heatmap)


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    def kmeans(data, k):
        mn = np.mean(data, 0)
        st = np.std(data, 0)
        data = (data - mn)/st
        np.random.shuffle(data)
        centers = data[:k]

        for i in range(500):
            dist = (data[:, :, None] - np.transpose(centers)[None, :, :])**2
            cls = np.argmin(np.sum(dist, axis=1), axis=1)
            nc = []
            for j in range(k):
                nc.append(np.mean(data[cls == j, :], axis=0))
            nc = np.array(nc)
            if (nc == centers).all():
                break
            centers = nc

        dist = (data[:, :, None] - np.transpose(centers)[None, :, :])**2
        cls = np.argmin(np.sum(dist, axis=1), axis=1)

        conf = []
        for i in range(k):
            dists = []
            for j in range(len(cls)):
                if cls[j] == i:
                    dists.append(dist[j])
            md = np.median(dists) + 1e-2
            std = np.std(dists) + 1e-2
            coeff = 2/(1 + np.exp(-2.5/(std*md))) - 1
            conf.append(max(min(1, coeff), 0))
        
        centers = centers * st + mn
        return list(centers), conf

    points = []
    for i in range(len(heatmap)):
        for j in range(len(heatmap[i])):
            if heatmap[i][j] > 0.9:
                points.append((i, j))
        
    points = np.array(points)
    if len(points) <= 30:
        # don't do anything if 20 px or less)
        return []

    bbox = []
    ctrs, conf = kmeans(points, 15)
    bsize = (10, 10)
    for c in range(len(ctrs)):
        center = ctrs[c]
        bbox.append([center[1] - bsize[0]/2, center[0] - bsize[1]/2,
                     center[1] + bsize[0]/2, center[0] + bsize[1]/2, conf[c]])
    
    return bbox

def normalize(I):
    return (I - np.min(I))/(np.max(I) - np.min(I))

def detect_red_light_mf(I, fname, ftr):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    I = normalize(I[:, :, 0])
    
    heatmap = normalize(compute_convolution(I, ftr[:, :, 0]))
    iters = 1
    for it in range(iters):
        heatmap = normalize(compute_convolution(heatmap, ftr[:, :, 0]))
    hmg = heatmap * 255
    tmp = Image.fromarray(hmg.astype(np.uint8), 'L')
    tmp.save(fname)
    output = predict_boxes(heatmap)

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}

ftr = Image.open('f2.png').resize((10, 10))
ftr = np.asarray(ftr)

def fnametmt(fname):
    global ftr
    # read image using PIL:
    I = Image.open(os.path.join(data_path, fname))
    # convert to numpy array:
    rdraw = ImageDraw.Draw(I)
    boxes = detect_red_light_mf(np.asarray(I), fname, ftr)
        
    for rc in boxes:
        rdraw.rectangle(rc[0:4], outline=(0, 0, int(rc[4]*255)))
    I.save(os.path.join(preds_path, fname))
    return boxes


p = Pool(mp.cpu_count())
tmp = p.map(fnametmt, file_names_train)
for i in range(len(file_names_train)):
    name = file_names_train[i]
    preds_train[name] = tmp[i]

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        preds_test[file_names_test[i]] = detect_red_light_mf(np.asarray(I))

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
