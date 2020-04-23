import os
import numpy as np
import json
from PIL import Image, ImageDraw
from multiprocessing import Pool
import multiprocessing as mp
import matplotlib.pyplot as plt

def compute_convolution(I, T, stride=1):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''

    I = I.astype(np.uint32)
    T = T.astype(np.uint32)
    
    fc1 = T[:, :, 0].flatten()
    fc2 = T[:, :, 1].flatten()
    fc3 = T[:, :, 2].flatten()

    heatmap = []
    for i in range(0, len(I) - len(T), stride):
        tmp = []
        for j in range(0, len(I[i]) - len(T[0]), stride):
            cur = I[i:i + len(T), j:j + len(T[0]), :]
            c1 = np.dot(cur[:, :, 0].flatten(), fc1)
            c2 = np.dot(cur[:, :, 1].flatten(), fc2)
            c3 = np.dot(cur[:, :, 2].flatten(), fc3)
            tmp.append([c1, c2, c3])
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
            if heatmap[i][j][1] < 0.4*heatmap[i][j][0] and heatmap[i][j][2] < 0.4*heatmap[i][j][0]:
                points.append((i, j))
        
    points = np.array(points)
    if len(points) <= 30:
        # don't do anything if 20 px or less)
        return []

    bbox = []
    ctrs, conf = kmeans(points, 6)
    bsize = (120, 120)
    for c in range(len(ctrs)):
        center = ctrs[c]
        bbox.append([center[1] - bsize[0]/2, center[0] - bsize[1]/2,
                     center[1] + bsize[0]/2, center[0] + bsize[1]/2, conf[c]])
    
    return bbox

def normalize(I):
    return (I - np.min(I))/(np.max(I) - np.min(I))

def detect_red_light_mf(I, fname, ftr):
    heatmap = compute_convolution(I, ftr)
    output = predict_boxes(heatmap)

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

data_path = '../data/RedLights2011_Medium'
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

ftr = Image.open('f1.png')
ftr = ftr.resize((4, 11))
ftr = np.array(ftr)

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
