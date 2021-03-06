import os
import numpy as np
import cv2
path_name = '/media/yangfei/Repository/KITTI/data_road/training/image_2'
filelist = os.listdir(path_name)
filelist.sort()
li=[]
mean = np.zeros(3, dtype=np.float32)
for filename in filelist:
    im = cv2.imread(os.path.join(path_name, filename))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im[-288:, :1216, :]/255.
    im = np.reshape(im, (-1, 3))
    li.append(im)
all_mat = np.vstack(li)
chan_mean = np.mean(all_mat, axis=0)
chan_std = np.std(all_mat, axis=0)
# torch.std()
print 'RGB mean is :', chan_mean
print 'RGB std is :', chan_std
