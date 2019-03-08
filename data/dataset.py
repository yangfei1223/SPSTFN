# -*- coding:utf-8 -*-
import os
import numpy as np
import cv2
from torch.utils import data
from torchvision import transforms as T


def readKittiCalib(filename, dtype='f8'):
    '''
    :param filename:
    :param dtype:
    '''
    outdict = dict()
    output = open(filename, 'rb')
    allcontent = output.readlines()
    output.close()
    for contentRaw in allcontent:
        content = contentRaw.strip()
        if content == '':
            continue
        if content[0] != '#':
            tmp = content.split(':')
            assert len(tmp) == 2, 'wrong file format, only one : per line!'
            var = tmp[0].strip()
            values = np.array(tmp[-1].strip().split(' '), dtype)

            outdict[var] = values

    return outdict


class KittiCalibration(object):
    calib_dir = None
    calib_end = None
    R0_rect = None
    P2 = None
    Tr33 = None
    Tr = None
    Tr_cam_to_road = None

    def __init__(self):
        '''
        '''
        pass

    def readFromFile(self, filekey=None, fn=None):
        '''

        @param fn:
        '''

        if filekey != None:
            fn = os.path.join(self.calib_dir, filekey + self.calib_end)

        assert fn != None, 'Problem! fn or filekey must be != None'
        cur_calibStuff_dict = readKittiCalib(fn)
        self.setup(cur_calibStuff_dict)

    def setup(self, dictWithKittiStuff, useRect=False):
        '''

        @param dictWithKittiStuff:
        '''
        dtype_str = 'f8'
        # dtype = np.float64

        self.P2 = np.matrix(dictWithKittiStuff['P2']).reshape((3, 4))

        if useRect:
            #
            R2_1 = self.P2
            # self.R0_rect = None

        else:
            R0_rect_raw = np.array(dictWithKittiStuff['R0_rect']).reshape((3, 3))
            # Rectification Matrix
            self.R0_rect = np.matrix(
                np.hstack((np.vstack((R0_rect_raw, np.zeros((1, 3), dtype_str))), np.zeros((4, 1), dtype_str))))
            self.R0_rect[3, 3] = 1.
            # intermediate result
            R2_1 = np.dot(self.P2, self.R0_rect)

        Tr_cam_to_road_raw = np.array(dictWithKittiStuff['Tr_cam_to_road']).reshape(3, 4)
        # Transformation matrixs
        self.Tr_cam_to_road = np.matrix(np.vstack((Tr_cam_to_road_raw, np.zeros((1, 4), dtype_str))))
        self.Tr_cam_to_road[3, 3] = 1.

        self.Tr = np.dot(R2_1, self.Tr_cam_to_road.I)
        self.Tr33 = self.Tr[:, [0, 2, 3]]

    def get_matrix33(self):
        '''

        '''
        assert self.Tr33.all() != None  # --yf. modify 1
        return self.Tr33


class BirdsEyeView(object):
    IMG_HEIGHT = 288
    IMG_WIDTH = 1216
    BEV_HEIGHT = 800
    BEV_WIDTH = 400
    SAMPLE_RATE = 1
    GRID_RES = 0.05 * SAMPLE_RATE

    def __init__(self):
        pass

    def wolrd2image(self, Tr33, Xw, Zw):
        p = np.vstack((Xw, Zw, np.ones_like(Xw)))
        result = np.dot(Tr33, p)
        resultB = np.broadcast_arrays(result, result[-1, :])
        return resultB[0] / resultB[1]  # (u, v, 1)

    def transformLable2BEV(self, im, Tr33, shift):
        grid_x = np.arange(-10 + self.GRID_RES / 2, 10, self.GRID_RES)
        grid_z = np.arange(46 - self.GRID_RES / 2, 6, -self.GRID_RES)

        grid = np.meshgrid(grid_x, grid_z)
        x_mesh_vec = np.reshape(grid[0], (self.BEV_WIDTH / self.SAMPLE_RATE) * (self.BEV_HEIGHT / self.SAMPLE_RATE),
                                order='F').astype(np.float32)
        z_mesh_vec = np.reshape(grid[1], (self.BEV_WIDTH / self.SAMPLE_RATE) * (self.BEV_HEIGHT / self.SAMPLE_RATE),
                                order='F').astype(np.float32)
        LUT = self.wolrd2image(Tr33, x_mesh_vec, z_mesh_vec)[:2]
        LUT_u, LUT_v = LUT[0] / self.SAMPLE_RATE, (LUT[1] - shift) / self.SAMPLE_RATE
        selector = ((LUT_u >= 1) & (LUT_u <= im.shape[1]) & (LUT_v >= 1) & (LUT_v <= im.shape[0]))
        im_u_float = LUT_u[selector]
        im_v_float = LUT_v[selector]

        ZX_ind = (np.mgrid[1: (self.BEV_HEIGHT / self.SAMPLE_RATE) + 1, 1:(self.BEV_WIDTH / self.SAMPLE_RATE) + 1]).astype(np.int32)
        Z_ind_vec = np.reshape(ZX_ind[0], selector.shape, order='F')
        X_ind_vec = np.reshape(ZX_ind[1], selector.shape, order='F')

        bev_x_ind = Z_ind_vec[selector]
        bev_z_ind = X_ind_vec[selector]

        # nearest interpolation
        if im.shape.__len__() == 2:
            bev = np.zeros((self.BEV_HEIGHT / self.SAMPLE_RATE, self.BEV_WIDTH / self.SAMPLE_RATE), dtype=np.uint8)
            bev[bev_x_ind-1, bev_z_ind-1] = im[im_v_float.astype('u4')-1, im_u_float.astype('u4')-1]
        else:
            bev = np.zeros((self.BEV_HEIGHT / self.SAMPLE_RATE, self.BEV_WIDTH / self.SAMPLE_RATE, 3), dtype=np.uint8)
            for channel in xrange(im.shape[2]):
                bev[bev_x_ind - 1, bev_z_ind - 1, channel] = im[im_v_float.astype('u4') - 1, im_u_float.astype('u4') - 1, channel]
        return bev


class KITTIRoadFusion(data.Dataset):
    def __init__(self, root, split, num_features=3, dataset='data', sample_rate=1, return_bev=False):
        assert split in ['trainval', 'train', 'val', 'test']
        self.split = split
        self.num_features = num_features
        self.sample_rate = sample_rate
        self.ignore_index = 255
        self.size = 0
        self.calib = KittiCalibration()
        self.bev = BirdsEyeView() if return_bev else None
        filename = os.path.join(dataset, self.split + '.txt')
        if self.split in ['trainval', 'train', 'val']:
            self.train_list = []
            with open(filename, 'r') as f:
                for line in f:
                    self.train_list.append(line.strip('\n'))
            self.im_dir = os.path.join(root, 'training/image_2')
            # self.cloud_dir = os.path.join(root, 'training/velodyne_image')
            self.cloud_dir = os.path.join(root, 'training/velodyne_normal')
            self.calib_dir = os.path.join(root, 'training/calib')
            self.gt_dir = os.path.join(root, 'training/gt_image_2')
        else:
            self.test_list = []
            with open(filename, 'r') as f:
                for line in f:
                    self.test_list.append(line.strip('\n'))
            self.im_dir = os.path.join(root, 'testing/image_2')
            # self.cloud_dir = os.path.join(root, 'testing/velodyne_image')
            self.cloud_dir = os.path.join(root, 'testing/velodyne_normal')
            self.calib_dir = os.path.join(root, 'testing/calib')

        self.mean_cloud_bgr = np.array([0.32973887, 0.35007734, 0.34609209])    # BGR order
        self.std_cloud_bgr = np.array([0.24253355, 0.24843998, 0.25952055])
        self.mean = np.array([0.33053341, 0.34080286, 0.32288151])      # RGB order
        self.std = np.array([0.27192578, 0.26952331, 0.27069592])
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

    def __getitem__(self, index):
        if self.split in ['trainval', 'train', 'val']:
            category, frame = self.train_list[index].split('_')
            name = category + '_road_' + frame + '.png'
            filename = os.path.join(self.im_dir, category + '_' + frame + '.png')
            im = self.load_image(filename)
            im = self.transform(im)
            filename = os.path.join(self.cloud_dir, category + '_' + frame + '.txt')
            cloud = self.load_cloud(filename, self.size)
            filename = os.path.join(self.calib_dir, category + '_' + frame + '.txt')
            theta = self.load_calib(filename)
            shift = self.size[0] - 288
            filename = os.path.join(self.gt_dir, category + '_road_' + frame + '.png')
            lb = self.load_label(filename)
            '''
            cv2.namedWindow('lb_bev')
            cv2.imshow('lb_bev', self.bev.transformLable2BEV(lb, theta, shift)*255)
            cv2.imwrite('lb.png', self.bev.transformLable2BEV(lb, theta, shift))
            cv2.waitKey(0)
            '''
            if self.bev is not None:
                lb_bev = self.bev.transformLable2BEV(lb, theta, shift)
                # im = self.bev.transformLable2BEV(im, theta, shift)
                return name, im, cloud, theta, shift, lb, lb_bev
            else:
                return name, im, cloud, theta, shift, lb
        else:
            category, frame = self.test_list[index].split('_')
            name = category + '_road_' + frame + '.png'
            filename = os.path.join(self.im_dir, category + '_' + frame + '.png')
            im = self.load_image(filename)
            im = self.transform(im)
            filename = os.path.join(self.cloud_dir, category + '_' + frame + '.txt')
            cloud = self.load_cloud(filename, self.size)
            filename = os.path.join(self.calib_dir, category + '_' + frame + '.txt')
            theta = self.load_calib(filename)
            shift = self.size[0] - 288
            return name, im, cloud, theta, shift

    def __len__(self):
        if self.split in ['trainval', 'train', 'val']:
            return len(self.train_list)
        else:
            return len(self.test_list)

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def load_image(self, filename):
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        self.size = im.shape[0], im.shape[1]
        # cv2.imshow('img', color)
        # cv2.waitKey(0)
        im = im[-288:, :1216, :]
        # im = im[::self.sample_rate, ::self.sample_rate, :]
        return im

    def load_cloud(self, filename, size):
        cloud = np.loadtxt(filename)
        cloud[np.isnan(cloud)] = 0
        if self.split in ['trainval', 'train', 'val']:
            idx, feats, label = cloud[:, :2], cloud[:, 2:-1], cloud[:, -1]
            # feats[:, :3] = self.pc_normalize(feats[:, :3])
            # feats[:, 8:] = (feats[:, 8:] / 255.-self.mean_cloud_bgr)/self.std_cloud_bgr
            im = np.zeros((size[0], size[1], self.num_features))
            gt = np.zeros(size)
            for i in range(idx.shape[0]):
                im[int(idx[i, 0]), int(idx[i, 1])] = feats[i]
                gt[int(idx[i, 0]), int(idx[i, 1])] = label[i]
            # cv2.imshow('cloud-label', gt)
            # cv2.imshow('cloud', im[:, :, 4:7][:, :, ::-1])
            # cv2.imshow('cloud', im[:, :, 0])
            # cv2.waitKey(0)
            # save rgb
            # cv2.imwrite('rgb.png', im[:, :, 4:7][:, :, ::-1]*255)
            # save xyz
            # cv2.imwrite('xyz.png', im[:, :, 0:3]*255)
            '''
            # save normal
            x = im[:, :, 15:18]
            y = (x[x != 0]+1.)/2
            x[x != 0] = y
            cv2.imwrite('normal3.png', x*255)
            '''
            '''
            # save curvature
            x = np.log(im[:, :, 10]*200+1.)
            cv2.normalize(x, x, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite('curvature1.png', x)
            '''
            im = im.transpose((2, 0, 1))
            im = im[:, -288:, :1216]
            return im
        else:
            idx, feats = cloud[:, :2], cloud[:, 2:]
            # feats[:, :3] = self.pc_normalize(feats[:, :3])
            # feats[:, 8:] = feats[:, 8:] / 255.
            im = np.zeros((size[0], size[1], self.num_features))
            for i in range(idx.shape[0]):
                im[int(idx[i, 0]), int(idx[i, 1])] = feats[i]
            # cv2.imshow('cloud', im[:, :, -3:])
            # cv2.waitKey(0)
            im = im.transpose((2, 0, 1))
            im = im[:, -288:, :1216]
            return im

    def load_label(self, filename):
        color = cv2.imread(filename)
        # cv2.imshow('groundtruth', color)
        # cv2.waitKey(0)
        height = color.shape[0]
        width = color.shape[1]
        lb = np.zeros((height, width), dtype=np.uint8)
        for r in range(height):
            for c in range(width):
                if color[r, c, 2] > 0:  # R
                    lb[r, c] = 1 if color[r, c, 0] > 0 else 0
                else:
                    lb[r, c] = self.ignore_index
        lb = lb[-288:, :1216]
        # lb = lb[::self.sample_rate, ::self.sample_rate]
        # cv2.imshow('label', lb)
        # cv2.waitKey(0)
        '''
        # save groundtruth
        groundtruth = np.zeros_like(lb)
        groundtruth[lb == 1] = 255
        cv2.imwrite('groundtruth.png', groundtruth)
        '''
        return lb

    def load_calib(self, filename):
        self.calib.readFromFile(fn=filename)
        matrix33 = self.calib.get_matrix33()
        return np.array(matrix33)


class KITTIRoadRGBXYZ(data.Dataset):
    def __init__(self, root, split, dataset='data'):
        assert split in ['trainval', 'train', 'val', 'test']
        self.split = split
        self.ignore_index = 255
        self.size = 0
        filename = os.path.join(dataset, self.split + '.txt')
        if self.split in ['trainval', 'train', 'val']:
            self.train_list = []
            with open(filename, 'r') as f:
                for line in f:
                    self.train_list.append(line.strip('\n'))
            self.im_dir = os.path.join(root, 'training/image_2')
            self.x_dir = os.path.join(root, 'training/x')
            self.y_dir = os.path.join(root, 'training/y')
            self.z_dir = os.path.join(root, 'training/z')
            self.gt_dir = os.path.join(root, 'training/gt_image_2')
        else:
            self.test_list = []
            with open(filename, 'r') as f:
                for line in f:
                    self.test_list.append(line.strip('\n'))
            self.im_dir = os.path.join(root, 'testing/image_2')
            self.x_dir = os.path.join(root, 'testing/x')
            self.y_dir = os.path.join(root, 'testing/y')
            self.z_dir = os.path.join(root, 'testing/z')

        self.mean = np.array([0.33053341, 0.34080286, 0.32288151])      # RGB order
        self.std = np.array([0.27192578, 0.26952331, 0.27069592])
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

    def __getitem__(self, index):
        if self.split in ['trainval', 'train', 'val']:
            category, frame = self.train_list[index].split('_')
            name = category + '_road_' + frame + '.png'
            filename = os.path.join(self.im_dir, category + '_' + frame + '.png')
            im = self.load_image(filename)
            im = self.transform(im)
            filename_x = os.path.join(self.x_dir, category + '_' + frame + '.png')
            filename_y = os.path.join(self.y_dir, category + '_' + frame + '.png')
            filename_z = os.path.join(self.z_dir, category + '_' + frame + '.png')
            cloud = self.load_xyz(filename_x, filename_y, filename_z)
            filename = os.path.join(self.gt_dir, category + '_road_' + frame + '.png')
            lb = self.load_label(filename)
            return name, im, cloud, lb
        else:
            category, frame = self.test_list[index].split('_')
            name = category + '_road_' + frame + '.png'
            filename = os.path.join(self.im_dir, category + '_' + frame + '.png')
            im = self.load_image(filename)
            im = self.transform(im)
            filename_x = os.path.join(self.x_dir, category + '_' + frame + '.txt')
            filename_y = os.path.join(self.y_dir, category + '_' + frame + '.txt')
            filename_z = os.path.join(self.z_dir, category + '_' + frame + '.txt')
            cloud = self.load_xyz(filename_x, filename_y, filename_z)
            return name, im, cloud

    def __len__(self):
        if self.split in ['trainval', 'train', 'val']:
            return len(self.train_list)
        else:
            return len(self.test_list)

    def load_image(self, filename):
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # cv2.imshow('im', im)
        # cv2.waitKey(0)
        im = im[-288:, :1216, :]
        return im

    def load_xyz(self, filename_x, filename_y, filename_z):
        x = cv2.imread(filename_x, cv2.IMREAD_UNCHANGED)
        y = cv2.imread(filename_y, cv2.IMREAD_UNCHANGED)
        z = cv2.imread(filename_z, cv2.IMREAD_UNCHANGED)
        xyz = np.concatenate((x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]), axis=2)
        # cv2.imshow('xyz', xyz)
        # cv2.waitKey(0)
        xyz = xyz[-288:, :1216, :]
        xyz = np.transpose(xyz, (2, 0, 1))
        return xyz

    def load_label(self, filename):
        color = cv2.imread(filename)
        height = color.shape[0]
        width = color.shape[1]
        lb = np.zeros((height, width), dtype=np.uint8)
        for r in range(height):
            for c in range(width):
                if color[r, c, 2] > 0:  # R
                    lb[r, c] = 1 if color[r, c, 0] > 0 else 0
                else:
                    lb[r, c] = self.ignore_index
        lb = lb[-288:, :1216]
        # cv2.imshow('label', lb*255)
        # cv2.waitKey(0)
        return lb


if __name__ == '__main__':
    root = '/media/yangfei/Repository/KITTI/data_road'
    '''
    kitti = KITTIRoadFusion(root, 'val', dataset='.')
    for i in range(kitti.__len__()):
        name, im, _, _, _, _, _ = kitti[i]
        print name
    '''
    kitti = KITTIRoadRGBXYZ(root, 'trainval', dataset='.')
    name, im, xyz, lb = kitti[50]
    print 'hello world'
