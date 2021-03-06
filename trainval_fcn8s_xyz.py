# -*- coding:utf-8 -*-
import os
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
from data import KITTIRoadRGBXYZ
import models
from configure import config
from utils import label_accuracy_score, eval_road
from utils import BooststrapBinaryCrossEntropyLoss2D
import time

# set devices
os.environ['CUDA_VISIBLE_DEVICES'] = config.device


def val(model, dataloader):
    model.eval()
    eval_acc = 0
    eval_acc_cls = 0
    eval_mean_iu = 0
    eval_fwavacc = 0
    li_pred = []
    li_gt = []
    for i, (_, _, xyz, lb) in enumerate(dataloader):
        xyz, lb = Variable(xyz), Variable(lb)
        xyz, lb = xyz.float().cuda(), lb.long().cuda()
        pred = model(xyz)
        # Mean IoU
        label_true = lb.data.cpu().numpy().astype(np.int8)
        label_pred = pred.data.cpu().numpy().squeeze(0)
        label_pred = (label_pred > 0.5).astype(np.int8)
        for (label, prob) in zip(label_true, label_pred):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(label, prob, n_class=2)
            eval_acc += acc
            eval_acc_cls += acc_cls
            eval_mean_iu += mean_iu
            eval_fwavacc += fwavacc
        # MaxF
        label_pred = pred.data.cpu().numpy().squeeze()
        label_true = lb.data.cpu().numpy().squeeze()
        li_pred.append(label_pred)
        li_gt.append(label_true)
    print 'Validation ======ACC: %lf,Mean IoU: %lf======' % (eval_acc/dataloader.__len__(),
                                                             eval_mean_iu/dataloader.__len__())
    max_f = eval_road(li_pred, li_gt)
    model.train()
    return max_f


def train():
    # step.1
    # model = getattr(models, config.model)().cuda()
    model = getattr(models, config.model)()
    model = torch.nn.DataParallel(model)    # multi-gpu
    model.cuda()
    print 'train', config.model
    # print model
    if config.load_model_path:
        model.load_state_dict(torch.load(config.load_model_path))
    # step.2
    train_data = KITTIRoadRGBXYZ(config.root, split='train')
    val_data = KITTIRoadRGBXYZ(config.root, split='val')
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
    val_dataloader = DataLoader(val_data, num_workers=config.num_workers)
    # step.3
    # binary cross entropy 2d
    criterion = BooststrapBinaryCrossEntropyLoss2D(int(config.IMG_SIZE*config.K1))

    if config.optim is 'Adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=config.weight_decay)   # optimizer
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)

    best_max_f = 0
    # step.4 train
    for epoch in range(config.max_epoch):
        for i, (_, _, xyz, lb) in enumerate(train_dataloader):
            xyz, lb = Variable(xyz), Variable(lb)
            xyz, lb = xyz.float().cuda(), lb.long().cuda()
            optimizer.zero_grad()
            # forward
            pred = model(xyz)
            loss = criterion(pred, lb)
            # backward
            loss.backward()
            optimizer.step()
            if i % config.print_freq == 0:
                print 'epoch: %03d,iter: %03d, loss=%lf' % (epoch, i, loss.data[0])

        if (epoch+1) % config.test_freq == 0:
            max_f = val(model, val_dataloader)
            if max_f > best_max_f:
                best_max_f = max_f
                print 'better model, MaxF = %f' % best_max_f
                # save the best model
                filename = '%sK_%.2f.pth' % (config.model_prefix, config.K1)
                torch.save(model.state_dict(), filename)
                print 'save %s succeed!' % filename
    return best_max_f


def eval_on_validation():
    if not os.path.exists(config.run_dir):
        os.mkdir(config.run_dir)
    model = getattr(models, config.model)()
    model = torch.nn.DataParallel(model)    # multi-gpu
    model.cuda()
    print 'test on validation set.', config.model
    print model
    if config.load_model_path:
        model.load_state_dict(torch.load(config.load_model_path))
    # data
    test_data = KITTIRoadRGBXYZ(config.root, split='val')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    # test
    model.eval()
    eval_acc = 0
    eval_acc_cls = 0
    eval_mean_iu = 0
    eval_fwavacc = 0
    li_pred = []
    li_gt = []
    total_time = 0

    for i, (name, _, xyz, lb) in enumerate(test_dataloader):
        xyz, lb = Variable(xyz), Variable(lb)
        xyz, lb = xyz.float().cuda(), lb.long().cuda()
        start = time.clock()
        pred = model(xyz)  # inference
        end = time.clock()
        total_time += (end - start)
        # save image
        label_pred = pred.data.cpu().numpy().squeeze()
        label_pred = np.array(label_pred*255, dtype=np.uint8)
        filename = os.path.join(config.run_dir, name[0])
        print filename
        # cv2.imwrite(filename, label_pred)
        # Mean IoU
        label_true = lb.data.cpu().numpy().astype(np.int8)
        label_pred = pred.data.cpu().numpy().squeeze(0)
        label_pred = (label_pred > 0.5).astype(np.int8)
        for (label, prob) in zip(label_true, label_pred):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(label, prob, n_class=2)
            eval_acc += acc
            eval_acc_cls += acc_cls
            eval_mean_iu += mean_iu
            eval_fwavacc += fwavacc
        # MaxF
        label_pred = pred.data.cpu().numpy().squeeze()
        label_true = lb.data.cpu().numpy().squeeze()
        li_pred.append(label_pred)
        li_gt.append(label_true)
    print 'Runtime ############# time(s) : %f ##########' % (total_time / test_dataloader.__len__())
    print 'Validation ======ACC: %lf,Mean IoU: %lf======' % (eval_acc / test_dataloader.__len__(),
                                                             eval_mean_iu / test_dataloader.__len__())
    eval_road(li_pred, li_gt)


if __name__ == '__main__':
    config.split = 'test'
    if config.split is 'train':
        config.model = 'FCN8s'
        # config.load_model_path = 'checkpoints/best-model-baseline@K_0.20.pth'
        config.load_model_path = ''
        config.model_prefix = 'checkpoints/best-model-fcn8s_xyz@'
        config.batch_size = 4
        config.max_epoch = 200
        config.K1 = 0.2
        print 'K = %.2f' % config.K1
        MaxF = train()
        print MaxF
    else:
        config.model = 'FCN8s'
        config.load_model_path = 'checkpoints/best-model-fcn8s_xyz@K_0.20.pth'
        config.run_dir = 'RUN/val_fcn8s_xyz'
        # eval_on_validation()
        eval_on_validation()


